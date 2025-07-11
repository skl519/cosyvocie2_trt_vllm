import json
from fastapi import FastAPI, WebSocket
import gzip
import sys
from typing import Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastapi.websockets import WebSocketState
import torchaudio
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

app = FastAPI()
set_all_random_seed(1024)
stream = True
auto_emotion = False    
DEFAULT_EMOTION_MODEL = "roberta"  # 默认使用roberta模型 (可选: roberta, qwen, chinese_roberta, none)
merger_ratio = 1 # 1:0.2,  2:0.8


# 模型加载，嵌入混合
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True, use_flow_cache=stream)
prompt1_speech_16k = load_wav('asset/zero_shot_promptaa.wav', 16000)
prompt2_speech_16k = load_wav('asset/zero_shot_promptaa.wav', 16000)

print(prompt1_speech_16k.shape)
print(prompt2_speech_16k.shape)
print('混合比例：', merger_ratio)

# === 嵌入级加权混合：仅对说话人嵌入做线性插值 ===
# 选用一段提示音频（取更长的一段）作为 zero-shot 提示语音
prompt_speech_16k = prompt1_speech_16k if prompt1_speech_16k.size(1) >= prompt2_speech_16k.size(1) else prompt2_speech_16k
zero_shot_prompta='哈哈，太棒了，我们终于成功完成了这个项目。奥，我知道了，明天早上八点在老地方集合是吧？哎，别提了，今天上班真是累死我了'
zero_shot_promptb='刁维列已被另案处理。油炸豆腐喷喷香，馓子麻花嘣嘣脆，姊妹团子数二姜。'
# 情感检测 加载
emotion_detector_instance = None
if auto_emotion:
    from cosyvoice.emotion_detector import EmotionDetector
    emotion_detector_instance = EmotionDetector(DEFAULT_EMOTION_MODEL)


# 模型预热
warmup_lengths = [1, 10, 20, 30, 40, 50, 60, 70, 80]
template_text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
num_runs_per_length = 3

for length in warmup_lengths:
    input_text = (template_text * (length // len(template_text) + 1))[:length]
    print(f"Warming up with text length: {len(input_text)}")
    for _ in range(num_runs_per_length):
        for i in cosyvoice.inference_zero_shot(input_text, zero_shot_prompta, prompt_speech_16k, stream=True, speed=1):
            pass


def construct_binary_message(payload: bytes = None,sequence_number: int = None, ACK=False) -> bytes:
    header =  bytearray(b'\x11\xb0\x11\x00') if ACK else bytearray(b'\x11\xb1\x11\x00')
    if sequence_number: # 等于0，表示最后一个包，大于0表示有数据
        header.extend(sequence_number.to_bytes(4, 'big'))
    if payload:
        header.extend(len(payload).to_bytes(4, 'big'))
        header.extend(payload)
    return bytes(header)

@app.websocket("/api/v1/tts/ws_binary")
async def tts_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_bytes()
            request_data = parse_client_request(message)
            if request_data:
                text = request_data.get("request", {}).get("text", "")
                text_type = request_data.get("request", {}).get("text_type", "")
                emotion = request_data.get('audio', {}).get('voice_type', '')
                speed_ratio = request_data.get('audio', {}).get('speed_ratio', 1.0)
                
                prompt_text = zero_shot_prompta
                # if text_type != '':
                #     prompt_text += f'用{text_type}说出这句话' 
                # elif emotion != '':
                #     prompt_text += f'用{emotion}的情感表达' 
                # prompt_text += '<|endofprompt|>'

                if text:
                    print('合成的文本:', text)
                    print('指定的语种:', text_type)
                    print('指定的情绪:', emotion)
                    print('prompt_text:', prompt_text)
                    #await websocket.send_bytes(construct_binary_message(ACK=True))
                    for idx, j in enumerate(cosyvoice.inference_zero_shot(
                        text,
                        prompt_text,
                        prompt_speech_16k,
                        stream=stream,
                        speed=speed_ratio,
                        text_frontend=True
                    )): 
                        speech = j['tts_speech']
                        audio_bytes = speech.cpu().numpy().tobytes()
                        audio_message = construct_binary_message(payload=audio_bytes, sequence_number=idx + 1)
                        await websocket.send_bytes(audio_message)
                    await websocket.send_bytes(construct_binary_message(sequence_number=0))  # 发送结束信号
                    break  # 发送完毕后退出循环
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 确保在关闭连接之前不再发送消息
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

def parse_client_request(message: bytes) -> Optional[dict]:
    header_size = (message[0] & 0x0f) * 4                   # 报头大小*4 整个报文字段为4个字节
    payload_size = int.from_bytes(message[header_size:header_size + 4], 'big')
    payload = message[header_size + 4:header_size + 4 + payload_size]
    payload = gzip.decompress(payload)
    return json.loads(payload.decode('utf-8'))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7868)

    
