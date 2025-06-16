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
merger_ratio = 1    # 1:0.2,  2:0.8
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=True, load_vllm=True, fp16=True, use_flow_cache=stream)
prompt1_speech_16k = load_wav('asset/Tingting6_prompt.wav', 16000)
prompt2_speech_16k = load_wav('asset/zero_shot_prompt.wav', 16000)

print(prompt1_speech_16k.shape)
print(prompt2_speech_16k.shape)
print('混合比例：',merger_ratio)
if merger_ratio!=1:
    prompt1_len = prompt1_speech_16k.size(1)
    prompt2_len = prompt2_speech_16k.size(1)
    # 保持prompt1_len为较长的
    if prompt1_len<prompt2_len:
        prompt1_speech_16k, prompt2_speech_16k = prompt2_speech_16k, prompt1_speech_16k
        prompt1_len,prompt2_len=prompt2_len,prompt1_len
    extend_len = int(merger_ratio/(1-merger_ratio) * prompt2_len)
    if extend_len<=prompt1_len:
        extent_prompt1_speech_16k = prompt1_speech_16k[:,:extend_len]
    else:
        repeat_count = int(np.ceil(extend_len / prompt1_len))  # 计算需要重复的次数
        extended_prompt = prompt1_speech_16k.repeat(1, repeat_count)  # 重复以扩展长度
        extent_prompt1_speech_16k = extended_prompt[:, :extend_len]  # 截断到所需长度
    prompt_speech_16k = torch.cat([prompt2_speech_16k,extent_prompt1_speech_16k],dim=1)
else:
    prompt_speech_16k = prompt1_speech_16k



resample_rate = 24000
prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
speech_feat, speech_feat_len = cosyvoice.frontend._extract_speech_feat(prompt_speech_resample)        # prompt语音提取梅尔频谱特征 (1,1113,80) (1)
speech_token, speech_token_len = cosyvoice.frontend._extract_speech_token(prompt_speech_16k)          # prompt语音2token (1,557) (1)
# cosyvoice2, force speech_feat % speech_token = 2
token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])       # 556
speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len     # 调整speech_feat，speech_feat_len
speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len          # 调整speech_token，speech_token_len

embedding = cosyvoice.frontend._extract_spk_embedding(prompt_speech_16k)      #  prompt语音提取语音深度的embedding (1,192)

model_input = {'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                'llm_embedding': embedding, 'flow_embedding': embedding}


for i in range(10):
    for j in cosyvoice.my_inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用丰富的情感表达' + '<|endofprompt|>', model_input, stream=True,speed=1):
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

                prompt_text = ''
                if text_type != '':
                    prompt_text += f'用{text_type}说出这句话' 
                elif emotion != '':
                    prompt_text += f'用{emotion}的情感表达' 
                prompt_text += '<|endofprompt|>'

                if text:
                    print('合成的文本:', text)
                    print('指定的语种:', text_type)
                    print('指定的情绪:', emotion)
                    print('prompt_text:', prompt_text)
                    #await websocket.send_bytes(construct_binary_message(ACK=True))
                    for idx, j in enumerate(cosyvoice.my_inference_instruct2(text, prompt_text, model_input, stream=stream,speed=speed_ratio)): 
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
    uvicorn.run(app, host="0.0.0.0", port=7860)

    
