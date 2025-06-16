#coding=utf-8

'''
requires Python 3.6 or later

pip install asyncio
pip install websockets

'''

import asyncio
import time
import torchaudio
import websockets
import uuid
import json
import gzip
import copy
import numpy as np
import torch

MESSAGE_TYPES = {11: "audio-only server response", 12: "frontend server response", 15: "error message from server"}
MESSAGE_TYPE_SPECIFIC_FLAGS = {0: "no sequence number", 1: "sequence number > 0", 2: "last message from server (seq < 0)", 3: "sequence number < 0"}
MESSAGE_SERIALIZATION_METHODS = {0: "no serialization", 1: "JSON", 15: "custom type"}
MESSAGE_COMPRESSIONS = {0: "no compression", 1: "gzip", 15: "custom compression method"}

appid = "xxx"
token = "xxx"
cluster = "xxx"
voice_type = "愤怒"
host = "192.168.1.89:7860"
api_url = f"ws://{host}/api/v1/tts/ws_binary"

# version: b0001 (4 bits)
# header size: b0001 (4 bits)

# message type: b0001 (Full client request) (4bits)
# message type specific flags: b0000 (none) (4bits)

# message serialization method: b0001 (JSON) (4 bits)
# message compression: b0001 (gzip) (4bits)
# reserved data: 0x00 (1 byte)

default_header = bytearray(b'\x11\x10\x11\x00')

request_json = {
    "app": {
        "appid": appid,
        "token": "access_token",
        "cluster": cluster
    },
    "user": {
        "uid": "388808087185088"
    },
    "audio": {
        "voice_type": "丰富",
        "encoding": "mp3",
        "speed_ratio": 1.0,
        "volume_ratio": 1.0,
        "pitch_ratio": 1.0,
    },
    "request": {
        "reqid": "xxx",
        "text": """作家莫言在演讲时讲过一段经历：
                    在他上学的时候，一次学校组织去参观展览，同学们在看过展览后放声大哭。
                    有的人明明没哭，为了迎合大家，还悄悄地将唾沫抹到脸上冒充泪水。
                    环顾一周发现，有位同学脸上竟没有一滴眼泪，他看着大家，眼神里都是困惑和惊讶。
                    展览结束后，十几位同学都去将这位同学的"与众不同"报告了老师。
                    很多年过去了，再次回想起这件事，莫言突然明白了一个道理："当众人都哭时，应该允许有的人不哭。"
                    世人千万种，各有各的悲欢，各有各的选择，永远别用自己的情感准则，去要求别人哭或笑。""",
        "text_type": "plain",
        "operation": "xxx"
    }
}


async def test_submit():
    submit_request_json = copy.deepcopy(request_json)
    submit_request_json["request"]["reqid"] = str(uuid.uuid4())
    submit_request_json["request"]["operation"] = "submit"
    payload_bytes = str.encode(json.dumps(submit_request_json))
    payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
    full_client_request = bytearray(default_header)
    full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
    full_client_request.extend(payload_bytes)  # payload
    print("\n------------------------ test 'submit' -------------------------")
    print("request json: ", submit_request_json)
    print("\nrequest bytes: ", full_client_request)
    
    header = {"Authorization": f"Bearer; {token}"}
    async with websockets.connect(api_url, extra_headers=header, ping_interval=None) as ws:
        await ws.send(full_client_request)
        all_audio_data = []  # 用于存储接收到的音频数据
        total_delay = 0  # 累积延迟
        try:
            seq = 1
            while True:
                t1 = time.time()
                res = await ws.recv()
                delay = time.time() - t1
                total_delay += delay  # 累积延迟
                print(f'第{seq}个包 延迟：{delay}')
                done = parse_response(res, all_audio_data)  # 修改为接收音频数据
                seq += 1
                if done:
                    break
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by the server.")
        finally:
            # 保存接收到的音频数据
            if all_audio_data:
                combined_audio = np.concatenate(all_audio_data)  # 合并所有音频数据
                torchaudio.save("output_api.wav", torch.from_numpy(combined_audio).unsqueeze(0), 24000)  # 保存为 WAV 文件
                
                # 计算合成音频的时长
                audio_length = combined_audio.shape[0] / 24000  # 24000 是采样率
                print(f'总延迟：{total_delay}')  # 打印总延迟
                print(f'合成音频时长：{audio_length:.2f} 秒')
                
                # 计算 RTF
                if total_delay > 0:
                    rtf =  total_delay / audio_length
                    print(f'RTF:{rtf:.2f}')
                else:
                    print('总延迟为零，无法计算 RTF。')
            
            print("\nclosing the connection...")



def parse_response(res, audio_data_list):
    #print("--------------------------- response ---------------------------")
    # print(f"response raw bytes: {res}")
    protocol_version = res[0] >> 4                  # 0b0001
    header_size = res[0] & 0x0f                     # 报头大小*4 整个报文字段为4个字节
    message_type = res[1] >> 4                      # 
    message_type_specific_flags = res[1] & 0x0f     # 0: 没有数据, 1: 有数据, 2: 最后一个数据, 3: "sequence number < 0"
    serialization_method = res[2] >> 4              # 序列化方法
    message_compression = res[2] & 0x0f             # 压缩方法
    reserved = res[3]                               # 保留字段，同时作为边界 (使整个报头大小为4个字节).
    header_extensions = res[4:header_size*4]        # 
    payload = res[header_size*4:]
    #print(f"            Protocol version: {protocol_version:#x} - version {protocol_version}")
    #print(f"                 Header size: {header_size:#x} - {header_size * 4} bytes ")
    #print(f"                Message type: {message_type:#x} - {MESSAGE_TYPES[message_type]}")
    #print(f" Message type specific flags: {message_type_specific_flags:#x} - {MESSAGE_TYPE_SPECIFIC_FLAGS[message_type_specific_flags]}")
    #print(f"Message serialization method: {serialization_method:#x} - {MESSAGE_SERIALIZATION_METHODS[serialization_method]}")
    #print(f"         Message compression: {message_compression:#x} - {MESSAGE_COMPRESSIONS[message_compression]}")
    #print(f"                    Reserved: {reserved:#04x}")
    if header_size != 1:
        print(f"           Header extensions: {header_extensions}")
    if message_type == 0xb:  # audio-only server response
        if message_type_specific_flags == 1:  # 有数据
            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload = payload[8:]
            audio_data_list.append(np.frombuffer(payload, dtype=np.float32))  # 将音频数据添加到列表中
        elif message_type_specific_flags == 0:  # no sequence number as ACK
            print("                Payload size: 0")
            return False
        if sequence_number == 0: # 最后一个包退出
            return True
        else:
            return False
    elif message_type == 0xf:
        code = int.from_bytes(payload[:4], "big", signed=False)
        msg_size = int.from_bytes(payload[4:8], "big", signed=False)
        error_msg = payload[8:]
        if message_compression == 1:
            error_msg = gzip.decompress(error_msg)
        error_msg = str(error_msg, "utf-8")
        print(f"          Error message code: {code}")
        print(f"          Error message size: {msg_size} bytes")
        print(f"               Error message: {error_msg}")
        return True
    elif message_type == 0xc:
        msg_size = int.from_bytes(payload[:4], "big", signed=False)
        payload = payload[4:]
        if message_compression == 1:
            payload = gzip.decompress(payload)
        print(f"            Frontend message: {payload}")
    else:
        print("undefined message type!")
        return True


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_submit())