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
        "voice_type": "愤怒",
        "encoding": "mp3",
        "speed_ratio": 1.0,
        "volume_ratio": 1.0,
        "pitch_ratio": 1.0,
    },
    "request": {
        "reqid": "xxx",
        "text": "看到这一幕，我气得浑身发抖，拳头攥得紧紧的，恨不得冲上去质问他们为什么要这样破坏别人辛苦建立的一切。", # '我收到录取通知书的那一刻,我高兴得跳起来'
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
    #file_to_save = open("test_submit.mp3", "wb")
    header = {"Authorization": f"Bearer; {token}"}
    async with websockets.connect(api_url, extra_headers=header, ping_interval=None) as ws:
        await ws.send(full_client_request)
        all_audio_data = []  # 用于存储接收到的音频数据
        try:
            while True:
                t1 = time.time()
                res = await ws.recv()
                print(f'延迟：{time.time()-t1}')
                done = parse_response(res, all_audio_data)  # 修改为接收音频数据
                print(done)
                if done:
                    break
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by the server.")
        finally:
            # 保存接收到的音频数据
            if all_audio_data:
                combined_audio = np.concatenate(all_audio_data)  # 合并所有音频数据
                torchaudio.save("output_api.wav", torch.from_numpy(combined_audio).unsqueeze(0), 24000)  # 保存为 WAV 文件
            print("\nclosing the connection...")


async def test_query():
    query_request_json = copy.deepcopy(request_json)
    query_request_json["audio"]["voice_type"] = voice_type
    query_request_json["request"]["reqid"] = str(uuid.uuid4())
    query_request_json["request"]["operation"] = "query"
    payload_bytes = str.encode(json.dumps(query_request_json))
    payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
    full_client_request = bytearray(default_header)
    full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
    full_client_request.extend(payload_bytes)  # payload
    print("\n------------------------ test 'query' -------------------------")
    print("request json: ", query_request_json)
    print("\nrequest bytes: ", full_client_request)
    file_to_save = open("test_query.mp3", "wb")
    header = {"Authorization": f"Bearer; {token}"}
    async with websockets.connect(api_url, extra_headers=header, ping_interval=None) as ws:
        await ws.send(full_client_request)
        res = await ws.recv()
        parse_response(res, file_to_save)
        file_to_save.close()
        print("\nclosing the connection...")


def parse_response(res, audio_data_list):
    print("--------------------------- response ---------------------------")
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
    print(f"            Protocol version: {protocol_version:#x} - version {protocol_version}")
    print(f"                 Header size: {header_size:#x} - {header_size * 4} bytes ")
    print(f"                Message type: {message_type:#x} - {MESSAGE_TYPES[message_type]}")
    print(f" Message type specific flags: {message_type_specific_flags:#x} - {MESSAGE_TYPE_SPECIFIC_FLAGS[message_type_specific_flags]}")
    print(f"Message serialization method: {serialization_method:#x} - {MESSAGE_SERIALIZATION_METHODS[serialization_method]}")
    print(f"         Message compression: {message_compression:#x} - {MESSAGE_COMPRESSIONS[message_compression]}")
    print(f"                    Reserved: {reserved:#04x}")
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
    #loop.run_until_complete(test_query())

"""

git:  添加到暂存区-》 提交到本地库 -> 推送到github中
变基:  Git 中用于重写历史的一种操作，通常用来将一个分支的修改整合到另一个分支上，并保持一个线性的提交历史
签出:  Git 用来切换分支或恢复文件的操作

初始化仓库
    git init：在当前目录下创建一个新的Git仓库。
工作区、暂存区与提交
    git status：查看文件的状态（是否已暂存、未暂存或未跟踪）。
    git add [file]：将文件添加到暂存区。
    git add .：将所有更改添加到暂存区。
    git commit -m "commit message"：提交暂存区的内容到本地仓库，并附上提交信息。
查看历史记录
    git log：显示提交历史。
    git log --oneline：以简洁的一行格式显示提交历史。
分支操作
    git branch：列出所有本地分支，并标出当前所在的分支。
    git branch [branch-name]：创建一个新分支。
    git checkout [branch-name]：切换到指定分支。
    git checkout -b [branch-name]：创建并切换到新分支。
    git merge [branch]：合并指定分支到当前分支。
远程仓库操作
    git remote add [remote-name] [url]：添加远程仓库。
    git remote -v 
    git push [remote-name] [本地 branch-name]：推送本地分支的更新到远程仓库。
    git pull [remote-name] [本地 branch-name]：从远程仓库获取最新内容并与本地仓库合并。
    git clone [url]：从远程仓库克隆项目到本地。
撤销操作
    git reset [file]：将暂存区中的文件恢复至HEAD状态，但保留工作区的更改。
    git checkout -- [file]：丢弃工作区的更改，恢复至HEAD状态。
    git revert [commit]：创建新的提交来撤销某次特定的提交。
标签操作
    git tag [tag-name]：为当前分支创建一个标签。
    git show [tag-name]：显示某个标签的具体信息

推送：
    # 1. 查看状态
    git status
    # 2. 添加所有更改到暂存区
    git add .
    # 3. 提交更改
    git commit -m "完成 fireredasr vllm 加速 功能"
    # 4. 拉取远程最新代码
    git pull origin main
    # 6. 推送代码到远程仓库
    git push origin master

git rm test.txt
git commit -m "删除 test.txt 文件"
git push origin main

"""