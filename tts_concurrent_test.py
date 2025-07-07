#coding=utf-8

import asyncio
import time
import websockets
import uuid
import json
import gzip
import copy
import numpy as np
import torch
import torchaudio
from collections import defaultdict
from datetime import datetime
import argparse
import statistics

# 协议常量定义
MESSAGE_TYPES = {11: "audio-only server response", 12: "frontend server response", 15: "error message from server"}
MESSAGE_TYPE_SPECIFIC_FLAGS = {0: "no sequence number", 1: "sequence number > 0", 2: "last message from server (seq < 0)", 3: "sequence number < 0"}
MESSAGE_SERIALIZATION_METHODS = {0: "no serialization", 1: "JSON", 15: "custom type"}
MESSAGE_COMPRESSIONS = {0: "no compression", 1: "gzip", 15: "custom compression method"}

# 默认配置
DEFAULT_CONFIG = {
    "appid": "xxx",
    "token": "xxx",
    "cluster": "xxx",
    "host": "192.168.1.89:7860",
    "emotions": ["平静", "愤怒", "愉快", "悲伤", "惊讶", "丰富"],
    "test_texts": [
        "在这个快速发展的时代，科技的力量正在改变着我们的生活方式，人工智能技术的不断进步为人类带来了前所未有的便利和机遇。",
        "春天来了，万物复苏，花儿开放，鸟儿歌唱，大地呈现出一片生机勃勃的景象，让人心情愉悦，充满希望和活力。",
        "面对困难和挑战，我们要保持乐观的心态，勇敢地面对，相信自己的能力，坚持不懈地努力，最终一定能够取得成功。",
        "友谊是人生中最珍贵的财富，真正的朋友会在你最需要的时候给予支持和帮助，让你感受到温暖和力量。",
        "学习是一个终身的过程，只有不断地学习和进步，才能跟上时代的步伐，实现自己的梦想和目标。"
    ]
}

# 默认请求头
default_header = bytearray(b'\x11\x10\x11\x00')

def generate_request_payload(text, emotion="平静", text_type="", config=None):
    """生成请求负载"""
    if config is None:
        config = DEFAULT_CONFIG
        
    request_json = {
        "app": {
            "appid": config["appid"],
            "token": "access_token",
            "cluster": config["cluster"]
        },
        "user": {
            "uid": "388808087185088"
        },
        "audio": {
            "voice_type": emotion,
            "encoding": "mp3",
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
            "auto_emotion": True,
            "emotion_model": "roberta"
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": text,
            "text_type": text_type,
            "operation": "submit"
        }
    }
    
    payload_bytes = str.encode(json.dumps(request_json))
    payload_bytes = gzip.compress(payload_bytes)
    
    full_client_request = bytearray(default_header)
    full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
    full_client_request.extend(payload_bytes)
    
    return bytes(full_client_request)

def parse_response(res, audio_data_list=None):
    """解析服务器响应"""
    if audio_data_list is None:
        audio_data_list = []
        
    try:
        header_size = res[0] & 0x0f
        message_type = res[1] >> 4
        message_type_specific_flags = res[1] & 0x0f
        payload = res[header_size*4:]
        
        result = {
            "message_type": message_type,
            "flags": message_type_specific_flags,
            "payload_msg": None,
            "payload_sequence": None,
            "audio_data": None,
            "error": None,
            "done": False
        }
        
        if message_type == 0xb:  # audio-only server response
            if message_type_specific_flags == 1:  # 有数据
                sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                audio_payload = payload[8:]
                audio_data = np.frombuffer(audio_payload, dtype=np.float32)
                audio_data_list.append(audio_data)
                
                result["payload_sequence"] = sequence_number
                result["audio_data"] = audio_data
                result["payload_msg"] = f"音频数据包 seq={sequence_number}, size={payload_size}"
                
                if sequence_number == 0:  # 最后一个包
                    result["done"] = True
                    
            elif message_type_specific_flags == 0:  # ACK
                result["payload_msg"] = "ACK"
                
        elif message_type == 0xf:  # 错误消息
            code = int.from_bytes(payload[:4], "big", signed=False)
            msg_size = int.from_bytes(payload[4:8], "big", signed=False)
            error_msg = payload[8:]
            try:
                error_msg = gzip.decompress(error_msg).decode('utf-8')
            except:
                error_msg = error_msg.decode('utf-8', errors='ignore')
            
            result["error"] = f"错误码: {code}, 消息: {error_msg}"
            result["payload_msg"] = result["error"]
            result["done"] = True
            
        elif message_type == 0xc:  # frontend server response
            msg_size = int.from_bytes(payload[:4], "big", signed=False)
            payload = payload[4:]
            try:
                payload = gzip.decompress(payload)
            except:
                pass
            result["payload_msg"] = f"前端响应: {payload.decode('utf-8', errors='ignore')}"
            
        return result
        
    except Exception as e:
        return {
            "message_type": -1,
            "flags": -1,
            "payload_msg": f"解析错误: {str(e)}",
            "payload_sequence": None,
            "audio_data": None,
            "error": str(e),
            "done": True
        }

async def single_tts_client_test(client_id, config, stats, test_text=None, emotion=None):
    """单个TTS客户端测试"""
    start_time = time.time()
    
    # 随机选择测试文本和情感
    if test_text is None:
        test_text = config["test_texts"][client_id % len(config["test_texts"])]
    if emotion is None:
        emotion = config["emotions"][client_id % len(config["emotions"])]
    
    api_url = f"ws://{config['host']}/api/v1/tts/ws_binary"
    header = {"Authorization": f"Bearer; {config['token']}"}
    
    client_stats = {
        "client_id": client_id,
        "text": test_text,
        "emotion": emotion,
        "text_length": len(test_text),
        "start_time": start_time,
        "first_packet_time": None,
        "end_time": None,
        "total_time": 0,
        "first_packet_latency": 0,
        "audio_length": 0,
        "rtf": 0,
        "packet_count": 0,
        "success": False,
        "error": None,
        "audio_data": []
    }
    
    try:
        # 生成请求
        request_payload = generate_request_payload(test_text, emotion, "", config)
        
        async with websockets.connect(api_url, extra_headers=header, ping_interval=None, max_size=100*1024*1024) as ws:
            # 发送请求
            await ws.send(request_payload)
            stats['requests_sent'] += 1
            
            # 接收响应
            all_audio_data = []
            
            while True:
                try:
                    packet_start_time = time.time()
                    res = await ws.recv()
                    packet_latency = (time.time() - packet_start_time) * 1000  # ms
                    
                    # 记录首包时间
                    if client_stats["first_packet_time"] is None:
                        client_stats["first_packet_time"] = time.time()
                        client_stats["first_packet_latency"] = client_stats["first_packet_time"] - start_time
                    
                    # 解析响应
                    result = parse_response(res, all_audio_data)
                    client_stats["packet_count"] += 1
                    
                    # 记录延迟
                    stats['latencies'].append(packet_latency)
                    
                    # 打印调试信息
                    if result["payload_msg"]:
                        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 客户端{client_id}({emotion}): {result['payload_msg']}")
                    
                    # 检查是否完成
                    if result["done"]:
                        if result["error"]:
                            client_stats["error"] = result["error"]
                            stats['errors']['server_error'] += 1
                        else:
                            client_stats["success"] = True
                            stats['success'] += 1
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    client_stats["error"] = "连接被服务器关闭"
                    stats['errors']['connection_closed'] += 1
                    break
                except Exception as e:
                    client_stats["error"] = f"接收数据异常: {str(e)}"
                    stats['errors']['receive_error'] += 1
                    break
            
            # 计算最终统计
            client_stats["end_time"] = time.time()
            client_stats["total_time"] = client_stats["end_time"] - start_time
            
            if all_audio_data and client_stats["success"]:
                combined_audio = np.concatenate(all_audio_data)
                client_stats["audio_length"] = combined_audio.shape[0] / 24000  # 24kHz采样率
                client_stats["rtf"] = client_stats["total_time"] / client_stats["audio_length"] if client_stats["audio_length"] > 0 else 0
                client_stats["audio_data"] = combined_audio
                
                # 可选：保存音频文件
                # output_path = f"asset/concurrent_test_{client_id}_{emotion}.wav"
                # torchaudio.save(output_path, torch.from_numpy(combined_audio).unsqueeze(0), 24000)
                
                print(f"客户端{client_id}({emotion}) 完成: 时长={client_stats['total_time']:.2f}s, RTF={client_stats['rtf']:.2f}, 首包延迟={client_stats['first_packet_latency']:.3f}s")
            
    except Exception as e:
        client_stats["error"] = f"连接异常: {str(e)}"
        client_stats["end_time"] = time.time()
        client_stats["total_time"] = client_stats["end_time"] - start_time
        stats['errors']['connection_error'] += 1
        print(f"客户端{client_id}({emotion}) 失败: {str(e)}")
    
    return client_stats

async def run_concurrent_tts_test(concurrency=10, duration=300, config=None):
    """运行并发TTS测试"""
    if config is None:
        config = DEFAULT_CONFIG
    
    stats = {
        'requests_sent': 0,
        'success': 0,
        'latencies': [],
        'errors': defaultdict(int),
        'start_time': time.time(),
        'client_results': []
    }
    
    print(f"启动TTS并发测试: {concurrency}个并发客户端")
    print(f"测试服务器: {config['host']}")
    print(f"最大测试时长: {duration}秒")
    print(f"测试文本数量: {len(config['test_texts'])}")
    print(f"情感类型: {', '.join(config['emotions'])}")
    print("-" * 80)
    
    # 创建并发任务
    tasks = []
    for i in range(concurrency):
        task = asyncio.create_task(single_tts_client_test(i+1, config, stats))
        tasks.append(task)
    
    # 等待所有任务完成或超时
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), 
            timeout=duration
        )
        stats['client_results'] = [r for r in results if isinstance(r, dict)]
        
    except asyncio.TimeoutError:
        print("警告：测试超时，强制终止未完成的任务")
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # 收集已完成的结果
        completed_results = []
        for task in tasks:
            if task.done() and not task.cancelled():
                try:
                    result = task.result()
                    if isinstance(result, dict):
                        completed_results.append(result)
                except:
                    pass
        stats['client_results'] = completed_results
    
    # 生成测试报告
    generate_test_report(stats, concurrency)

def generate_test_report(stats, concurrency):
    """生成详细的测试报告"""
    total_time = time.time() - stats['start_time']
    total_requests = stats['requests_sent']
    successful_requests = stats['success']
    failed_requests = sum(stats['errors'].values())
    
    print(f"\n{'='*80}")
    print(f"TTS并发测试报告")
    print(f"{'='*80}")
    print(f"测试配置:")
    print(f"  并发数: {concurrency}")
    print(f"  总测试时间: {total_time:.2f}秒")
    print(f"  请求统计: 发送={total_requests}, 成功={successful_requests}, 失败={failed_requests}")
    print(f"  成功率: {(successful_requests/total_requests*100):.1f}%" if total_requests > 0 else "  成功率: 0%")
    print(f"  QPS: {successful_requests/total_time:.2f} 请求/秒")
    
    # 延迟统计
    if stats['latencies']:
        sorted_latencies = sorted(stats['latencies'])
        print(f"\n网络延迟分布(ms):")
        print(f"  平均值: {statistics.mean(sorted_latencies):.2f}")
        print(f"  中位数: {statistics.median(sorted_latencies):.2f}")
        print(f"  P90: {sorted_latencies[int(len(sorted_latencies)*0.9)]:.2f}")
        print(f"  P95: {sorted_latencies[int(len(sorted_latencies)*0.95)]:.2f}")
        print(f"  P99: {sorted_latencies[int(len(sorted_latencies)*0.99)]:.2f}")
        print(f"  最大值: {max(sorted_latencies):.2f}")
        print(f"  最小值: {min(sorted_latencies):.2f}")
    
    # 成功请求的详细分析
    successful_clients = [r for r in stats['client_results'] if r.get('success', False)]
    if successful_clients:
        total_times = [r['total_time'] for r in successful_clients]
        rtfs = [r['rtf'] for r in successful_clients]
        first_packet_latencies = [r['first_packet_latency'] for r in successful_clients]
        audio_lengths = [r['audio_length'] for r in successful_clients]
        
        print(f"\n成功请求性能分析:")
        print(f"  总响应时间(s): 平均={statistics.mean(total_times):.2f}, 中位数={statistics.median(total_times):.2f}, 最大={max(total_times):.2f}")
        print(f"  RTF: 平均={statistics.mean(rtfs):.2f}, 中位数={statistics.median(rtfs):.2f}, 最大={max(rtfs):.2f}")
        print(f"  首包延迟(s): 平均={statistics.mean(first_packet_latencies):.3f}, 中位数={statistics.median(first_packet_latencies):.3f}, 最大={max(first_packet_latencies):.3f}")
        print(f"  音频时长(s): 平均={statistics.mean(audio_lengths):.2f}, 中位数={statistics.median(audio_lengths):.2f}")
        
        # 按情感分组统计
        emotion_stats = defaultdict(list)
        for client in successful_clients:
            emotion_stats[client['emotion']].append(client)
        
        print(f"\n按情感分组统计:")
        for emotion, clients in emotion_stats.items():
            emotion_rtfs = [c['rtf'] for c in clients]
            emotion_times = [c['total_time'] for c in clients]
            print(f"  {emotion}: 数量={len(clients)}, 平均RTF={statistics.mean(emotion_rtfs):.2f}, 平均时间={statistics.mean(emotion_times):.2f}s")
    
    # 错误统计
    if stats['errors']:
        print(f"\n错误统计:")
        for error_type, count in stats['errors'].items():
            print(f"  {error_type}: {count}次")
    
    # 失败请求详情
    failed_clients = [r for r in stats['client_results'] if not r.get('success', False)]
    if failed_clients:
        print(f"\n失败请求详情:")
        for client in failed_clients[:10]:  # 只显示前10个
            print(f"  客户端{client['client_id']}({client['emotion']}): {client.get('error', '未知错误')}")
        if len(failed_clients) > 10:
            print(f"  ... 还有{len(failed_clients)-10}个失败请求")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TTS并发压力测试工具')
    parser.add_argument('-c', '--concurrency', type=int, default=10, help='并发连接数 (默认: 10)')
    parser.add_argument('-t', '--time', type=int, default=300, help='最大测试时长(秒) (默认: 300)')
    parser.add_argument('--host', default='192.168.1.89:7860', help='服务器地址 (默认: 192.168.1.89:7860)')
    parser.add_argument('--token', default='xxx', help='认证令牌')
    parser.add_argument('--emotions', nargs='+', help='指定测试的情感类型')
    
    args = parser.parse_args()
    
    # 构建配置
    config = DEFAULT_CONFIG.copy()
    config['host'] = args.host
    config['token'] = args.token
    if args.emotions:
        config['emotions'] = args.emotions
    
    print("CosyVoice TTS 并发压力测试工具")
    print("=" * 80)
    
    # 运行测试
    asyncio.run(run_concurrent_tts_test(args.concurrency, args.time, config))

if __name__ == "__main__":
    main() 