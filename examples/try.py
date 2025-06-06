import sys

import torch
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio # type: ignore

stream = True
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True,load_vllm=True, fp16=True, use_flow_cache=stream)

# instruct usage
prompt_speech_16k = load_wav('asset/Tingting6_prompt.wav', 16000)
print(prompt_speech_16k.shape)
cosyvoice.model.tts_test()
all_speech = []
for i, j in enumerate(cosyvoice.inference_instruct2("有一天我路过街边的小吃摊,我终于买到了我想吃的蛋糕", '用愉快的情感表达', prompt_speech_16k, stream=stream)):
    all_speech.append(j['tts_speech'])

# '用愉快的情感表达'



#for i, j in enumerate(cosyvoice.inference_instruct2("我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。", '用丰富的情感表达', prompt_speech_16k, stream=stream)):
    #all_speech.append(j['tts_speech'])


if all_speech:
    combined_speech = torch.cat(all_speech, dim=1)
    torchaudio.save('output.wav', combined_speech, cosyvoice.sample_rate)
    print(f"已保存完整音频到 output.wav")
    










"""
docker run --gpus=all -it --name cosyvoice -v C:/Users/Administrator/Desktop/CosyVoice:/cosyvoice -p 7860:7860 cosyvoice-image
conda create -n cosyvoice -y python=3.10

docker exec -it cosyvoice /bin/bash
cd /cosyvoice
conda activate cosyvoice
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
sudo apt-get install sox libsox-dev
python test1.py


modelscope download --model iic/CosyVoice2-0.5B --local_dir pretrained_models
pip install matcha-tts
cd third_party/Matcha-TTS
git clone https://github.com/shivammehta25/Matcha-TTS



conda activate cosyvoice
python try.py


cd runtime/python
docker build -t cosyvoice:v1.0 .
# change iic/CosyVoice-300M to iic/CosyVoice-300M-Instruct if you want to use instruct inference
# for grpc usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M && sleep infinity"
cd grpc && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
# for fastapi usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
cd fastapi && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
"""