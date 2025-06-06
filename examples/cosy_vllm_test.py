import sys

import torchaudio
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

def main():
    set_all_random_seed(999)
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=False, fp16=True)

    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
        print(j['tts_speech'],len(j['tts_speech'][0]))
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


if __name__=='__main__':
    main()

"""
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
sudo apt-get install sox libsox-dev


conda create -n cosyvoice_vllm --clone cosyvoice\
conda activate cosyvoice_vllm
pip install vllm==v0.9.0 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
conda activate cosyvoice_vllm
python vllm_example.py
"""
