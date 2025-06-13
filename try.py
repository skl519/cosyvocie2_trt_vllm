import numpy as np
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import sys
sys.path.append('third_party/Matcha-TTS')
try:
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
except:
    pass
merger_ratio = 1
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=True, load_vllm=True, fp16=True,use_flow_cache=True)
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



# 开始语音合成
speech_list = []
#for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=True)):
for j in cosyvoice.my_inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用丰富的情感表达' + '<|endofprompt|>', model_input, stream=True,speed=1):
    
    speech_tensor = j['tts_speech']
    # 确保所有片段的采样率一致
    if cosyvoice.sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(orig_freq=cosyvoice.sample_rate, new_freq=24000)
        speech_tensor = resampler(speech_tensor)
    speech_list.append(speech_tensor)

# 合并所有音频片段
if speech_list:
    merged_speech = torch.cat(speech_list, dim=1)  # 合并在时间维度上
    torchaudio.save('instruct_merged.wav', merged_speech, 24000)
    print("✅ 成功保存合并音频: instruct_merged.wav")
else:
    print("❌ 没有生成任何音频片段")