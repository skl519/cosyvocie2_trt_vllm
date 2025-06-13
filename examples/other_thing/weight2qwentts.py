import sys
import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from safetensors.torch import save_model
# 加载原始模型
model = Qwen2ForCausalLM.from_pretrained('pretrained_models/qwen2-0.5B')
speech_token_size = 6561

model.model.llm_embedding = torch.nn.Embedding(2, model.config.hidden_size)
model.llm_decoder =nn.Linear(model.config.hidden_size, speech_token_size + 3)
print(model)

# 打印模型结构
print("Original Model Parameters:")
for name, param in model.named_parameters():
    print(f"{name}   {param.shape}")

# 加载 llm.pt 中的参数
llm_state_dict = torch.load('pretrained_models/CosyVoice2-0.5B/llm.pt')

new_llm_state_dict = {}
# 遍历 llm.pt 中的参数
for name, param in llm_state_dict.items():
    name = name.replace('llm.model.','')
    if name in model.state_dict():
        # 如果参数存在，覆盖
        print(f"Overriding parameter: {name} with shape {param.shape}")
        new_llm_state_dict[name] = param
    elif name =='llm_embedding.weight':
        # 如果参数不存在，添加
        print(f"Adding new parameter: {name} with shape {param.shape}")
        new_llm_state_dict['model.'+name] = param
    elif 'llm_decoder' in name:
        new_llm_state_dict[name] = param

new_llm_state_dict['model.embed_tokens.weight'][151646] = new_llm_state_dict['model.llm_embedding.weight'][0]
new_llm_state_dict['model.embed_tokens.weight'][151647] = new_llm_state_dict['model.llm_embedding.weight'][1]

torch.save( new_llm_state_dict['model.llm_embedding.weight'],' llm_embeding.pt')

model.load_state_dict(new_llm_state_dict)
model.save_pretrained('pretrained_models/qwen2-tts-0.5B')

import json

with open('pretrained_models/qwen2-tts-0.5B/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

config['architectures'] = ['Qwen2TTSForCausalLM']
config['_name_or_path'] = ['pretrained_models/qwen2-tts-0.5B']
