import torch
from vllm import LLM, SamplingParams
model_path = 'pretrained_models/qwen2-tts-0.5B'
input_text = 'sos_eos用愉快的情感表达<|endofprompt|>有一天我路过街边的小吃摊,我终于买到了我想吃的蛋糕task_id'

'''
from vllm.inputs.data import TokensPrompt
from transformers import AutoTokenizer,AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_path) 
transformers_model = AutoModelForCausalLM.from_pretrained(model_path)
input_ids = tokenizer.encode(input_text, return_tensors='pt')
embedding_layer = transformers_model.get_input_embeddings()
prompt_embeds = embedding_layer(input_ids).squeeze(0)
print(prompt_embeds.shape)

#embed_prompt: EmbedsPrompt = {"prompt_embeds": prompt_embeds,}
token_prompt:TokensPrompt = {"prompt_embeds": prompt_embeds,}'''
sampling_params = SamplingParams(
    n=1,                    # 每个提示生成1个序列
    temperature=1.0,        # 控制随机性
    top_p=0.8,             # 考虑所有token的累积概率
    max_tokens=154,         # 最大生成长度
    min_tokens=1,          # 最小生成长度
    repetition_penalty=1.0, # 重复惩罚
    presence_penalty=0.0,   # 存在惩罚
    frequency_penalty=0.0,  # 频率惩罚
    skip_special_tokens=False,  # 跳过特殊token
    spaces_between_special_tokens=False,  # 特殊token之间添加空格
)

llm = LLM(model=model_path,
            tokenizer_mode='auto',
            trust_remote_code=False,
            enforce_eager=False,
            enable_prefix_caching=True,
            gpu_memory_utilization=1,
            max_model_len=2048,       
            tensor_parallel_size=1,
            #enable_prompt_embeds=True,
            )



outputs = llm.generate(input_text, sampling_params=sampling_params)
print(outputs)
print(outputs[0].outputs[0].token_ids)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}\nGenerated text: {generated_text}")

'''
conda create -n myvllm python=3.12 -y
docker start -ai cosyvoice
docker exec -it cosyvoice /bin/bash

cd /cosyvoice
conda activate newenv
pip install vllm
docker cp vllm/vllm/model_executor/models/registry.py cosyvoice:/opt/conda/envs/newenv/lib/python3.12/site-packages/vllm/model_executor/models/
docker cp vllm/vllm/model_executor/models/qwen2_tts.py cosyvoice:/opt/conda/envs/newenv/lib/python3.12/site-packages/vllm/model_executor/models/

[3677, 6486, 4299, 4218, 3453, 3238, 146, 524, 1052, 5669, 5101, 1122, 4758, 3327, 2726, 4463, 5189, 4537, 840, 2919, 5832, 3402, 2032, 2113, 2149, 4373, 623, 3880, 1585, 5096, 5830, 663, 2914, 2297, 231, 5650, 5641, 4849, 4594, 1920, 4607, 4578, 5853, 4541, 4444, 4677, 737, 1217, 1188, 6024, 6051, 2290, 228, 2217, 5832, 5843, 5266, 4537, 3000, 4299, 6162, 6159, 3891, 3969, 4134, 3402, 1303, 4299, 2113, 2186, 644, 884, 1896, 5244, 1659, 3611, 5062, 5058, 5317, 4993, 4526, 3096, 4311, 4376, 1463, 2222, 1923, 2243, 6158, 4862, 6230, 2738, 1373, 1348, 2891, 4495, 4675, 4390, 1460, 910, 1770, 5322, 5236, 948, 2332, 2380, 1842, 4487, 5998, 202, 4075, 1032, 2300, 5429, 5509, 5508, 4862, 1975, 2112, 4299, 6486]
[3700, 6563, 6563, 6563, 6563, 6563, 4012, 1024, 6563, 6563, 6563, 1050, 2074, 3675, 5026, 1986, 
1482, 2031, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 1824, 6563, 6563, 6563, 6563, 6563, 110, 6563, 
6563, 6563, 1788, 4299, 4490, 6563, 3283, 6563, 6563, 6563, 2031, 3646, 6563, 6563, 6563, 6563, 2139, 1950, 6563, 6563, 6563, 6563, 6563, 4218, 6563, 6563, 6563, 6563, 3890, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 6563, 1951, 6563, 6563, 6563, 6563, 6563, 4246, 6563, 6563, 6563, 6563, 6563, 6563, 4299, 1978, 6563, 2058, 6563, 6563, 5050, 6563, 6563, 4138, 4171, 6563, 6563, 6563, 3174, 5006, 6563, 6563, 6563, 6563, 6486, 1797, 3455, 6563, 6563, 4299, 3405, 6563, 6563, 6563, 2031, 4733, 6563, 6563, 6563, 6563, 2166, 6563, 6563, 6563, 
6563, 4299, 1948, 6563, 6563, 6563, 6563, 2112, 3930, 6563, 6563, 6563, 226]

'''