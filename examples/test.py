
import sys
import torch
from transformers import AutoTokenizer,Qwen2ForCausalLM
# 
input_text = 'sos_eos用愉快的情感表达<|endofprompt|>有一天我路过街边的小吃摊,我终于买到了我想吃的蛋糕task_id'
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('pretrained_models/qwen2-tts-0.5B')

a = tokenizer.encode(input_text,return_tensors='pt')
print(a)



sys.exit()
a = tokenizer.encode('sos_eos',return_tensors='pt')
b = tokenizer.encode('task_id',return_tensors='pt')
print(a,b)


print("词汇表大小:", len(tokenizer))    # vocab_size  151936    151935 151936
print("特殊标记:", tokenizer.special_tokens_map)    # 特殊标记: {'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>']}


# 加载模型
model = Qwen2ForCausalLM.from_pretrained('pretrained_models/qwen2-tts-0.5B')
print(model)
embeddings = model.get_input_embeddings()
print(embeddings)   # Embedding(151936, 896)

t1 = embeddings(tokenizer.encode('task_id',return_tensors='pt'))
print(t1)
t2  = embeddings.weight.data[151647]
print(t2)

# 进行值的对比
if torch.all(torch.eq(t1, t2)):
    print("t1 和 t2 的值相等")
else:
    print("t1 和 t2 的值不相等")

origin_llm_emdedding = torch.load('pretrained_models/llm_embeding.pt')

# 进行值的对比
if torch.all(torch.eq(t1, origin_llm_emdedding[1])):
    print("t1 和 t2 的值相等")
else:
    print("t1 和 t2 的值不相等")


#embeddings.weight.data[151646] = 
#embeddings.weight.data[151647] = 



"""
词汇表大小: 151646 
max  151642
  "<|endoftext|>": 151643,
  "<|im_end|>": 151645,
  "<|im_start|>": 151644,
  "sos_eos": 151646,
  "task_id": 151647
"""