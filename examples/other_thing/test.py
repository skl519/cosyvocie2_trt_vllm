from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "D:/FireRedASR-main/pretrained_models/FireRedASR-LLM-L/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 输入文本
input_text = "收到拒信的那一刻，我感到无比伤心。虽然知道失败是成长的一部分，但仍然难以掩饰心中的失落"

text = f'''
要求：直接的回答output
任务：我正在做一个语音生成的任务。需要判断一段文本应该使用什么语气来表达，请对于我给出的可能富有语气的文本，生成应该使用什么语气的提示词
示例：
    input: 早上，我收到了一封邮件，打开一看，竟然是我一直梦寐以求的公司发来的录用通知
    output: 用惊讶的语气说
input: {input_text}
output: 
'''

# 生成输出
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)


#modelscope download --model Qwen/Qwen3-0.6B --local_dir C:\Users\Administrator\Desktop\CosyVoice\pretrained_models\Qwen3-0.6B
#modelscope download --model Qwen/Qwen3-1.7B --local_dir C:\Users\Administrator\Desktop\CosyVoice\pretrained_models\Qwen3-1.7B

