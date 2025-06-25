#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
情感检测模块
支持多种小型预训练模型进行文本情感分析
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import logging
import time


class EmotionDetector:
    """情感检测器类"""
    
    def __init__(self, model_type="roberta"):
        self.models = {}
        self.tokenizers = {}
        self.default_model_type = model_type
        # 初始化时只加载指定模型
        self.load_model(model_type)
    
    def load_model(self, model_type="roberta"):
        """加载指定类型的模型"""
        if model_type in self.models:
            return  # 模型已加载
        
        try:
            if model_type == "roberta":
                model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
                self.tokenizers[model_type] = AutoTokenizer.from_pretrained(model_name)
                self.models[model_type] = AutoModelForSequenceClassification.from_pretrained(model_name)
                
            elif model_type == "qwen":
                model_name = "Qwen/Qwen2-0.5B-Instruct"
                self.tokenizers[model_type] = AutoTokenizer.from_pretrained(model_name)
                self.models[model_type] = AutoModelForCausalLM.from_pretrained(model_name)
                
            elif model_type == "chinese_roberta":
                model_name = "uer/roberta-base-finetuned-chinanews-chinese"
                self.tokenizers[model_type] = AutoTokenizer.from_pretrained(model_name)
                self.models[model_type] = AutoModelForSequenceClassification.from_pretrained(model_name)
                
            elif model_type == "distilbert":
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.tokenizers[model_type] = AutoTokenizer.from_pretrained(model_name)
                self.models[model_type] = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            logging.info(f"成功加载情感检测模型: {model_type}")
            
        except Exception as e:
            logging.error(f"加载情感检测模型 {model_type} 失败: {e}")
            raise
    
    def detect_emotion_roberta(self, text):
        """使用RoBERTa模型进行情感检测"""
        try:
            tokenizer = self.tokenizers["roberta"]
            model = self.models["roberta"]
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
            
            # 更新情感映射以包含更多情绪状态
            emotion_mapping = {0: "悲伤", 1: "丰富", 2: "高兴", 3: "愤怒", 4: "惊讶", 5: "厌恶", 6: "恐惧"}
            return emotion_mapping.get(predicted_class, "丰富")
            
        except Exception as e:
            logging.warning(f"RoBERTa情感检测失败: {e}")
            return "丰富"
    
    def detect_emotion_qwen(self, text):
        """使用Qwen小模型进行情感检测"""
        try:
            tokenizer = self.tokenizers["qwen"]
            model = self.models["qwen"]
            
            prompt = f"""请分析以下文本的情感，只回答一个词（愤怒/高兴/悲伤/平静/温柔/严肃）：
文本：{text}
情感："""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 提取情感词
            emotions = ["愤怒", "高兴", "悲伤", "平静", "温柔", "严肃"]
            for emotion in emotions:
                if emotion in response:
                    return emotion
            
            return "平静"
            
        except Exception as e:
            logging.warning(f"Qwen情感检测失败: {e}")
            return "平静"
    
    def detect_emotion_chinese_roberta(self, text):
        """使用中文RoBERTa模型进行情感检测"""
        try:
            tokenizer = self.tokenizers["chinese_roberta"]
            model = self.models["chinese_roberta"]
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
            
            # 简化的情感映射
            emotion_mapping = {0: "悲伤", 1: "高兴"}  # 根据具体模型调整
            return emotion_mapping.get(predicted_class, "平静")
            
        except Exception as e:
            logging.warning(f"中文RoBERTa情感检测失败: {e}")
            return "平静"
    
    def detect_emotion(self, text):
        """统一的情感检测接口"""
        model_type = self.default_model_type
        if model_type == "qwen":
            return self.detect_emotion_qwen(text)
        elif model_type == "chinese_roberta":
            return self.detect_emotion_chinese_roberta(text)
        else:
            return self.detect_emotion_roberta(text)

if __name__ == "__main__":
    emotion_detector_instance = EmotionDetector("qwen")
    print(emotion_detector_instance.detect_emotion("我非常愤怒！这件事让我无法忍受！"))