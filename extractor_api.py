# import torch
# import torch.nn.functional as F
# import yaml
# from transformers import AutoTokenizer
#
# # 导入你项目中重构好的 SA-2 SBASC 核心模型
# # 如果你的模型类名不同，请修改此处导入
# from models.SBASC.model import BERTLinear as SBASC_Model
#
#
# class YelpFeatureExtractor:
#     """
#     工业级无状态情感抽取引擎 (Stateless Sentiment Extraction API)
#     已升级：支持连续型浮点数 (Continuous Floats) 与 极性抵消算法
#     """
#
#     def __init__(self, model_pt_path, config_yaml_path):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"🚀 初始化提取引擎... 使用计算设备: {self.device}")
#
#         # 1. 加载 YAML 领域本体配置 (5大通用维度)
#         with open(config_yaml_path, 'r', encoding='utf-8') as f:
#             self.cfg = yaml.safe_load(f)
#
#         self.aspects = self.cfg['aspect_category_mapper']
#         self.num_aspects = len(self.aspects)
#         self.bert_model_name = self.cfg['bert_mapper']
#
#         # ⭐️ 置信度拦截阈值：如果模型最高预测概率低于此值，强制视为未提及 (0.0)
#         self.aspect_threshold = 0.45
#
#         # 2. 初始化 Tokenizer 和 Model
#         self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
#
#         # 实例化模型 (这里的参数请与你实际的 BERTLinear 构造函数对齐)
#         # 实例化模型 (去除参数名，直接按位置顺序传入)
#         # 🚨 因为之前保存的是整个模型对象，所以直接用 torch.load 即可。
#         # 还要加上 weights_only=False 消除警告，防止它拦截整个对象的反序列化。
#         self.model = torch.load(model_pt_path, map_location=self.device, weights_only=False)
#         self.model.to(self.device)
#         self.model.eval()
#
#     def extract_batch(self, texts):
#         """
#         批量推理接口：将文本列表转化为 5 维浮点数情感特征矩阵
#         """
#         if not texts:
#             return []
#
#         # 批量 Tokenization，限制最大长度以提高运行速度
#         encoded = self.tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=256,
#             return_tensors='pt'
#         ).to(self.device)
#
#         with torch.no_grad():
#
#             with torch.no_grad():
#                 # 【核心修复 1】：生成纯一维的占位标签 [batch_size]
#                 batch_size = encoded['input_ids'].size(0)
#                 dummy_labels_cat = torch.zeros((batch_size,), dtype=torch.long).to(self.device)
#                 dummy_labels_pol = torch.zeros((batch_size,), dtype=torch.long).to(self.device)
#
#                 # 获取模型输出 (传入输入和假标签)
#                 outputs = self.model(
#                     input_ids=encoded['input_ids'],
#                     attention_mask=encoded['attention_mask'],
#                     labels_cat=dummy_labels_cat,
#                     labels_pol=dummy_labels_pol
#                 )
#
#                 # 兼容模型返回格式 (传入 labels 后，模型通常会多计算并返回一个假 loss，凑成 3 个元素)
#                 if isinstance(outputs, tuple):
#                     if len(outputs) == 3:
#                         _dummy_loss, logits_aspect, logits_sentiment = outputs  # 丢弃无用的假 loss
#                     elif len(outputs) == 2:
#                         logits_aspect, logits_sentiment = outputs
#                     else:
#                         raise ValueError(f"模型输出格式异常，元素个数: {len(outputs)}")
#                 else:
#                     raise ValueError("模型输出格式不匹配，预期返回 Tuple")
#
#             # 激活函数转换
#             probs_aspect = torch.sigmoid(logits_aspect)
#             probs_sentiment = F.softmax(logits_sentiment, dim=-1)
#
#         batch_results = []
#
#         # 遍历 Batch 中的每一行数据，进行极性抵消运算
#         for i in range(len(texts)):
#             row_features = {}
#             for j, aspect_name in enumerate(self.aspects):
#
#                 # 【机制 1：双重置信度拦截】未过阈值，直接为 0.0
#                 if probs_aspect[i, j].item() < self.aspect_threshold:
#                     row_features[aspect_name] = 0.0
#                 else:
#                     # 【机制 2：极性抵消算法】
#                     # 假定 SA-2 中 Sentiment index: 0 是 Negative, 1 是 Positive
#                     neg_prob = probs_sentiment[i, 0].item()
#                     pos_prob = probs_sentiment[i, 1].item()
#
#                     # 最终得分 = 正向概率 - 负向概率
#                     score = pos_prob - neg_prob
#
#                     # 【机制 3：高精度截断】保留 4 位小数
#                     row_features[aspect_name] = round(score, 4)
#
#             batch_results.append(row_features)
#
#         return batch_results

import torch
import torch.nn.functional as F
import yaml
import re
from transformers import AutoTokenizer


class YelpFeatureExtractor:
    """
    工业级无状态情感抽取引擎 (Stateless Sentiment Extraction API)
    已升级：引入“亚句子级解析 (Sub-Sentence Parsing)”机制，彻底解决多维度情感同分问题。
    """

    def __init__(self, model_pt_path, config_yaml_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 初始化细粒度提取引擎... 使用计算设备: {self.device}")

        # 1. 加载 YAML 领域本体配置
        with open(config_yaml_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        self.aspects = self.cfg['aspect_category_mapper']
        self.num_aspects = len(self.aspects)
        self.bert_model_name = self.cfg['bert_mapper']

        # ⭐️ 置信度拦截阈值：单句中某个维度的概率低于此值，视为未提及
        self.aspect_threshold = 0.45

        # 2. 初始化 Tokenizer 和 Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)

        # 加载完整模型
        self.model = torch.load(model_pt_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

    def extract_batch(self, texts):
        """
        批量推理接口：将评论列表进行切句解析，再聚合为高维独立特征矩阵
        """
        if not texts:
            return []

        # ====================================================
        # 1. 细粒度切句 (Sub-Sentence Parsing)
        # ====================================================
        all_sentences = []
        sentence_to_review_idx = []

        for i, text in enumerate(texts):
            # 按标点符号切分评论为单句，过滤掉过短的无意义字符
            sentences = [s.strip() for s in re.split(r'[.!?;\n]+', text) if len(s.strip()) > 3]

            # 兜底设计：如果切分后为空（例如用户发了一堆乱码），给个占位符维持物理行号对齐
            if not sentences:
                sentences = ["neutral"]

            for s in sentences:
                all_sentences.append(s)
                sentence_to_review_idx.append(i)

        # ====================================================
        # 2. 分批次通过大模型 (内部微批处理防 OOM)
        # ====================================================
        inner_batch_size = 128
        all_probs_aspect = []
        all_probs_sentiment = []

        with torch.no_grad():
            for idx in range(0, len(all_sentences), inner_batch_size):
                batch_sents = all_sentences[idx: idx + inner_batch_size]

                encoded = self.tokenizer(
                    batch_sents,
                    padding=True,
                    truncation=True,
                    max_length=128,  # 因为已经切成了单句，128 长度绝对够用，且极大提速
                    return_tensors='pt'
                ).to(self.device)

                # 生成维度匹配的假标签 (Dummy Labels)
                bsz = encoded['input_ids'].size(0)
                dummy_cat = torch.zeros((bsz,), dtype=torch.long).to(self.device)
                dummy_pol = torch.zeros((bsz,), dtype=torch.long).to(self.device)

                outputs = self.model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    labels_cat=dummy_cat,
                    labels_pol=dummy_pol
                )

                # 提取模型真实的预测输出
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:
                        _, logits_aspect, logits_sentiment = outputs
                    elif len(outputs) == 2:
                        logits_aspect, logits_sentiment = outputs

                probs_aspect = torch.sigmoid(logits_aspect)
                probs_sentiment = F.softmax(logits_sentiment, dim=-1)

                all_probs_aspect.append(probs_aspect)
                all_probs_sentiment.append(probs_sentiment)

        # 拼接小批次结果矩阵
        all_probs_aspect = torch.cat(all_probs_aspect, dim=0)
        all_probs_sentiment = torch.cat(all_probs_sentiment, dim=0)

        # ====================================================
        # 3. 动态聚合与极性抵消 (Dynamic Aggregation)
        # ====================================================
        # 初始化存放分数的字典: review_scores[第i行评论][某个维度] = [得分1, 得分2, ...]
        review_scores = {i: {asp: [] for asp in self.aspects} for i in range(len(texts))}

        for k, rev_idx in enumerate(sentence_to_review_idx):
            for j, aspect_name in enumerate(self.aspects):

                # 如果该短句提到了该维度，就计算独立的极性得分
                if all_probs_aspect[k, j].item() >= self.aspect_threshold:
                    neg_prob = all_probs_sentiment[k, 0].item()
                    pos_prob = all_probs_sentiment[k, 1].item()

                    # 极性抵消算法
                    score = pos_prob - neg_prob
                    review_scores[rev_idx][aspect_name].append(score)

        # ====================================================
        # 4. 生成最终单行 CSV 特征
        # ====================================================
        batch_results = []
        for i in range(len(texts)):
            row_features = {}
            for aspect_name in self.aspects:
                scores = review_scores[i][aspect_name]
                if scores:
                    # 求出这篇评论中，针对这个维度的所有短句的情感平均得分
                    avg_score = sum(scores) / len(scores)
                    row_features[aspect_name] = round(avg_score, 4)
                else:
                    # 该评论的所有句子都没提到这个维度
                    row_features[aspect_name] = 0.0

            batch_results.append(row_features)

        return batch_results