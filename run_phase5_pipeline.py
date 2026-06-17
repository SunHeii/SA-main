import pandas as pd
import re
import os
from tqdm import tqdm
from extractor_api import YelpFeatureExtractor

# ==========================================
# ⚙️ 核心配置区 (请根据你的实际路径调整)
# ==========================================
# 输入数据路径
CSV_INTERACTION_PATH = "/home/mysjz/mywork/SA-2-main/data/general_poi/interaction_aligned.csv"  # 第一阶段生成的无特征原始交互表
TXT_WITH_ID_PATH = "/home/mysjz/mywork/SA-2-main/data/general_poi/inference_with_id.txt"  # 第一阶段生成的带 [rev_id] 的纯文本

# 模型与配置路径
MODEL_WEIGHT_PATH = "/home/mysjz/mywork/SA-2-main/data/general_poi/model_final.pth"  # 你通过 run.py 训练出的最优权重
YAML_CONFIG_PATH = "/home/mysjz/mywork/SA-2-main/conf/domain/general_poi.yaml"  # 刚才定义的 5 维通用配置文件

# 输出数据路径
OUTPUT_FINAL_CSV = "/home/mysjz/mywork/SA-2-main/data/sid_with_sentiment.csv"  # 最终包含浮点数特征的融合大表

# GPU 批处理大小 (显存如果不够可以改小到 64 或 32)
BATCH_SIZE = 128


# ==========================================

def run_pipeline():
    print("🚀 启动交互级连续情感特征提取与绝对物理并表流水线...")

    # 1. 加载抽取引擎
    if not os.path.exists(MODEL_WEIGHT_PATH):
        raise FileNotFoundError(f"找不到权重文件: {MODEL_WEIGHT_PATH}，请确保 run.py 训练成功！")

    extractor = YelpFeatureExtractor(
        model_pt_path=MODEL_WEIGHT_PATH,
        config_yaml_path=YAML_CONFIG_PATH
    )

    # 2. 读取文本数据 (剥离 ID 以防止污染推理)
    print(f"\n📂 正在读取文本数据: {TXT_WITH_ID_PATH}")
    texts = []

    with open(TXT_WITH_ID_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            # 利用正则仅剥离行首的 [rev_id]，保留纯净文本
            match = re.match(r'^\[.*?\]\s*(.*)', line)
            if match:
                texts.append(match.group(1).strip() or "neutral text")
            else:
                # 兼容防错位兜底
                texts.append(line.strip() or "neutral text")

    total_lines = len(texts)
    print(f"✅ 共读取 {total_lines} 条待推理评论。")

    # 3. 批量推理
    print("\n🧠 正在进行深度神经网络批量推理 (极性抵消计算)...")
    all_sentiment_features = []

    for i in tqdm(range(0, total_lines, BATCH_SIZE), desc="GPU Inference"):
        batch_texts = texts[i: i + BATCH_SIZE]
        batch_features = extractor.extract_batch(batch_texts)
        all_sentiment_features.extend(batch_features)

    print(f"✅ 推理完成！成功提取 {len(all_sentiment_features)} 条浮点数特征向量。")

    # 4. 读取原始 CSV 并执行物理对齐并表
    print(f"\n🔗 正在读取交互 CSV 执行 O(1) 绝对物理挂载: {CSV_INTERACTION_PATH}")
    df_csv = pd.read_csv(CSV_INTERACTION_PATH)

    # 【核心安全墙】强制校验行数
    if len(df_csv) != len(all_sentiment_features):
        raise ValueError(
            f"🚨 致命错误：行数对不齐！\n"
            f"CSV 行数: {len(df_csv)}  |  文本提取行数: {len(all_sentiment_features)}\n"
            f"这会导致特征发生错位，程序已安全终止。"
        )

    # 将字典列表转化为 Pandas DataFrame (这 5 列会自动成为 float64 类型)
    df_sentiments = pd.DataFrame(all_sentiment_features)

    # 执行物理横向拼接
    df_final = pd.concat([df_csv, df_sentiments], axis=1)

    # 5. 结果落盘
    df_final.to_csv(OUTPUT_FINAL_CSV, index=False)
    print(f"\n🎉 大功告成！全链路成功闭环！")
    print(f"📦 包含小数特征的最终数据已保存至: {OUTPUT_FINAL_CSV}")
    print(f"📊 最终表头为:\n {df_final.columns.tolist()}")


if __name__ == "__main__":
    run_pipeline()