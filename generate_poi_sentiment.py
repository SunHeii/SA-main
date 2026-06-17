# import pandas as pd
# import numpy as np
# import os
# import time
#
#
# def generate_robust_poi_sentiment(
#         input_csv_path,
#         output_csv_path,
#         sentiment_cols=['service', 'environment', 'price', 'location', 'core_experience'],
#         half_life_days=365.0,
#         confidence_c=10.0
# ):
#     """
#     工业级 POI 情感聚合器 (时间衰减 + 贝叶斯平滑)
#
#     参数:
#     - input_csv_path: 包含交互级别情感特征的 CSV (如 interaction_aligned_with_sentiment.csv)
#     - output_csv_path: 输出的 POI 静态特征字典路径
#     - sentiment_cols: 需要聚合的情感特征列名
#     - half_life_days: 时间半衰期(天)。默认 365 天(一年前的评论效力衰减为现在的50%)
#     - confidence_c: 贝叶斯平滑收缩系数。默认 10.0 (相当于需要10篇最新评论的权重，才能摆脱全局平均分)
#     """
#     print("启动高鲁棒性 POI 情感聚合引擎...")
#     start_time = time.time()
#
#     # 1. 安全读取数据
#     if not os.path.exists(input_csv_path):
#         raise FileNotFoundError(f"找不到输入文件：{input_csv_path}")
#
#     print(f"正在读取交互数据: {input_csv_path}")
#     df = pd.read_csv(input_csv_path)
#
#     # 统一列名大小写容错 (如果你之前的列名是小写，这里自动适配)
#     col_mapping = {c.lower(): c for c in df.columns}
#     df.rename(columns={c: col_mapping.get(c.lower(), c) for c in sentiment_cols}, inplace=True)
#
#     # 检查情感列是否存在
#     for col in sentiment_cols:
#         if col not in df.columns:
#             raise KeyError(f"数据表中缺失必需的情感列: {col}")
#
#     # 2. 时间解析与衰减权重计算 (Time-Decay Weighting)
#     print(f"正在计算指数时间衰减 (半衰期: {half_life_days} 天)...")
#     # 将时间转换为 datetime 对象，并处理可能的时区问题
#     df['Time'] = pd.to_datetime(df['Time'], utc=True)
#
#     # 获取全局最新时间作为基准点 (T_ref)
#     t_ref = df['Time'].max()
#
#     # 计算每条评论距离最新时间的天数差
#     df['days_diff'] = (t_ref - df['Time']).dt.total_seconds() / (24 * 3600)
#
#     # 计算衰减常数 lambda:  lambda = ln(2) / half_life
#     lambda_decay = np.log(2) / half_life_days
#
#     # 计算每条评论的最终权重 W_i = exp(-lambda * days_diff)
#     df['weight'] = np.exp(-lambda_decay * df['days_diff'])
#
#     # 3. 计算全局先验基准 (Global Prior)
#     print("正在建立全局贝叶斯先验基准线...")
#     global_priors = df[sentiment_cols].mean().to_dict()
#     for col, mu in global_priors.items():
#         print(f"   - {col} 城市平均分: {mu:.4f}")
#
#     # 4. 局部加权求和 (Local Weighted Sum)
#     print("正在执行 POI 局部特征加权聚合...")
#     # 计算 W_i * S_i
#     for col in sentiment_cols:
#         df[f'{col}_weighted_score'] = df[col] * df['weight']
#
#     # 按 PID 分组求和
#     agg_funcs = {
#         'UId': 'count',  # 统计物理评论数
#         'weight': 'sum'  # 统计有效证据权重 (W_sum)
#     }
#     for col in sentiment_cols:
#         agg_funcs[f'{col}_weighted_score'] = 'sum'
#
#     poi_stats = df.groupby('PId').agg(agg_funcs).rename(columns={'UId': 'Total_Reviews', 'weight': 'Effective_Weight'})
#
#     # 5. 贝叶斯平滑融合 (Bayesian Smoothing)
#     print(f"正在应用贝叶斯收缩平滑 (置信阈值 C={confidence_c})...")
#     for col in sentiment_cols:
#         sum_w_s = poi_stats[f'{col}_weighted_score']
#         w_sum = poi_stats['Effective_Weight']
#         mu = global_priors[col]
#
#         # 核心数学公式: S_final = (Sum(W_i * S_i) + C * mu) / (W_sum + C)
#         final_col_name = f"{col}"
#         poi_stats[final_col_name] = (sum_w_s + confidence_c * mu) / (w_sum + confidence_c)
#
#         # 四舍五入保留4位小数，保持特征整洁
#         poi_stats[final_col_name] = poi_stats[final_col_name].round(4)
#
#     # 6. 整理与输出
#     final_features = ['Total_Reviews', 'Effective_Weight'] + [f"{col}" for col in sentiment_cols]
#     result_df = poi_stats[final_features].reset_index()
#
#     # 保留有效权重到小数点后两位
#     result_df['Effective_Weight'] = result_df['Effective_Weight'].round(2)
#
#     result_df.to_csv(output_csv_path, index=False)
#
#     elapsed_time = time.time() - start_time
#     print(f"处理完成！耗时: {elapsed_time:.2f} 秒")
#     print(f"成功生成高鲁棒性 POI 字典: {output_csv_path}")
#     print(f"   - 共提取 {len(result_df)} 个独立 POI 的高维静态情感特征。")
#
#     # 打印前 3 条预览一下
#     print("\n数据预览 (Top 3):")
#     print(result_df.head(3).to_string(index=False))
#
#
# if __name__ == "__main__":
#     # ================= 配置区 =================
#     # 1. 你的输入文件路径 (带有情感的交互级 CSV)
#     INPUT_FILE = "/home/mysjz/mywork/SA-2-main/data/NOLA_with_sentiment.csv"  # <-- 请修改为你切分后的训练集路径
#
#     # 2. 你的输出文件路径 (这就是我们要喂给 SID 的绝对物理字典)
#     OUTPUT_FILE = "/home/mysjz/mywork/SA-2-main/data/poi_sentiment.csv"
#
#     # ================= 执行区 =================
#     generate_robust_poi_sentiment(
#         input_csv_path=INPUT_FILE,
#         output_csv_path=OUTPUT_FILE,
#         half_life_days=365.0,  # 默认 1 年半衰期，可根据 Yelp 数据的跨度调整
#         confidence_c=10.0  # 如果 Yelp 评论较稀疏，可调低到 5.0；如果很密集，可调高到 20.0
#     )

import pandas as pd
import numpy as np
import os
import time


def generate_robust_poi_sentiment_v2(
        input_csv_path,
        output_csv_path,
        sentiment_cols=['service', 'environment', 'price', 'location', 'core_experience'],
        half_life_days=365.0,
        unmentioned_val=0.0  # ⚠️ 关键点：如果你之前抽取时没有提到的维度记为了0.0，填0.0；如果是NaN，填None
):
    print("启动全动态解耦 POI 情感聚合引擎 (V2 增强版)...")
    start_time = time.time()

    # 1. 数据读取与基础处理
    df = pd.read_csv(input_csv_path)

    # # 统一列名小写
    # df.columns = [c.lower() for c in df.columns]
    # sentiment_cols = [c.lower() for c in sentiment_cols]

    # 统一列名大小写容错 (如果你之前的列名是小写，这里自动适配)
    col_mapping = {c.lower(): c for c in df.columns}
    df.rename(columns={c: col_mapping.get(c.lower(), c) for c in sentiment_cols}, inplace=True)

    # 【修复2：缺失值清洗】将未提及的维度(0.0)替换为 NaN，防止污染计算
    if unmentioned_val is not None:
        for col in sentiment_cols:
            df[col] = df[col].replace(unmentioned_val, np.nan)

    # 2. 基础时间衰减权重计算
    print(f"计算时间指数衰减 (半衰期: {half_life_days} 天)...")
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    t_ref = df['Time'].max()
    df['days_diff'] = (t_ref - df['Time']).dt.total_seconds() / (24 * 3600)
    lambda_decay = np.log(2) / half_life_days

    # 基础权重 (Base Weight)
    df['base_weight'] = np.exp(-lambda_decay * df['days_diff'])

    # 用于存放各维度独立统计数据的字典
    dynamic_priors = {}
    dynamic_C = {}

    # ========================================================
    # 核心引擎：按【维度(Aspect)】独立进行掩码隔离与计算
    # ========================================================
    print("正在执行维度解耦的权重掩码与动态先验计算...")

    for col in sentiment_cols:
        # 【修复2：掩码矩阵】如果该评论未提及该维度，该维度的独立权重为 0
        df[f'{col}_weight'] = df['base_weight'].where(df[col].notna(), 0.0)

        # 分子: Wi * Si (缺失值自动被 pandas fillna(0) 处理掉，因为权重已经是0了)
        df[f'{col}_w_score'] = df[col].fillna(0) * df[f'{col}_weight']

        # 【修复1：时间衰减的全局先验】
        # 城市的真实平均分 = (全市该维度的总加权分) / (全市该维度的总权重)
        col_prior = df[f'{col}_w_score'].sum() / df[f'{col}_weight'].sum()
        dynamic_priors[col] = col_prior

    # ========================================================
    # POI 级别聚合
    # ========================================================
    print("正在聚合 POI 级别的动态特征...")

    agg_dict = {'UId': 'count'}  # 物理交互次数
    for col in sentiment_cols:
        agg_dict[f'{col}_weight'] = 'sum'  # W_sum (该POI在该维度的有效权重积累)
        agg_dict[f'{col}_w_score'] = 'sum'  # 该POI的加权总分

    poi_stats = df.groupby('PId').agg(agg_dict).reset_index()
    poi_stats.rename(columns={'UId': 'Total_Interactions'}, inplace=True)

    # ========================================================
    # 【修复3：自适应贝叶斯平滑】
    # ========================================================
    print("正在注入自适应贝叶斯平滑 (Adaptive C)...")

    for col in sentiment_cols:
        # 获取该维度有效权重的宏观分布 (只看有数据积累的POI)
        active_pois = poi_stats[poi_stats[f'{col}_weight'] > 0]

        # 动态 C 值：取全市有该维度数据的 POI 的有效权重 中位数(Median)
        # 含义：想摆脱平滑，你的有效评论积累必须击败全城一半的店！
        C_adaptive = active_pois[f'{col}_weight'].quantile(0.5)

        # 防止极端情况 C=0 导致除以0
        C_adaptive = max(C_adaptive, 0.1)
        dynamic_C[col] = C_adaptive

        # 应用贝叶斯收缩公式
        w_sum = poi_stats[f'{col}_weight']
        score_sum = poi_stats[f'{col}_w_score']
        mu = dynamic_priors[col]

        poi_stats[f'{col}'] = (score_sum + C_adaptive * mu) / (w_sum + C_adaptive)
        poi_stats[f'{col}'] = poi_stats[f'{col}'].round(4)

    # 3. 输出汇总信息与落盘
    print("\n[全局动态参数解密]")
    for col in sentiment_cols:
        print(
            f"  - 维度 [{col.upper()}]: 全局先验均值(μ) = {dynamic_priors[col]:.4f} | 自适应平滑系数(C) = {dynamic_C[col]:.4f}")

    # 提取最终需要的列
    final_cols = ['PId', 'Total_Interactions'] + [f'{col}' for col in sentiment_cols]
    result_df = poi_stats[final_cols]

    result_df.to_csv(output_csv_path, index=False)

    print(f"\n处理完成！耗时: {time.time() - start_time:.2f} 秒")
    print(f"成功生成极度鲁棒的 SA-SID 特征表: {output_csv_path}")


if __name__ == "__main__":
    # 配置你的文件路径
    INPUT_FILE = "/home/mysjz/mywork/SA-2-main/data/NOLA_with_sentiment.csv"  # 包含情感分数的时序表
    OUTPUT_FILE = "/home/mysjz/mywork/SA-2-main/data/poi_sentiment.csv"

    generate_robust_poi_sentiment_v2(
        input_csv_path=INPUT_FILE,
        output_csv_path=OUTPUT_FILE,
        half_life_days=365.0,
        unmentioned_val=0.0  # 如果你的抽取代码将没提到的维度打分为 0.0，这里填0.0
    )

# 启动全动态解耦 POI 情感聚合引擎 (V2 增强版)...
# 计算时间指数衰减 (半衰期: 365.0 天)...
# 正在执行维度解耦的权重掩码与动态先验计算...
# 正在聚合 POI 级别的动态特征...
# 正在注入自适应贝叶斯平滑 (Adaptive C)...
#
# [全局动态参数解密]
#   - 维度 [SERVICE]: 全局先验均值(μ) = 0.4354 | 自适应平滑系数(C) = 5.8336
#   - 维度 [ENVIRONMENT]: 全局先验均值(μ) = 0.3061 | 自适应平滑系数(C) = 5.5619
#   - 维度 [PRICE]: 全局先验均值(μ) = -0.1954 | 自适应平滑系数(C) = 4.0560
#   - 维度 [LOCATION]: 全局先验均值(μ) = 0.4802 | 自适应平滑系数(C) = 5.8581
#   - 维度 [CORE_EXPERIENCE]: 全局先验均值(μ) = 0.4470 | 自适应平滑系数(C) = 6.1185
#
# 处理完成！耗时: 0.08 秒
# 成功生成极度鲁棒的 SA-SID 特征表: /home/mysjz/mywork/SA-2-main/data/poi_sentiment.csv