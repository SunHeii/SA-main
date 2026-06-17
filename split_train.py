import os
import re

# ================= 配置您的路径 =================
# 指向您刚刚预处理生成的那个带有 [review_id] 的 train.txt
ORIGINAL_FILE = "/home/mysjz/mywork/SA-2-main/data/general_poi/train.txt"

# 将带 ID 的文件重命名为 inference_with_id.txt (留给 Phase 5 用)
INFERENCE_FILE = "/home/mysjz/mywork/SA-2-main/data/general_poi/inference_with_id.txt"

# 重新生成一个纯净的 train.txt (留给 run.py 训练用)
PURE_TRAIN_FILE = "/home/mysjz/mywork/SA-2-main/data/general_poi/train.txt"


# ================================================

def split_data_for_training_and_inference():
    print("开始进行数据分身处理...")

    if not os.path.exists(ORIGINAL_FILE):
        print(f"找不到文件: {ORIGINAL_FILE} (可能已经重命名过了？)")
        return

    # 1. 保护现场：把带 ID 的文件改名为 inference 专用文件
    os.rename(ORIGINAL_FILE, INFERENCE_FILE)
    print(f"已将原始文件安全转移至: {INFERENCE_FILE}")

    # 2. 剥离 ID，生成纯文本给 SA-2 训练
    print("正在剥离 Review_ID，生成纯净版训练文本...")
    with open(INFERENCE_FILE, 'r', encoding='utf-8') as f_in, \
            open(PURE_TRAIN_FILE, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            # 正则匹配：剥离开头的 [ID] ，只保留后面的纯文本
            match = re.match(r'^\[.*?\]\s*(.*)', line)
            if match:
                f_out.write(match.group(1) + '\n')
            else:
                f_out.write(line)

    print(f"纯净版 train.txt 已生成，可以安全用于 run.py 训练！")


if __name__ == "__main__":
    split_data_for_training_and_inference()