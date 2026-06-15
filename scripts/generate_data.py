"""
数据生成脚本 — AI凉茶机项目
============================

依据 proof.md 第六节「特征与体质对应关系总览」表，生成 1000 条中医体质分类样本。

产出文件：
  - data/tcm_data_labeled.csv  : 带 AI 分类标签和分类依据
  - data/tcm_data_unlabeled.csv: 相同特征数据，无标签（供人工标注）

用法：
  python scripts/generate_data.py
"""

import csv
import os
import random
import sys

# ============================================================
# 配置参数
# ============================================================

RANDOM_SEED = 42
TOTAL_SAMPLES = 1000
SAMPLES_PER_CLASS = 111  # 9 * 111 = 999, 剩余 1 条分配给平和质
EXTRA_SAMPLE_CLASS = "A"  # 多出的一条给平和质

# 噪声概率（按 question.md 第三节建议）
CORE_FEATURE_NOISE_MIN = 0.05   # 核心特征最低扰动概率
CORE_FEATURE_NOISE_MAX = 0.10   # 核心特征最高扰动概率
NONCORE_FEATURE_NOISE_MIN = 0.15  # 非核心特征最低扰动概率
NONCORE_FEATURE_NOISE_MAX = 0.25  # 非核心特征最高扰动概率
VISUAL_FEATURE_NOISE_MIN = 0.10   # 望诊特征最低扰动概率
VISUAL_FEATURE_NOISE_MAX = 0.15   # 望诊特征最高扰动概率

OUTPUT_DIR = "data"
LABELED_FILE = "tcm_data_labeled.csv"
UNLABELED_FILE = "tcm_data_unlabeled.csv"

# 人工验证集：从 1000 条中抽取前 N 条作为无标签数据供人工标注
VALIDATION_SUBSET_SIZE = 100  # 取前 100 条（随机打乱后等效于随机抽样）

# Q1-Q10 选项集合
Q_OPTIONS = ["A", "B", "C", "D"]

# ============================================================
# 体质名称映射
# ============================================================

CONSTITUTION_NAMES = {
    "A": "平和质",
    "B": "气虚质",
    "C": "阳虚质",
    "D": "阴虚质",
    "E": "痰湿质",
    "F": "湿热质",
    "G": "血瘀质",
    "H": "气郁质",
    "I": "特禀质",
}

# ============================================================
# 各体质标准特征模式（来源：proof.md 第六节，v1.1 修订版）
# ============================================================

STANDARD_PATTERNS = {
    "A": {  # 平和质
        "Q1": "A", "Q2": "A", "Q3": "A", "Q4": "A", "Q5": "A",
        "Q6": "A", "Q7": "A", "Q8": "A", "Q9": "A", "Q10": "A",
        "F1": 0, "F2": 0, "F3": 0,
    },
    "B": {  # 气虚质
        "Q1": "D", "Q2": "C", "Q3": "A", "Q4": "C", "Q5": "A",
        "Q6": "B", "Q7": "A", "Q8": "A", "Q9": "C", "Q10": "C",
        "F1": 1, "F2": 0, "F3": 1,  # 舌苔, 眼圈, 脸色
    },
    "C": {  # 阳虚质
        "Q1": "C", "Q2": "D", "Q3": "A", "Q4": "C", "Q5": "A",
        "Q6": "B", "Q7": "B", "Q8": "A", "Q9": "C", "Q10": "C",
        "F1": 1, "F2": 1, "F3": 1,
    },
    "D": {  # 阴虚质
        "Q1": "C", "Q2": "A", "Q3": "D", "Q4": "A", "Q5": "A",
        "Q6": "B", "Q7": "A", "Q8": "A", "Q9": "C", "Q10": "C",
        "F1": 1, "F2": 1, "F3": 1,
    },
    "E": {  # 痰湿质
        "Q1": "C", "Q2": "A", "Q3": "A", "Q4": "D", "Q5": "C",
        "Q6": "B", "Q7": "A", "Q8": "A", "Q9": "C", "Q10": "B",
        "F1": 1, "F2": 1, "F3": 0,  # 舌苔, 眼圈, 脸色
    },
    "F": {  # 湿热质
        "Q1": "C", "Q2": "A", "Q3": "C", "Q4": "C", "Q5": "D",
        "Q6": "B", "Q7": "A", "Q8": "B", "Q9": "C", "Q10": "D",
        "F1": 1, "F2": 0, "F3": 1,  # 舌苔, 眼圈, 脸色
    },
    "G": {  # 血瘀质
        "Q1": "C", "Q2": "B", "Q3": "A", "Q4": "A", "Q5": "A",
        "Q6": "C", "Q7": "D", "Q8": "A", "Q9": "C", "Q10": "C",
        "F1": 1, "F2": 1, "F3": 1,
    },
    "H": {  # 气郁质
        "Q1": "C", "Q2": "A", "Q3": "B", "Q4": "A", "Q5": "A",
        "Q6": "D", "Q7": "A", "Q8": "B", "Q9": "D", "Q10": "C",
        "F1": 0, "F2": 0, "F3": 0,
    },
    "I": {  # 特禀质
        "Q1": "A", "Q2": "A", "Q3": "A", "Q4": "A", "Q5": "A",
        "Q6": "B", "Q7": "A", "Q8": "D", "Q9": "C", "Q10": "B",
        "F1": 0, "F2": 0, "F3": 0,
    },
}

# ============================================================
# 各体质核心诊断特征（映射表中加粗的 D 或 C 所在特征）
# ============================================================

CORE_FEATURES = {
    "A": [],                # 平和质 — 无核心症状，以全 A 为特征
    "B": ["Q1"],            # 气虚质 — Q1=D 疲劳气短
    "C": ["Q2"],            # 阳虚质 — Q2=D 怕冷肢凉
    "D": ["Q3", "Q10"],     # 阴虚质 — Q3=D 口干冷饮, Q10=C 便秘
    "E": ["Q4"],            # 痰湿质 — Q4=D 身重痰多
    "F": ["Q5", "Q10"],     # 湿热质 — Q5=D 出油长痘, Q10=D 便黏
    "G": ["Q7"],            # 血瘀质 — Q7=D 瘀斑晦暗
    "H": ["Q6", "Q9"],      # 气郁质 — Q6=D 情绪低落, Q9=D 睡眠差
    "I": ["Q8"],            # 特禀质 — Q8=D 过敏史
}

# ============================================================
# 特征描述标签（用于生成 reason 字段）
# ============================================================

Q_LABELS = {
    "Q1": "疲劳气短", "Q2": "怕冷肢凉", "Q3": "口干冷饮",
    "Q4": "身重痰多", "Q5": "出油长痘", "Q6": "情绪低落",
    "Q7": "瘀斑晦暗", "Q8": "过敏史", "Q9": "睡眠质量",
    "Q10": "二便状况",
}

F_LABELS = {
    "F1": "舌苔", "F2": "眼圈", "F3": "脸色",
}

# ============================================================
# 工具函数
# ============================================================

def adjacent_options(value: str) -> list[str]:
    """返回 Q 选项的相邻值列表（用于扰动）。A→[B], B→[A,C], C→[B,D], D→[C]"""
    idx = Q_OPTIONS.index(value)
    result = []
    if idx > 0:
        result.append(Q_OPTIONS[idx - 1])
    if idx < len(Q_OPTIONS) - 1:
        result.append(Q_OPTIONS[idx + 1])
    return result


def perturb_q(value: str) -> str:
    """将 Q 选项随机扰动到相邻等级。"""
    candidates = adjacent_options(value)
    return random.choice(candidates)


def perturb_f(value: int) -> int:
    """翻转二值特征。"""
    return 1 - value


def random_noise_prob(min_pct: float, max_pct: float) -> float:
    """在给定范围内随机取一个噪声概率。"""
    return random.uniform(min_pct, max_pct)


def make_reason(
    constitution: str,
    standard: dict,
    perturbed: dict,
    perturbations: list[str],
) -> str:
    """
    生成分类依据说明字符串。

    格式：
      核心匹配：{体质名} | 核心特征：{特征=值(标签)} | 扰动(N处)：{详情} | 标准模式 或 含噪声
    """
    name = CONSTITUTION_NAMES[constitution]
    core_list = CORE_FEATURES[constitution]

    # 核心特征描述
    core_desc_parts = []
    for feat in core_list:
        if feat.startswith("Q"):
            val = perturbed[feat]
            core_desc_parts.append(f"{feat}={val}({Q_LABELS[feat]})")
        else:
            val = perturbed[feat]
            core_desc_parts.append(f"{feat}={val}({F_LABELS[feat]}{'异常' if val == 1 else '正常'})")

    core_desc = ", ".join(core_desc_parts) if core_desc_parts else "无特定核心特征（平和质基准）"

    # 扰动详情
    if perturbations:
        pert_desc = "; ".join(perturbations)
        tail = "含噪声"
    else:
        pert_desc = "无"
        tail = "标准模式"

    return f"核心匹配：{name} | 核心特征：{core_desc} | 扰动({len(perturbations)}处)：{pert_desc} | {tail}"


def generate_samples_for_constitution(
    constitution: str,
    count: int,
    start_id: int,
) -> list[dict]:
    """
    为指定体质生成 count 条带噪声的样本。

    Args:
        constitution: 体质编码 A-I
        count: 需要生成的样本数
        start_id: 起始序号

    Returns:
        list[dict]: 样本列表
    """
    standard = STANDARD_PATTERNS[constitution]
    core_set = set(CORE_FEATURES[constitution])
    samples = []

    for i in range(count):
        sample_id = start_id + i
        perturbed = {}
        perturbations = []

        # --- 处理 Q1-Q10 ---
        for q in [f"Q{j}" for j in range(1, 11)]:
            std_val = standard[q]
            is_core = q in core_set

            if is_core:
                noise_prob = random_noise_prob(CORE_FEATURE_NOISE_MIN, CORE_FEATURE_NOISE_MAX)
            else:
                noise_prob = random_noise_prob(NONCORE_FEATURE_NOISE_MIN, NONCORE_FEATURE_NOISE_MAX)

            if random.random() < noise_prob:
                new_val = perturb_q(std_val)
                # 平和质特殊约束：不允许出现 D 级症状
                if constitution == "A" and new_val == "D":
                    new_val = "C"  # 最多扰动到 C
                if new_val != std_val:
                    perturbations.append(f"{q} {std_val}→{new_val}")
                perturbed[q] = new_val
            else:
                perturbed[q] = std_val

        # --- 处理 F1-F3 ---
        for f in ["F1", "F2", "F3"]:
            std_val = standard[f]
            noise_prob = random_noise_prob(VISUAL_FEATURE_NOISE_MIN, VISUAL_FEATURE_NOISE_MAX)

            if random.random() < noise_prob:
                new_val = perturb_f(std_val)
                if new_val != std_val:
                    perturbations.append(f"{f} {std_val}→{new_val}")
                perturbed[f] = new_val
            else:
                perturbed[f] = std_val

        # --- 平和质后校验：确保无 D 级症状 ---
        if constitution == "A":
            for q in [f"Q{j}" for j in range(1, 11)]:
                if perturbed[q] == "D":
                    perturbed[q] = "C"
                    # 如果之前不在扰动列表中，添加
                    already_recorded = any(p.startswith(q) for p in perturbations)
                    if not already_recorded:
                        perturbations.append(f"{q} D→C(平和质约束)")

        # --- 生成分类依据 ---
        reason = make_reason(constitution, standard, perturbed, perturbations)

        sample = {
            "id": sample_id,
            "Q1": perturbed["Q1"], "Q2": perturbed["Q2"], "Q3": perturbed["Q3"],
            "Q4": perturbed["Q4"], "Q5": perturbed["Q5"], "Q6": perturbed["Q6"],
            "Q7": perturbed["Q7"], "Q8": perturbed["Q8"], "Q9": perturbed["Q9"],
            "Q10": perturbed["Q10"],
            "F1": perturbed["F1"], "F2": perturbed["F2"], "F3": perturbed["F3"],
            "label": constitution,
            "reason": reason,
        }
        samples.append(sample)

    return samples


def write_labeled_csv(filepath: str, samples: list[dict]):
    """写入带标签 CSV 文件。"""
    fieldnames = [
        "id",
        "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
        "F1", "F2", "F3",
        "label", "reason",
    ]
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow(s)


def write_unlabeled_csv(filepath: str, samples: list[dict]):
    """写入无标签 CSV 文件（仅特征，无 label 和 reason）。"""
    fieldnames = [
        "id",
        "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
        "F1", "F2", "F3",
    ]
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow({k: s[k] for k in fieldnames})


def print_statistics(samples: list[dict]):
    """打印数据统计信息。"""
    label_counts = {}
    noise_counts = []
    for s in samples:
        label = s["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        # 统计扰动数量（从 reason 中解析）
        reason = s["reason"]
        if "扰动(0处)" in reason:
            noise_counts.append(0)
        else:
            # 提取扰动数量
            import re
            match = re.search(r"扰动\((\d+)处\)", reason)
            if match:
                noise_counts.append(int(match.group(1)))

    print("=" * 60)
    print("数据生成统计")
    print("=" * 60)
    print(f"\n总样本数：{len(samples)}")
    print(f"\n各类别分布：")
    for code in sorted(label_counts.keys()):
        name = CONSTITUTION_NAMES[code]
        count = label_counts[code]
        bar = "█" * (count // 5)
        print(f"  {code} ({name}): {count:4d}  {bar}")

    print(f"\n噪声统计：")
    print(f"  标准模式（无扰动）：{noise_counts.count(0)} 条 ({noise_counts.count(0)/len(samples)*100:.1f}%)")
    print(f"  含噪声样本        ：{len(samples) - noise_counts.count(0)} 条 ({(len(samples) - noise_counts.count(0))/len(samples)*100:.1f}%)")
    if noise_counts:
        print(f"  平均每样本扰动数  ：{sum(noise_counts)/len(noise_counts):.2f}")
        print(f"  最大扰动数        ：{max(noise_counts)}")
        print(f"  最小扰动数（非零）：{min(c for c in noise_counts if c > 0)}")

    print(f"\n输出文件：")
    print(f"  带标签：{os.path.join(OUTPUT_DIR, LABELED_FILE)}")
    print(f"  无标签：{os.path.join(OUTPUT_DIR, UNLABELED_FILE)}")
    print("=" * 60)


# ============================================================
# 主流程
# ============================================================

def main():
    # 设置随机种子以确保可复现
    random.seed(RANDOM_SEED)

    # 计算各类样本数
    constitutions = list(CONSTITUTION_NAMES.keys())
    class_counts = {c: SAMPLES_PER_CLASS for c in constitutions}
    class_counts[EXTRA_SAMPLE_CLASS] += TOTAL_SAMPLES - SAMPLES_PER_CLASS * len(constitutions)

    print(f"开始生成数据（随机种子={RANDOM_SEED}）...")
    print(f"目标样本量：{TOTAL_SAMPLES}")
    print(f"各类分配：{dict(class_counts)}")

    # 生成所有样本
    all_samples = []

    for code in constitutions:
        count = class_counts[code]
        samples = generate_samples_for_constitution(code, count, start_id=1)  # 临时id，稍后重排
        all_samples.extend(samples)
        name = CONSTITUTION_NAMES[code]
        noisy = sum(1 for s in samples if "扰动(0处)" not in s["reason"])
        print(f"  {code} ({name}): 生成 {len(samples)} 条, 含噪声 {noisy} 条 ({noisy/len(samples)*100:.0f}%)")

    # --- 随机打乱样本顺序，避免同体质聚集影响人工标注 ---
    # 使用固定种子的随机置换，保证可复现
    rng = random.Random(RANDOM_SEED + 1)  # 独立于生成噪声的种子
    rng.shuffle(all_samples)

    # 重新分配连续序号（1..N）
    for new_id, sample in enumerate(all_samples, start=1):
        sample["id"] = new_id

    print(f"\n样本已随机重排，新序号 1–{len(all_samples)}，各类体质均匀分散。")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 写入 CSV
    labeled_path = os.path.join(OUTPUT_DIR, LABELED_FILE)
    unlabeled_path = os.path.join(OUTPUT_DIR, UNLABELED_FILE)

    # labeled: 全量 1000 条（含标签和分类依据）
    write_labeled_csv(labeled_path, all_samples)

    # unlabeled: 仅前 VALIDATION_SUBSET_SIZE 条，去标签，供人工标注
    validation_subset = all_samples[:VALIDATION_SUBSET_SIZE]
    write_unlabeled_csv(unlabeled_path, validation_subset)

    # 打印统计
    print_statistics(all_samples)

    # --- 验证子集统计 ---
    val_labels = {}
    for s in validation_subset:
        lbl = s["label"]
        val_labels[lbl] = val_labels.get(lbl, 0) + 1
    print(f"\n人工验证子集（前 {VALIDATION_SUBSET_SIZE} 条）体质分布：")
    for code in sorted(val_labels.keys()):
        name = CONSTITUTION_NAMES[code]
        count = val_labels[code]
        print(f"  {code} ({name}): {count} 条")


if __name__ == "__main__":
    main()
