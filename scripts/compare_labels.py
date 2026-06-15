"""
标注误差分析脚本 — AI凉茶机项目
================================

对比 AI 生成标签（tcm_data_labeled.csv 验证子集）与人工标注标签（tcm_data_manual.csv），
输出一致性统计、差异明细和混淆摘要。

用法：
  python scripts/compare_labels.py [labeled_csv] [manual_csv]

默认：
  python scripts/compare_labels.py data/tcm_data_labeled.csv data/tcm_data_manual.csv
"""

import csv
import os
import sys
from collections import defaultdict


# ============================================================
# 配置
# ============================================================

DEFAULT_LABELED = "data/tcm_data_labeled.csv"
DEFAULT_MANUAL = "data/tcm_data_manual.csv"
OUTPUT_DIFF = "data/label_diff.csv"

CONSTITUTION_NAMES = {
    "A": "平和质", "B": "气虚质", "C": "阳虚质",
    "D": "阴虚质", "E": "痰湿质", "F": "湿热质",
    "G": "血瘀质", "H": "气郁质", "I": "特禀质",
}

ALL_CODES = list("ABCDEFGHI")

# ============================================================
# 工具函数
# ============================================================

def load_labeled_subset(filepath: str, n: int = 100) -> dict[str, dict]:
    """从 labeled CSV 加载前 n 条验证子集。返回 {id: {label, reason, features...}}"""
    records = {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            records[row["id"]] = row
    return records


def load_manual(filepath: str) -> dict[str, str]:
    """从人工标注 CSV 加载标签。返回 {id: label}"""
    records = {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records[row["id"]] = row["label"].strip().upper()
    return records


def write_diff_csv(filepath: str, diffs: list[dict]):
    """写入差异明细 CSV。"""
    fieldnames = [
        "id",
        "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
        "F1", "F2", "F3",
        "ai_label", "manual_label", "ai_reason",
    ]
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in diffs:
            writer.writerow(d)


# ============================================================
# 主流程
# ============================================================

def main():
    # 强制 stdout 使用 utf-8，避免 GBK 终端编码错误
    sys.stdout.reconfigure(encoding="utf-8")

    labeled_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LABELED
    manual_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MANUAL

    # --- 加载数据 ---
    print("=" * 60)
    print("AI凉茶机 — 标注一致性验证报告")
    print("=" * 60)

    ai_data = load_labeled_subset(labeled_path)
    manual_data = load_manual(manual_path)

    print(f"\nAI 标签数据来源 : {labeled_path}  (前 {len(ai_data)} 条)")
    print(f"人工标注数据来源: {manual_path}  ({len(manual_data)} 条)")

    # --- 按 id 对齐 ---
    common_ids = sorted(set(ai_data.keys()) & set(manual_data.keys()), key=int)
    only_ai = set(ai_data.keys()) - set(manual_data.keys())
    only_manual = set(manual_data.keys()) - set(ai_data.keys())

    if only_ai:
        print(f"\n警告: {len(only_ai)} 条仅在 AI 数据中存在")
    if only_manual:
        print(f"\n警告: {len(only_manual)} 条仅在人工数据中存在")

    # --- 逐条对比 ---
    matches = 0
    diffs = []

    for sid in common_ids:
        ai_label = ai_data[sid]["label"].strip().upper()
        manual_label = manual_data[sid]
        if ai_label == manual_label:
            matches += 1
        else:
            diffs.append({
                "id": sid,
                "Q1": ai_data[sid]["Q1"], "Q2": ai_data[sid]["Q2"],
                "Q3": ai_data[sid]["Q3"], "Q4": ai_data[sid]["Q4"],
                "Q5": ai_data[sid]["Q5"], "Q6": ai_data[sid]["Q6"],
                "Q7": ai_data[sid]["Q7"], "Q8": ai_data[sid]["Q8"],
                "Q9": ai_data[sid]["Q9"], "Q10": ai_data[sid]["Q10"],
                "F1": ai_data[sid]["F1"], "F2": ai_data[sid]["F2"],
                "F3": ai_data[sid]["F3"],
                "ai_label": ai_label,
                "manual_label": manual_label,
                "ai_reason": ai_data[sid].get("reason", ""),
            })

    total = len(common_ids)
    accuracy = matches / total * 100 if total > 0 else 0

    # --- 总体统计 ---
    print(f"\n{'─' * 40}")
    print(f"对比样本数: {total}")
    print(f"一致样本数: {matches}")
    print(f"不一致样本数: {len(diffs)}")
    print(f"总体一致率: {matches}/{total} = {accuracy:.2f}%")
    print(f"目标一致率: ≥ 90%")
    print(f"达标判定  : {'✓ 达标' if accuracy >= 90 else '✗ 未达标'}")

    # --- 按体质类型分组统计 ---
    print(f"\n{'─' * 40}")
    print("按 AI 标签体质类型分组的一致率：")
    class_stats = defaultdict(lambda: {"total": 0, "match": 0})
    for sid in common_ids:
        ai_label = ai_data[sid]["label"].strip().upper()
        manual_label = manual_data[sid]
        class_stats[ai_label]["total"] += 1
        if ai_label == manual_label:
            class_stats[ai_label]["match"] += 1

    for code in ALL_CODES:
        st = class_stats[code]
        if st["total"] > 0:
            rate = st["match"] / st["total"] * 100
            bar = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
            name = CONSTITUTION_NAMES[code]
            print(f"  {code} ({name}): {st['match']}/{st['total']:2d} = {rate:5.1f}%  {bar}")
        else:
            print(f"  {code} ({CONSTITUTION_NAMES[code]}): 无样本")

    # --- 按人工标签体质类型分组统计 ---
    print(f"\n{'─' * 40}")
    print("按人工标签体质类型分组的一致率：")
    class_stats_m = defaultdict(lambda: {"total": 0, "match": 0})
    for sid in common_ids:
        ai_label = ai_data[sid]["label"].strip().upper()
        manual_label = manual_data[sid]
        class_stats_m[manual_label]["total"] += 1
        if ai_label == manual_label:
            class_stats_m[manual_label]["match"] += 1

    for code in ALL_CODES:
        st = class_stats_m[code]
        if st["total"] > 0:
            rate = st["match"] / st["total"] * 100
            name = CONSTITUTION_NAMES[code]
            print(f"  {code} ({name}): {st['match']}/{st['total']:2d} = {rate:5.1f}%")
        else:
            print(f"  {code} ({CONSTITUTION_NAMES[code]}): 无样本")

    # --- 混淆摘要（AI 标签 vs 人工标签） ---
    if diffs:
        print(f"\n{'─' * 40}")
        print("不一致样本混淆矩阵 (AI标签 → 人工标签)：")
        confusion = defaultdict(lambda: defaultdict(int))
        for d in diffs:
            confusion[d["ai_label"]][d["manual_label"]] += 1

        for ai_code in ALL_CODES:
            for man_code in ALL_CODES:
                count = confusion[ai_code][man_code]
                if count > 0:
                    ai_name = CONSTITUTION_NAMES[ai_code]
                    man_name = CONSTITUTION_NAMES[man_code]
                    print(f"  AI={ai_code}({ai_name}) → 人工={man_code}({man_name}): {count} 条")

    # --- 不一致样本明细 ---
    if diffs:
        print(f"\n{'─' * 40}")
        print(f"不一致样本明细（前 15 条）：")
        for d in diffs[:15]:
            ai_name = CONSTITUTION_NAMES[d["ai_label"]]
            man_name = CONSTITUTION_NAMES[d["manual_label"]]
            q_str = "".join([d[f"Q{j}"] for j in range(1, 11)])
            f_str = f"F:{d['F1']}{d['F2']}{d['F3']}"
            print(f"  id={d['id']:4s} | Q={q_str} | {f_str} | AI={d['ai_label']}({ai_name}) → 人工={d['manual_label']}({man_name})")

        if len(diffs) > 15:
            print(f"  ... 共 {len(diffs)} 条，完整明细见 {OUTPUT_DIFF}")

    # --- 写入差异明细 ---
    if diffs:
        write_diff_csv(OUTPUT_DIFF, diffs)
        print(f"\n差异明细已写入: {OUTPUT_DIFF}")
    else:
        print(f"\n无差异记录，AI 标签与人工标注完全一致。")

    print(f"\n{'=' * 60}")
    print("验证完成")
    print("=" * 60)

    return 0 if accuracy >= 90 else 1


if __name__ == "__main__":
    sys.exit(main())
