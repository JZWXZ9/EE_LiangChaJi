"""
模型训练脚本 — AI凉茶机项目
===========================

加载 tcm_data_labeled.csv，训练 sklearn 决策树（CART）和手写 ID3 决策树，
评估对比后将较优模型导出为 deploy/model.json。

用法：
  python scripts/train_model.py
"""

import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# 配置
# ============================================================

RANDOM_SEED = 42
DATA_FILE = "data/tcm_data_labeled.csv"
MODEL_OUTPUT = "deploy/model.json"

TEST_SIZE = 0.15
VAL_SIZE = 0.15 / 0.85  # 从剩余 85% 中再切 15%（总体的 ~15%）

# 特征列表
Q_FEATURES = [f"Q{i}" for i in range(1, 11)]
F_FEATURES = ["F1", "F2", "F3"]
ALL_FEATURES = Q_FEATURES + F_FEATURES
LABEL_COL = "label"

# 编码映射
Q_ENCODE = {"A": 0, "B": 1, "C": 2, "D": 3}
Q_DECODE = {0: "A", 1: "B", 2: "C", 3: "D"}
LABEL_CLASSES = list("ABCDEFGHI")

CONSTITUTION_NAMES = {
    "A": "平和质", "B": "气虚质", "C": "阳虚质",
    "D": "阴虚质", "E": "痰湿质", "F": "湿热质",
    "G": "血瘀质", "H": "气郁质", "I": "特禀质",
}

# ============================================================
# 数据加载与预处理
# ============================================================

def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """加载 CSV 数据，返回特征矩阵 X 和标签向量 y（均为数值编码）。"""
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    X_list = []
    y_list = []

    for r in rows:
        feat = []
        for q in Q_FEATURES:
            feat.append(Q_ENCODE[r[q]])
        for f in F_FEATURES:
            feat.append(int(r[f]))
        X_list.append(feat)
        y_list.append(r[LABEL_COL].strip().upper())

    return np.array(X_list), np.array(y_list)


def split_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """分层划分训练/验证/测试集。"""
    # 先切出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    # 再从剩余中切出验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"训练集: {len(X_train)} 条")
    print(f"验证集: {len(X_val)} 条")
    print(f"测试集: {len(X_test)} 条")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 模型一：sklearn CART 决策树
# ============================================================

def train_sklearn_models(X_train, y_train, X_val, y_val):
    """训练多组 sklearn 决策树参数，返回验证集最优模型。"""
    param_list = [
        {"criterion": "gini", "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "entropy", "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "entropy", "max_depth": 4, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "entropy", "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
        {"criterion": "entropy", "max_depth": 5, "min_samples_split": 3, "min_samples_leaf": 2},
        {"criterion": "entropy", "max_depth": 6, "min_samples_split": 5, "min_samples_leaf": 2},
        {"criterion": "gini", "max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 2},
        {"criterion": "entropy", "max_depth": 7, "min_samples_split": 10, "min_samples_leaf": 3},
    ]

    results = []
    best_model = None
    best_f1 = 0.0

    print("\n--- sklearn 决策树参数搜索 ---")
    for params in param_list:
        clf = DecisionTreeClassifier(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=RANDOM_SEED,
        )
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_val_pred)
        f1_macro = f1_score(y_val, y_val_pred, average="macro", labels=LABEL_CLASSES)

        results.append({
            **params,
            "accuracy": acc,
            "f1_macro": f1_macro,
        })

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model = clf

        print(f"  {params['criterion']:8s} depth={params['max_depth']} "
              f"split={params['min_samples_split']} leaf={params['min_samples_leaf']}"
              f"  → Acc={acc:.4f}  F1_macro={f1_macro:.4f}")

    print(f"\n最优 sklearn 参数: depth={best_model.max_depth}, "
          f"criterion={best_model.criterion}, F1_macro={best_f1:.4f}")

    return best_model, results


# ============================================================
# 模型二：手写 ID3 多叉决策树
# ============================================================

def calc_entropy(y: np.ndarray) -> float:
    """计算标签数组的信息熵。"""
    if len(y) == 0:
        return 0.0
    counter = Counter(y)
    total = len(y)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def calc_info_gain(X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
    """计算某个特征的信息增益。"""
    origin_entropy = calc_entropy(y)
    values = set(X[:, feature_idx])
    weighted_entropy = 0.0
    for val in values:
        mask = X[:, feature_idx] == val
        subset_y = y[mask]
        weight = len(subset_y) / len(y)
        weighted_entropy += weight * calc_entropy(subset_y)
    return origin_entropy - weighted_entropy


def majority_class(y: np.ndarray) -> str:
    """返回多数类标签。"""
    return Counter(y).most_common(1)[0][0]


def build_id3_tree(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: list[int],
    feature_names: list[str],
    max_depth: int | None = None,
    current_depth: int = 0,
) -> dict | str:
    """
    递归构建 ID3 多叉决策树。

    Returns:
        dict: {"feature": str, "majority": str, "children": {value: subtree}}
        str: 叶子节点标签
    """
    # 叶子条件
    if len(set(y)) == 1:
        return y[0]
    if len(feature_indices) == 0:
        return majority_class(y)
    if max_depth is not None and current_depth >= max_depth:
        return majority_class(y)

    # 选择最佳分裂特征
    gains = [calc_info_gain(X, y, fi) for fi in feature_indices]
    best_local_idx = int(np.argmax(gains))
    best_feature_idx = feature_indices[best_local_idx]
    best_feature_name = feature_names[best_feature_idx]

    # 构建节点
    tree = {
        "feature": best_feature_name,
        "majority": majority_class(y),
        "children": {},
    }

    remaining_indices = [i for i in feature_indices if i != best_feature_idx]
    values = sorted(set(X[:, best_feature_idx]))

    for val in values:
        mask = X[:, best_feature_idx] == val
        child_tree = build_id3_tree(
            X[mask], y[mask], remaining_indices, feature_names,
            max_depth, current_depth + 1,
        )
        # 用字母存储 Q 特征值，数字存储 F 特征值
        if best_feature_name.startswith("Q"):
            key = Q_DECODE[val]
        else:
            key = str(val)
        tree["children"][key] = child_tree

    return tree


def train_id3(X_train, y_train, feature_names, max_depth=None):
    """训练 ID3 决策树。"""
    feature_indices = list(range(len(feature_names)))
    tree = build_id3_tree(X_train, y_train, feature_indices, feature_names, max_depth)
    return tree


def predict_id3_one(tree: dict | str, sample: np.ndarray, feature_names: list[str]) -> str:
    """ID3 单样本预测。"""
    if not isinstance(tree, dict):
        return tree

    feature_name = tree["feature"]
    feature_idx = feature_names.index(feature_name)
    value = sample[feature_idx]

    # 用字母存储 Q 特征值
    if feature_name.startswith("Q"):
        key = Q_DECODE[int(value)]
    else:
        key = str(int(value))

    if key in tree["children"]:
        return predict_id3_one(tree["children"][key], sample, feature_names)
    else:
        return tree["majority"]


def predict_id3(tree: dict | str, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """ID3 批量预测。"""
    return np.array([predict_id3_one(tree, x, feature_names) for x in X])


# ============================================================
# 模型评估
# ============================================================

def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = True):
    """打印模型完整评估指标，含混淆矩阵和按类 Precision/Recall/F1。"""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", labels=LABEL_CLASSES, zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", labels=LABEL_CLASSES, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=LABEL_CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_CLASSES)

    per_class_prec = precision_score(y_true, y_pred, average=None, labels=LABEL_CLASSES, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=LABEL_CLASSES, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=LABEL_CLASSES, zero_division=0)

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"{name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec_macro:.4f} (macro)")
        print(f"  Recall   : {rec_macro:.4f} (macro)")
        print(f"  F1-score : {f1_macro:.4f} (macro)")

        # 按类详细指标
        print(f"\n  按类详细指标:")
        print(f"  {'体质':6s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}  {'支持':>6s}")
        print(f"  {'─' * 48}")
        for i, code in enumerate(LABEL_CLASSES):
            support = int(cm[i].sum())
            if support > 0:
                name = CONSTITUTION_NAMES[code]
                print(f"  {code} ({name:4s}): {per_class_prec[i]:10.4f} {per_class_rec[i]:10.4f} {per_class_f1[i]:10.4f}  {support:6d}")
            else:
                print(f"  {code} ({CONSTITUTION_NAMES[code]:4s}): 无样本")

        # 混淆矩阵
        print(f"\n  混淆矩阵 (行=真实, 列=预测):")
        header = "       " + "".join(f"  {c}  " for c in LABEL_CLASSES)
        print(header)
        for i, code_row in enumerate(LABEL_CLASSES):
            row_str = "".join(f"{cm[i][j]:5d}" for j in range(len(LABEL_CLASSES)))
            print(f"  {code_row} │ {row_str}")

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            code: {
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "f1": float(per_class_f1[i]),
                "support": int(cm[i].sum()),
            }
            for i, code in enumerate(LABEL_CLASSES)
        },
    }


# ============================================================
# JSON 导出
# ============================================================

def sklearn_tree_to_dict(clf: DecisionTreeClassifier, feature_names: list[str]) -> dict:
    """
    将 sklearn 决策树递归转换为部署用 JSON 格式（二叉判定树）。

    格式:
        {"feature": "Q1", "threshold": 1.5, "majority": "C",
         "left": {...}, "right": {...}}
    叶子节点:
        {"feature": null, "majority": "C"}
    """
    tree = clf.tree_
    label_encoder = {i: cls for i, cls in enumerate(clf.classes_)}

    def _recurse(node_id: int) -> dict:
        if tree.children_left[node_id] == -1:  # 叶子
            values = tree.value[node_id][0]
            majority_idx = int(np.argmax(values))
            return {
                "feature": None,
                "majority": label_encoder[majority_idx],
            }

        feat_name = feature_names[tree.feature[node_id]]
        majority_idx = int(np.argmax(tree.value[node_id][0]))

        return {
            "feature": feat_name,
            "threshold": float(tree.threshold[node_id]),
            "majority": label_encoder[majority_idx],
            "left": _recurse(tree.children_left[node_id]),
            "right": _recurse(tree.children_right[node_id]),
        }

    return _recurse(0)


def export_tree_to_json(tree: dict, filepath: str, label: str = ""):
    """将决策树导出为 JSON 文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    print(f"  模型已导出至: {filepath}  {label}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("AI凉茶机 — 决策树模型训练")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1/5] 加载数据: {DATA_FILE}")
    X, y = load_data(DATA_FILE)
    feature_names = ALL_FEATURES.copy()
    print(f"  样本数: {len(X)}, 特征数: {len(feature_names)}")
    print(f"  标签分布: {dict(sorted(Counter(y).items()))}")

    # 2. 划分数据集
    print(f"\n[2/5] 划分训练/验证/测试集")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. sklearn 模型训练
    print(f"\n[3/5] 训练 sklearn CART 决策树")
    sklearn_model, sklearn_results = train_sklearn_models(X_train, y_train, X_val, y_val)

    # 4. ID3 模型训练（参考对比）
    print(f"\n[4/5] 训练手写 ID3 决策树（对比参考）")
    id3_best = None
    id3_best_f1 = 0.0
    id3_best_depth = None

    for depth in [3, 4, 5, 6, 7, None]:
        id3_tree = train_id3(X_train, y_train, feature_names, max_depth=depth)
        y_val_pred = predict_id3(id3_tree, X_val, feature_names)
        f1 = f1_score(y_val, y_val_pred, average="macro", labels=LABEL_CLASSES, zero_division=0)
        depth_label = str(depth) if depth is not None else "无限制"
        print(f"  max_depth={depth_label:6s} → Val F1_macro={f1:.4f}")

        if f1 > id3_best_f1:
            id3_best_f1 = f1
            id3_best = id3_tree
            id3_best_depth = depth

    print(f"  最优 ID3 max_depth: {id3_best_depth}, Val F1_macro={id3_best_f1:.4f}")

    # 5. 全数据集评估 + 导出
    print(f"\n[5/5] 训练/验证/测试集评估与模型导出")

    # sklearn — 训练集
    y_train_pred_sk = sklearn_model.predict(X_train)
    train_results = evaluate_model("sklearn CART — 训练集", y_train, y_train_pred_sk)

    # sklearn — 验证集
    y_val_pred_sk = sklearn_model.predict(X_val)
    val_results = evaluate_model("sklearn CART — 验证集", y_val, y_val_pred_sk)

    # sklearn — 测试集
    y_test_pred_sk = sklearn_model.predict(X_test)
    test_results = evaluate_model("sklearn CART — 测试集", y_test, y_test_pred_sk)

    # ID3 — 测试集（简要对比）
    y_test_pred_id3 = predict_id3(id3_best, X_test, feature_names)
    id3_results = evaluate_model("手写 ID3 (测试集)", y_test, y_test_pred_id3)

    # 汇总对比表
    print(f"\n{'─' * 60}")
    print("三集表现汇总 (sklearn CART):")
    print(f"  {'数据集':8s} {'样本数':>6s} {'Accuracy':>10s} {'Macro F1':>10s}")
    print(f"  {'─' * 38}")
    print(f"  {'训练集':8s} {len(y_train):6d} {train_results['accuracy']:10.4f} {train_results['f1_macro']:10.4f}")
    print(f"  {'验证集':8s} {len(y_val):6d} {val_results['accuracy']:10.4f} {val_results['f1_macro']:10.4f}")
    print(f"  {'测试集':8s} {len(y_test):6d} {test_results['accuracy']:10.4f} {test_results['f1_macro']:10.4f}")

    # 过拟合检查
    gap = train_results["f1_macro"] - test_results["f1_macro"]
    if gap < 0.05:
        print(f"\n  训练集→测试集 F1 差距: {gap:.4f} (过拟合程度低，泛化良好)")
    else:
        print(f"\n  训练集→测试集 F1 差距: {gap:.4f} (存在一定过拟合)")

    # 导出模型
    print(f"\n  导出模型...")
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    sk_json = sklearn_tree_to_dict(sklearn_model, feature_names)
    export_tree_to_json(sk_json, MODEL_OUTPUT, label="← sklearn CART")

    # 保存完整评估结果供开发报告使用
    results_json = {
        "model": {
            "type": "sklearn CART",
            "criterion": sklearn_model.criterion,
            "max_depth": sklearn_model.max_depth,
            "min_samples_split": sklearn_model.min_samples_split,
            "min_samples_leaf": sklearn_model.min_samples_leaf,
        },
        "data": {
            "total": len(X),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "train": train_results,
        "val": val_results,
        "test": test_results,
        "id3_test": id3_results,
        "class_names": {code: CONSTITUTION_NAMES[code] for code in LABEL_CLASSES},
    }
    results_path = "data/training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    print(f"  评估结果已保存至: {results_path}")

    print(f"\n{'=' * 60}")
    print("训练完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
