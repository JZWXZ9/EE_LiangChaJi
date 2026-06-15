"""
AI凉茶机 — 中医体质分类器（部署版）
====================================

纯 Python 标准库实现，无 ML 依赖。加载 JSON 决策树模型，根据 10 个问诊回答
和 3 个望诊特征预测体质编码（A-I）。

用法:
    from deploy.classifier import TCMClassifier

    clf = TCMClassifier("deploy/model.json")
    result = clf.predict("CBBAAABACA", [1, 0, 1])  # features: [舌苔, 眼圈, 脸色]
    print(result)  # → "D" (阴虚质)
"""

import json
import os
from typing import Union

# ============================================================
# 编码映射
# ============================================================

Q_ENCODE = {"A": 0, "B": 1, "C": 2, "D": 3}

CONSTITUTION_NAMES = {
    "A": "平和质", "B": "气虚质", "C": "阳虚质",
    "D": "阴虚质", "E": "痰湿质", "F": "湿热质",
    "G": "血瘀质", "H": "气郁质", "I": "特禀质",
}

# ============================================================
# 分类器
# ============================================================


class TCMClassifier:
    """中医体质分类器。

    加载训练好的 JSON 决策树模型，提供 predict() 方法进行体质分类。

    模型格式（sklearn 二叉判定树）:
        {"feature": "Q1", "threshold": 1.5, "majority": "C",
         "left": {...}, "right": {...}}
        叶子节点: {"feature": None, "majority": "C"}
    """

    def __init__(self, model_path: str):
        """
        初始化分类器，加载 JSON 模型文件。

        Args:
            model_path: 决策树 JSON 模型文件路径

        Raises:
            FileNotFoundError: 模型文件不存在
            ValueError: 模型文件格式无效
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        with open(model_path, "r", encoding="utf-8") as f:
            self._tree = json.load(f)

        self._validate_tree(self._tree)

    @staticmethod
    def _validate_tree(node: dict):
        """校验模型结构。"""
        if not isinstance(node, dict):
            raise ValueError(f"模型节点必须是 dict，实际: {type(node)}")
        if "majority" not in node:
            raise ValueError("模型节点缺少 'majority' 字段")
        if node.get("feature") is not None:
            if "threshold" not in node:
                raise ValueError("内部节点缺少 'threshold' 字段")
            if "left" not in node or "right" not in node:
                raise ValueError("内部节点缺少 'left' 或 'right' 子树")
            TCMClassifier._validate_tree(node["left"])
            TCMClassifier._validate_tree(node["right"])

    def predict(self, answers: str, features: list[int]) -> str:
        """
        根据问诊回答和望诊特征预测体质编码。

        Args:
            answers: 长度为 10 的字符串，仅含 A/B/C/D
            features: 长度为 3 的列表 [舌苔, 眼圈, 脸色]，仅含 0/1

        Returns:
            str: 体质编码 A-I

        Raises:
            ValueError: 输入格式无效
        """
        self._validate_input(answers, features)
        return self._traverse(self._tree, answers, features)

    def _validate_input(self, answers: str, features: list[int]):
        """校验输入格式。"""
        if not isinstance(answers, str) or len(answers) != 10:
            raise ValueError(
                f"answers 必须是长度为 10 的字符串，实际长度: {len(answers) if isinstance(answers, str) else 'N/A'}"
            )
        for i, ch in enumerate(answers):
            if ch not in "ABCD":
                raise ValueError(f"answers 第 {i+1} 个字符 '{ch}' 无效，仅允许 A/B/C/D")

        if not isinstance(features, list) or len(features) != 3:
            raise ValueError(
                f"features 必须是长度为 3 的列表，实际长度: {len(features) if isinstance(features, list) else 'N/A'}"
            )
        for i, v in enumerate(features):
            if v not in (0, 1):
                raise ValueError(f"features 第 {i+1} 个值 '{v}' 无效，仅允许 0/1")

    def _traverse(self, node: dict, answers: str, features: list[int]) -> str:
        """递归遍历决策树。"""
        # 叶子节点：返回多数类
        if node["feature"] is None:
            return node["majority"]

        feat_name = node["feature"]
        threshold = node["threshold"]

        # 获取特征值，特征值训练时候用的是int，利用索引
        if feat_name.startswith("Q"):
            idx = int(feat_name[1:]) - 1  # Q1 → 0, Q10 → 9
            value = Q_ENCODE[answers[idx]]
        else:
            idx = int(feat_name[1:]) - 1  # F1 → 0, F3 → 2
            value = features[idx]

        # 二叉判定
        if value <= threshold:
            return self._traverse(node["left"], answers, features)
        else:
            return self._traverse(node["right"], answers, features)

    @property
    def model_info(self) -> dict:
        """返回模型基本信息。"""
        return {
            "format": "sklearn CART binary tree",
            "constitutions": CONSTITUTION_NAMES,
        }


# ============================================================
# 便捷函数
# ============================================================


def load_classifier(model_path: str = "deploy/model.json") -> TCMClassifier:
    """工厂函数：加载分类器。"""
    return TCMClassifier(model_path)


# ============================================================
# 命令行测试
# ============================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # 默认测试用例（来自 proof.md 第六节标准模式）
    # features 顺序: [舌苔, 眼圈, 脸色]
    test_cases = [
        # (answers, features, expected_label, description)
        ("AAAAAAAAAA", [0, 0, 0], "A", "平和质标准模式"),
        ("DCACABAACC", [1, 0, 1], "B", "气虚质标准模式"),
        ("CDACABBACC", [1, 1, 1], "C", "阳虚质标准模式"),
        ("CADAABAACC", [1, 1, 1], "D", "阴虚质标准模式"),
        ("CAADCBAACB", [1, 1, 0], "E", "痰湿质标准模式"),
        ("CACCDBABCD", [1, 0, 1], "F", "湿热质标准模式"),
        ("CBAAACDACC", [1, 1, 1], "G", "血瘀质标准模式"),
        ("CABAADABDC", [0, 0, 0], "H", "气郁质标准模式"),
        ("AAAAABADCB", [0, 0, 0], "I", "特禀质标准模式"),
    ]

    # 默认模型路径：优先同级目录，其次上级 deploy 目录
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    elif os.path.exists("model.json"):
        model_path = "model.json"
    else:
        model_path = os.path.join(os.path.dirname(__file__), "model.json")

    print(f"加载模型: {model_path}")
    clf = TCMClassifier(model_path)

    print(f"\n测试用例验证:")
    print(f"{'answers':12s} {'F123':5s} {'预测':4s} {'期望':4s} {'结果':6s}  {'说明'}")
    print("-" * 60)

    passed = 0
    failed = 0

    for answers, features, expected, desc in test_cases:
        pred = clf.predict(answers, features)
        status = "✓" if pred == expected else "✗"
        if pred == expected:
            passed += 1
        else:
            failed += 1
        f_str = "".join(str(v) for v in features)
        print(f"{answers:12s} {f_str:5s} {pred:4s} {expected:4s} {status:6s} {desc}")

    print(f"\n通过: {passed}/{passed + failed}")
    if failed > 0:
        print(f"失败: {failed} 条（标准模式应该完全匹配）")
