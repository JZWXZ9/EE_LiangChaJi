"""
AI凉茶机 — 部署分类器使用示例
==============================

演示 TCMClassifier 的基本用法：加载模型、单条预测。
"""

import sys
import os

# 输出中文
sys.stdout.reconfigure(encoding="utf-8")
# 优先搜索当前路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import TCMClassifier

# 模型路径
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_MODEL_DIR, "model.json")


def main():
    clf = TCMClassifier(MODEL_PATH)

    # 一个典型的气虚质用户
    answers = "DCACABAACC"   # 10 个问题回答 (Q1–Q10)
    features = [1, 0, 1]     # 望诊: 舌苔异常, 眼圈正常, 脸色异常

    result = clf.predict(answers, features)

    print(f"问诊回答: {answers}")
    print(f"望诊特征 (舌苔/眼圈/脸色): {features}")
    print(f"体质编码: {result}  (气虚质)")


if __name__ == "__main__":
    main()
