"""
AI凉茶机 — 余弦相似度分类器使用示例
====================================

演示 CosSimClassifier 的基本用法。
"""

import sys
import os

# 输出中文
sys.stdout.reconfigure(encoding="utf-8")
# 优先搜索当前路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import CosSimClassifier


def main():
    clf = CosSimClassifier()

    # 气虚质用户
    answers = "DCACABAACC"
    features = [1, 0, 1]     # [舌苔, 眼圈, 脸色]

    result = clf.predict(answers, features)
    print(f"问诊: {answers}")
    print(f"望诊 (舌苔/眼圈/脸色): {features}")
    print(f"体质: {result} (气虚质)")

    # 查看与所有体质的相似度
    detail = clf.predict_with_scores(answers, features)
    print(f"\n各体质余弦相似度:")
    for code, score in sorted(detail["scores"].items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        print(f"  {code}: {score:.4f}  {bar}")


if __name__ == "__main__":
    main()
