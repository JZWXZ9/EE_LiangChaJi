# AI凉茶机 — 中医体质分类系统

基于**加权余弦相似度**的中医九种体质自动分类。通过 10 个问诊问题和 3 个望诊特征，与 proof.md 九种体质标准模式匹配，预测用户体质类型。

---

## 快速开始

### 安装

无需安装第三方库。将 `deploy/` 目录拷贝至目标环境即可。

```bash
python deploy/classifier.py          # 运行内置测试
python deploy/example.py             # 查看使用示例
```

### 基本用法

```python
from deploy.classifier import CosSimClassifier

clf = CosSimClassifier()
result = clf.predict("DCACABAACC", [1, 0, 1])  # features: [舌苔, 眼圈, 脸色]
print(result)  # → "B" (气虚质)
```

| 参数 | 格式 | 示例 |
|-----|------|------|
| `answers` | 长度 10 的字符串，仅含 A/B/C/D | `"DCACABAACC"` |
| `features` | 长度 3 的列表 [舌苔, 眼圈, 脸色] | `[1, 0, 1]` |
| 返回值 | 单个字母 A–I | `"B"` |

### 体质编码

| A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|
| 平和质 | 气虚质 | 阳虚质 | 阴虚质 | 痰湿质 | 湿热质 | 血瘀质 | 气郁质 | 特禀质 |

---

## 数据集

### 数据来源

基于周萍《基于中医体质理论与六大茶类的体质茶疗法成果研究》(2026) 的中医体质分类理论，参照《中医体质分类与判定》国家标准设计特征体系。

- **理论基础文档**：[doc/proof.md](doc/proof.md) — 体质分类标准、问题设计依据、茶疗推荐
- **问题与评判标准**：[doc/question.md](doc/question.md) — 十问问卷与体质判别速查表
- **需求文档**：[doc/proposal.md](doc/proposal.md) — 接口规格与技术方案

### 数据特征

| 类别 | 数量 | 说明 |
|-----|------|------|
| 问诊特征 (Q1–Q10) | 10 个 | A/B/C/D 四选项，对应症状严重程度 |
| 望诊特征 (F1–F3) | 3 个 | 脸色/舌苔/眼圈，0=正常, 1=异常 |
| 体质类别 | 9 种 | A–I，每类约 111 条，均衡分布 |
| 总样本量 | 1000 条 | 含真实噪声扰动（核心特征 5-10%，非核心 15-25%） |

### 数据划分

| 子集 | 样本数 | 比例 |
|-----|-------|------|
| 训练集 | 700 | 70% |
| 验证集 | 150 | 15% |
| 测试集 | 150 | 15% |

---

## 分类方法

### 加权余弦相似度

采用**基于原型向量的加权余弦相似度**分类，无需训练：

| 项目 | 说明 |
|-----|------|
| 方法 | 加权余弦相似度（Weighted Cosine Similarity） |
| 原型来源 | proof.md 第六节九种体质标准模式 |
| 编码方式 | Q: A=0/B=1/C=2/D=3（序数），F: 0/1 |
| 核心特征加权 | 加粗诊断特征 ×2.0 |
| 平和质保护 | 向量模长 < 2.3 时直接判为平和质 |
| 依赖 | 仅 Python 标准库 + `deploy/standard_patterns.json` |

### 人工标注验证

从 1000 条中随机抽取 100 条进行人工标注，余弦相似度分类器达到：

| 指标 | 结果 |
|-----|------|
| 一致率 | **90.00%** |
| 目标 | ≥ 90% |
| 达标 | ✓ |

> 10 条差异中，8 条为气虚↔阳虚互混（中医体质分类公认难点）。平和质 0 条误判。

### 决策树（保留，供日后扩展）

决策树模型和训练脚本保留在 `deploy_tree/` 和 `scripts/` 中。当获取到可靠的真实临床数据后，可重新启用：

| 模型 | 测试集 Macro F1 | 说明 |
|-----|---------------|------|
| sklearn CART | 95.88% | 二叉判定树，需 sklearn 训练 |
| 手写 ID3 | 88.58% | 多叉树，纯 Python |
| **余弦相似度（当前）** | **90.00%** (vs 人工) | 零训练，直接使用标准模式 |

---

## 项目结构

```
EE_LiangChaJi/
├── doc/
│   ├── proof.md              # 理论依据与数据生成标准
│   ├── proposal.md           # 需求文档
│   ├── question.md           # 问诊问题与评判标准
│   ├── update/               # 阶段开发报告
│   │   ├── dev_report_phase1.md
│   │   └── dev_report_phase2.md
├── data/
│   ├── tcm_data_labeled.csv  # 1000 条带标签训练数据
│   ├── tcm_data_unlabeled.csv# 100 条无标签验证子集
│   ├── tcm_data_manual.csv   # 人工标注结果
│   ├── label_diff.csv        # AI vs 人工差异明细
│   └── training_results.json # 模型训练完整指标
├── scripts/
│   ├── generate_data.py      # 数据生成脚本
│   ├── train_model.py        # 模型训练脚本
│   └── compare_labels.py     # 标注误差分析脚本
├── deploy/                   # ★ 部署目录（余弦相似度，可独立拷贝）
│   ├── classifier.py         # CosSimClassifier 纯 Python 分类器
│   ├── standard_patterns.json# 九种体质标准模式（原型向量）
│   └── example.py            # 使用示例
├── deploy_tree/              # 决策树方案（保留，供日后扩展）
│   ├── classifier.py         # TCMClassifier（需 model.json）
│   ├── model.json            # 训练好的决策树模型
│   └── example.py            # 决策树使用示例
└── README.md                 # 本文档
```

---

## 参考

- 周萍. 基于中医体质理论与六大茶类的体质茶疗法成果研究[J]. 福建茶叶, 2026, 48(1).
- 中华中医药学会. 中医体质分类与判定(ZYYXH/T157-2009)[S]. 2009.
- 潘康宁 等. 基于机器学习的中医体质分类研究[J]. 中国医疗设备, 2024, 39(1): 6-11.
