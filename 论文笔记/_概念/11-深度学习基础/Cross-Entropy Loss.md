---
type: concept
aliases: [交叉熵损失, CE Loss]
---

# Cross-Entropy Loss

## 定义
衡量预测概率分布与真实分布之间差异的损失函数，是分类与语言模型 next-token 预测的标准训练目标。

## 数学形式
$$
\mathcal{L}_{\text{CE}} = -\sum_{i} y_i \log \hat{y}_i
$$

## 核心要点
1. **语言建模标配**: 对每个位置的目标 token 求负对数似然。
2. **可叠加**: 多任务/多流训练可对不同序列分别计算并相加，如 $\mathcal{L} = \mathcal{L}_{\text{ntp}} + \mathcal{L}_{\text{mtp}}$。

## 代表工作
- [[LocateAnything]]: 联合最小化 NTP 流与 MTP 块级流的交叉熵损失。

## 相关概念
- [[Next-Token Prediction]]
- [[Softmax]]
