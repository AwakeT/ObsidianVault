---
type: concept
aliases: [Quantile Normalization, 分位数归一化]
---

# Quantile Normalization

## 定义
用数据的分位数（如 1% 与 99%）做线性映射并裁剪，去除不同来源间的尺度差异，同时保留各源内部相对结构。

## 数学形式
$$
\tilde{a}_d = 2\cdot\frac{a_d - q_{01}^k}{q_{99}^k - q_{01}^k} - 1
$$

裁剪到 $[-1, 1]$。

## 核心要点
1. 相比 min-max，对离群值更鲁棒（用百分位而非极值）。
2. Qwen-VLA 对每个数据集每个动作维做 per-dataset 分位数归一化，统一跨形态/动作空间的尺度。

## 代表工作
- [[Qwen-VLA]]: 动作维度归一化

## 相关概念
- [[Embodiment-Aware Prompt Conditioning]]
