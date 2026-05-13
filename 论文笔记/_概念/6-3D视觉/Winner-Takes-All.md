---
type: concept
aliases: [WTA, 胜者全取]
---

# Winner-Takes-All

## 定义
在聚合后的代价体积中选取代价最小的视差作为最终预测的简单策略。

## 数学形式
$$d^*(p) = \arg\min_d S(p, d)$$

## 核心要点
1. 最简单的视差选择策略，计算效率高
2. 输出整数像素视差，需配合亚像素精化
3. 在深度学习方法中被 soft-argmin 替代以支持端到端训练

## 代表工作
- [[SGBM]]: WTA + 抛物线亚像素精化

## 相关概念
- [[Soft-Argmin]]
- [[Cost Aggregation]]
