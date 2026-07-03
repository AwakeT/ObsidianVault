---
type: concept
aliases: [DeltaNet, Delta Rule, 增量规则]
---

# DeltaNet

## 定义
一类基于 **delta rule（增量规则）** 的线性注意力 / 快权重模型：把状态视为关联记忆矩阵，每步用当前 key/value 对做一次误差修正式更新，等价于对 KV 重建误差做一步梯度下降（linear Test-Time Training）。

## 数学形式
$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t\,\nabla \|\mathbf{S}_{t-1}\mathbf{K}_t - \mathbf{V}_t\|^2
= \mathbf{S}_{t-1} + \beta_t(\mathbf{V}_t - \mathbf{S}_{t-1}\mathbf{K}_t)\mathbf{K}_t^\top
$$

## 核心要点
1. key 指定"写到哪"、value 指定"写什么"、学习率 $\beta_t$ 门控 memory plasticity。
2. 是 [[Linear Attention]] 的改进：显式擦除旧关联再写入新关联，缓解容量冲突。
3. 与 [[Test-Time Training]] 数学同构，是"state 即 fast weight"的典型实现。

## 代表工作
- DeltaNet / Gated DeltaNet：语言建模的线性复杂度序列模型。
- [[TTT3R]]：借 DeltaNet 视角分析 CUT3R 状态更新并推导置信度学习率。

## 相关概念
- [[Linear Attention]]
- [[Test-Time Training]]
- [[Gated Linear Attention]]
- [[Memory Mechanism]]
