---
type: concept
aliases: [Aleatoric Uncertainty, 数据不确定性]
---

# Aleatoric Uncertainty

## 定义
由数据本身的噪声和模糊性引起的不可约不确定性，可通过预测输出分布的参数来建模。

## 数学形式

$$$\mathcal{L} = \|\Sigma \odot (\hat{y} - y)\| - \alpha \log \Sigma$$$

## 核心要点
1. 与 epistemic uncertainty（模型不确定性）相对
2. 通过预测方差/精度图来加权损失
3. VGGT 用于深度和点云图损失

## 代表工作
- [[VGGT]]: 预测不确定性图 $\Sigma$ 加权深度和点云损失

## 相关概念
- [[Multi-Task Learning]]
