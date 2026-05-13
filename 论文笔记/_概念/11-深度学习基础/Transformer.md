---
type: concept
aliases: [注意力机制, Attention Mechanism]
---

# Transformer

## 定义
基于自注意力机制的序列建模架构，通过 Query-Key-Value 机制计算全局依赖关系。

## 数学形式
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

## 核心要点
1. 自注意力可捕获任意距离的依赖，不受卷积感受野限制
2. 计算复杂度 $O(n^2)$，FlashAttention 等优化可降低实际耗时
3. 在立体匹配中用于特征增强（CREStereo）或 cost volume 全局推理（FoundationStereo DT）

## 代表工作
- [[FoundationStereo]]: Disparity Transformer 在视差维度做全局自注意力
- [[CREStereo]]: 特征图上的线性注意力

## 相关概念
- [[Self-Attention]]
- [[Positional Encoding]]
- [[FlashAttention]]
