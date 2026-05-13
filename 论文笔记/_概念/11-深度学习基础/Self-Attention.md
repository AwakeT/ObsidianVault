---
type: concept
aliases: [自注意力]
---

# Self-Attention

## 定义
Transformer 中 Query、Key、Value 均来自同一输入序列的注意力变体，用于建模序列内部的依赖关系。

## 核心要点
1. Q=K=V 来自同一输入，计算序列中每个位置对其他所有位置的关注度
2. 多头自注意力（Multi-Head）允许关注不同子空间的信息
3. 在视觉中用于全局上下文聚合

## 代表工作
- [[CREStereo]]: 线性自注意力增强特征图
- [[FoundationStereo]]: DT 中的多头自注意力

## 相关概念
- [[Transformer]]
- [[Positional Encoding]]
