---
type: concept
aliases: [Fourier Feature Embedding, Fourier位置编码]
---

# Fourier Features

## 定义
Fourier Features 是将连续坐标映射到多频率正弦/余弦特征的编码方式，使神经网络更容易表达高频空间变化。

## 核心要点
1. 常用于 NeRF、坐标网络和隐式表示。
2. 对连续坐标 query 尤其有用。
3. D4RT 用 Fourier feature embedding 编码 2D query 坐标 $(u,v)$。

## 代表工作
- [[D4RT]]: 用 Fourier features 构造 pointwise query token。

## 相关概念
- [[Positional Encoding]]
- [[Query-Based Decoder]]
