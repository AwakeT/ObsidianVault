---
type: concept
aliases: [深度可分离卷积, DSConv]
---

# Depthwise Separable Convolution

## 定义
将标准卷积分解为逐通道的 depthwise 卷积和 1x1 的 pointwise 卷积，大幅降低计算量。

## 数学形式
标准卷积 MACs: $k^2 \times C_{in} \times C_{out}$
DSConv MACs: $k^2 \times C_{in} + C_{in} \times C_{out}$
降低因子: $\approx k^2$（当 $C_{out} \gg 1$ 时）

## 核心要点
1. MobileNetV1 的核心模块
2. 3D 扩展为 $k \times k \times k$ depthwise + $1 \times 1 \times 1$ pointwise
3. 降低计算量约 $k^2$ 倍（2D）或 $k^3$ 倍（3D）

## 代表工作
- [[MobileStereoNet]]: 将 DSConv 从 2D 扩展到 3D 用于 cost volume 处理
- [[MobileNetV1]]: 首次提出

## 相关概念
- [[Inverted Residual Block]]
- [[Separable Convolution]]
