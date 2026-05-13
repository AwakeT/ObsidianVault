---
type: concept
aliases: [可分离卷积]
---

# Separable Convolution

## 定义
将多维卷积分解为多个低维卷积的通用策略，包括空间可分离（行/列分解）和深度可分离（通道/空间分解）。

## 核心要点
1. 空间可分离：$k \times k$ → $k \times 1$ + $1 \times k$
2. 深度可分离：depthwise + pointwise
3. FoundationStereo 的 APC 是空间-视差维度的分离

## 代表工作
- [[FoundationStereo]]: APC 将 3D 卷积分解为空间 + 视差两个方向

## 相关概念
- [[Depthwise Separable Convolution]]
- [[Dilated Convolution]]
