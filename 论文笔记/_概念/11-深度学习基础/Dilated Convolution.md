---
type: concept
aliases: [空洞卷积, Atrous Convolution]
---

# Dilated Convolution

## 定义
在标准卷积核的采样点之间插入空洞（dilation）以扩大感受野而不增加参数量的卷积变体。

## 核心要点
1. dilation rate $r$ 使 $k \times k$ 核的有效感受野扩大为 $(k + (k-1)(r-1))^2$
2. 不增加参数和计算量即可扩大感受野
3. 在语义分割（DeepLab）和立体匹配（HITNet）中广泛使用

## 代表工作
- [[HITNet]]: Tile Update CNN 中使用空洞卷积
- [[CREStereo]]: AGCL 中 2D 搜索使用 dilation

## 相关概念
- [[Depthwise Separable Convolution]]
- [[Receptive Field]]
