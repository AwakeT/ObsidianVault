---
type: concept
aliases: [凸上采样]
---

# Convex Upsampling

## 定义
通过预测局部凸组合权重将低分辨率预测上采样到全分辨率的可学习上采样方法，保留锐利边缘。

## 数学形式
$$d^{\text{full}}(x,y) = \sum_{(i,j) \in \mathcal{N}} w_{ij}(x,y) \cdot d^{\text{low}}(i,j)$$

其中 $w_{ij} \geq 0$, $\sum w_{ij} = 1$（通过 softmax 保证）。

## 核心要点
1. 网络预测 $k \times k$ 邻域的凸组合权重（通常 $k = 3$，即 9 个权重 per 像素）
2. 相比双线性插值，能保留锐利的深度不连续边界
3. 首次在 RAFT 中提出，后被 RAFT-Stereo、CREStereo 等广泛采用

## 代表工作
- [[RAFT]]: 首次提出 convex upsampling
- [[RAFT-Stereo]]: 从 1/8 分辨率上采样视差
- [[CREStereo]]: 从 1/4 分辨率上采样视差
- [[FoundationStereo]]: 最终输出上采样

## 相关概念
- [[RAFT]]
- [[Correlation Volume]]
