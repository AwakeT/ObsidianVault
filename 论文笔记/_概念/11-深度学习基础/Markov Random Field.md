---
type: concept
aliases: [马尔可夫随机场, MRF]
---

# Markov Random Field

## 定义
定义在无向图上的概率图模型，每个节点的条件概率仅依赖于其邻居节点，常用于图像分割和立体匹配的全局优化。

## 核心要点
1. 2D MRF 能量最小化是 NP-hard
2. 近似方法：Graph Cut、Belief Propagation、SGM
3. SGM 通过多方向 1D DP 高效近似 2D MRF

## 代表工作
- [[SGBM]]: 近似 2D MRF 优化
- [[NMRF]]: Neural Markov Random Field

## 相关概念
- [[Energy Function]]
- [[Dynamic Programming]]
