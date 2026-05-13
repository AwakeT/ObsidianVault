---
type: concept
aliases: [动态规划, DP]
---

# Dynamic Programming

## 定义
通过将问题分解为重叠子问题并存储中间结果来高效求解最优化问题的算法设计范式。

## 核心要点
1. SGM 中沿每个方向的路径代价递推即为 1D 动态规划
2. 2D MRF 优化是 NP-hard，但 1D 可用 DP 在线性时间求解
3. SGM 通过多方向 1D DP 的聚合近似 2D 全局最优

## 代表工作
- [[SGBM]]: 多方向 1D DP 代价聚合

## 相关概念
- [[Markov Random Field]]
- [[Cost Aggregation]]
- [[Energy Function]]
