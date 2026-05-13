---
type: concept
aliases: [能量函数, 目标函数]
---

# Energy Function

## 定义
立体匹配中将数据项（匹配代价）和平滑项（视差连续性约束）组合为统一优化目标的函数。

## 数学形式
$$E(D) = \sum_p C(p, d_p) + \sum_{(p,q)} V(d_p, d_q)$$

## 核心要点
1. 数据项度量像素级匹配质量
2. 平滑项惩罚邻域视差不连续（P1/P2 惩罚）
3. 最小化能量函数等价于求最优视差图

## 代表工作
- [[SGBM]]: $E(D)$ = 数据项 + P1 惩罚 + P2 惩罚

## 相关概念
- [[Markov Random Field]]
- [[Cost Aggregation]]
