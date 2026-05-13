---
type: concept
aliases: [互信息, MI]
---

# Mutual Information

## 定义
度量两个随机变量之间统计依赖程度的信息论度量，等于联合熵与各自边缘熵之差。

## 数学形式
$$MI(X;Y) = H(X) + H(Y) - H(X,Y)$$

## 核心要点
1. MI = 0 当且仅当 X, Y 统计独立
2. 对单调灰度变换不变，适合不同传感器/光照条件的匹配
3. SGM 原始论文使用 MI 作为匹配代价

## 代表工作
- [[SGBM]]: 原始 SGM 的匹配代价函数

## 相关概念
- [[Census Transform]]
- [[Birchfield-Tomasi]]
