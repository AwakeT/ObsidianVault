---
type: concept
aliases: [PCA, Principal Component Analysis, 主成分分析]
---

# PCA

## 定义
通过正交线性变换把高维数据投影到方差最大的若干主成分上，实现降维与去冗余。

## 核心要点
1. 保留前 $k$ 个主成分可捕获数据主要变化模式。
2. Qwen-VLA 对 45 维轴角手部关节姿态做 PCA，取前 10 个主成分系数（[[Eigengrasp|eigengrasps]]），紧凑表示人手姿态变化。

## 代表工作
- [[Qwen-VLA]]: 人体手部姿态压缩为 eigengrasp
- [[Eigengrasp]]

## 相关概念
- [[Eigengrasp]]
- [[MANO]]
