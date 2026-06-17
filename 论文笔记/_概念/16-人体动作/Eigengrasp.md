---
type: concept
aliases: [Eigengrasp, eigengrasps, 特征抓取]
---

# Eigengrasp

## 定义
通过 [[PCA]] 在高维手部关节姿态空间中提取的主成分基，用少量低维系数描述人手抓取姿态的主要变化模式，降低灵巧手控制的维度复杂度。

## 核心要点
1. 把高复杂度的灵巧抓取问题降到低维系数空间。
2. Qwen-VLA 对 45 维轴角手部关节姿态做 PCA 取前 10 个主成分，每手用 10 个 eigengrasp 系数 + 6 维腕部 SE(3)，双手共 32 维。

## 代表工作
- Ciocarlie et al. 2007: 原始提出
- Yuan et al. 2025: 跨形态灵巧抓取
- [[Qwen-VLA]]: 人体手部动作紧凑表示

## 相关概念
- [[PCA]]
- [[MANO]]
