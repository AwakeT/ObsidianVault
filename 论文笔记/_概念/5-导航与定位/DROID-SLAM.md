---
type: concept
aliases: [DROID-SLAM]
---

# DROID-SLAM

## 定义
基于深度学习的稠密 SLAM 方法，使用可微 Bundle Adjustment 实现端到端位姿和深度估计。

## 核心要点
1. 使用 correlation volumes 和 GRU 更新
2. 需要已知相机内参
3. 在静态场景表现优异但对动态场景鲁棒性不足

## 代表工作
- [[MonST3R]]: 不需 GT 内参即达到可比精度

## 相关概念
- [[Visual SLAM]]
- [[Bundle Adjustment]]
