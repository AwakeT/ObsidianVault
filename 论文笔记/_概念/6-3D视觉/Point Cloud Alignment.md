---
type: concept
aliases: [Point Cloud Alignment, 点云对齐]
---

# Point Cloud Alignment

## 定义
将多个局部坐标系下的点云变换到统一全局坐标系的过程。

## 核心要点
1. DUSt3R 使用全局优化对齐成对 pointmap
2. MonST3R 扩展到时变 pointmap 的动态对齐

## 代表工作
- [[MonST3R]]: 对齐损失约束全局与局部 pointmap 的一致性

## 相关概念
- [[Point Map]]
- [[Bundle Adjustment]]
