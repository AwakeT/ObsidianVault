---
type: concept
aliases: [Structure from Motion, SfM, 运动恢复结构]
---

# Structure from Motion

## 定义
从多视图图像同时恢复三维结构和相机运动参数的经典计算机视觉问题。

## 核心要点
1. 传统管线：特征提取→匹配→三角化→Bundle Adjustment
2. COLMAP 是最流行的 SfM 实现
3. VGGT 等前馈方法正在挑战传统 SfM 管线

## 代表工作
- [[VGGT]]: 前馈替代 SfM
- [[VGGSfM]]: 端到端可微 SfM

## 相关概念
- [[Bundle Adjustment]]
- [[COLMAP]]
- [[Epipolar Constraint]]
