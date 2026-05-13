---
type: concept
aliases: [Bundle Adjustment, BA, 光束法平差]
---

# Bundle Adjustment

## 定义
同时优化相机参数和 3D 点坐标以最小化重投影误差的非线性最小二乘问题，是 SfM 和 SLAM 的核心组件。

## 数学形式

$$$\min_{P,X} \sum_{i,j} \|\pi(P_i, X_j) - x_{ij}\|^2$$$

## 核心要点
1. 传统 SfM 管线的最终优化步骤
2. Levenberg-Marquardt 或 Gauss-Newton 求解
3. VGGT 证明前馈网络可以替代大部分 BA 的功能

## 代表工作
- [[VGGT]]: 可选 BA 后处理提升精度至 93.5 AUC@30
- [[COLMAP]]: BA 的标准实现

## 相关概念
- [[Structure from Motion]]
- [[COLMAP]]
- [[PnP]]
