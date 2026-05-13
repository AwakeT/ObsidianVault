---
type: concept
aliases: [PnP, Perspective-n-Point]
---

# PnP

## 定义
从 n 个 3D-2D 点对应关系估计相机位姿的经典算法。

## 核心要点
1. 最少需要 4 个非共面点（P3P 需 3 个+1 个验证）
2. 通常与 RANSAC 结合提高鲁棒性
3. DUSt3R 从 pointmap 恢复位姿时使用 PnP

## 代表工作
- [[VGGT]]: 相机参数可通过 PnP 从 pointmap 推导
- [[MonST3R]]: RANSAC+PnP 从 pointmap 恢复相对位姿

## 相关概念
- [[RANSAC]]
- [[Bundle Adjustment]]
