---
type: concept
aliases: [Camera Pose, Camera Extrinsics, 相机位姿估计, 相机外参估计]
---

# Camera Pose Estimation

## 定义
Camera Pose Estimation 指估计相机在世界坐标或相对坐标中的位置与朝向，通常包括旋转和平移。

## 核心要点
1. 多视图几何中常由特征匹配、PnP、BA 或点云对齐求解。
2. 前馈模型可直接预测或通过 3D 点对应间接恢复相机位姿。
3. D4RT 通过同一批点在不同相机参考系下的 3D 坐标，用 Umeyama 对齐求相对位姿。

## 代表工作
- [[D4RT]]: 用统一 query decoder 推导相机外参。
- [[VGGT]]: 直接输出相机参数的前馈重建模型。

## 相关概念
- [[Umeyama Algorithm]]
- [[PnP]]
- [[Structure from Motion]]
