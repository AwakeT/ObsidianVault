---
type: concept
aliases: [Dynamic 4D Reconstruction, 4D Scene Reconstruction, 动态4D重建]
---

# 4D Reconstruction

## 定义
4D Reconstruction 指从图像或视频中恢复三维场景随时间变化的几何、相机和对应关系，核心对象是空间 $x,y,z$ 加时间 $t$ 的动态场景表示。

## 核心要点
1. 不只估计每帧深度，还要建立跨时间的物理点对应。
2. 动态物体、遮挡和相机运动会显著增加难度。
3. 可以输出点云、深度、相机位姿、3D point tracks 等多种形式。

## 代表工作
- [[D4RT]]: 用统一查询接口完成动态 4D 重建与跟踪。
- [[VGGT]]: 前馈式 3D 重建基线，但动态对应能力有限。

## 相关概念
- [[Point Tracking]]
- [[Point Cloud]]
- [[Camera Pose Estimation]]
- [[Video Depth Estimation]]
