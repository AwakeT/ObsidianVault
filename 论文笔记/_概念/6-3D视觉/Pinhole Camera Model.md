---
type: concept
aliases: [针孔相机模型, Pinhole Model]
---

# Pinhole Camera Model

## 定义
Pinhole Camera Model 是最基础的相机投影模型，用焦距和主点描述 3D 点到 2D 像素平面的透视投影。

## 核心要点
1. 常用于相机内参估计与多视图几何推导。
2. 简化形式通常假设主点位于图像中心。
3. 畸变相机需要额外非线性 refinement 或畸变模型。

## 代表工作
- [[D4RT]]: 假设主点为 $(0.5,0.5)$，由预测 3D 点和 2D 坐标估计 $f_x,f_y$。

## 相关概念
- [[Camera Pose Estimation]]
- [[Structure from Motion]]
