---
type: concept
aliases: [Point Map, Pointmap, 点云图]
---

# Point Map

## 定义
将每个像素映射到 3D 空间坐标的稠密表示，$P \in \mathbb{R}^{3 \times H \times W}$，由 DUSt3R 提出用于替代传统深度图+相机参数的解耦表示。

## 数学形式

$$$P_i(y) \in \mathbb{R}^3$，将像素 $y$ 映射到参考坐标系下的 3D 点$$

## 核心要点
1. 在参考帧坐标系下定义，隐式编码相机信息
2. 深度图和相机参数可从 pointmap 推导
3. 可自然扩展到动态场景（每时间步一个 pointmap）

## 代表工作
- [[DUSt3R]]: 提出 pointmap 表示
- [[VGGT]]: 前馈预测 pointmap
- [[MonST3R]]: 时变 pointmap 处理动态场景

## 相关概念
- [[DUSt3R]]
- [[Depth Map]]
