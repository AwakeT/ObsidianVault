---
type: concept
aliases: [倾斜平面, 倾斜平面假设]
---

# Slanted Plane

## 定义
将视差图表示为分段倾斜平面的几何假设。每个区域（tile）用平面参数 $(d, d_x, d_y)$ 表示视差及其空间梯度。

## 数学形式
$$d(i,j) = d_0 + (i - c_x) \cdot d_x + (j - c_y) \cdot d_y$$

## 核心要点
1. 比逐像素常数视差更紧凑，能建模倾斜和曲面
2. HITNet 将每个 4x4 tile 表示为倾斜平面假设
3. 传统方法（PatchMatch Stereo）中也广泛使用

## 代表工作
- [[HITNet]]: tile hypothesis $(d, d_x, d_y, \mathbf{p})$
- [[SOS Stereo]]: slanted support windows

## 相关概念
- [[Stereo Matching]]
- [[HITNet]]
