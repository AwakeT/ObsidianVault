---
type: concept
aliases: [Motion Averaging, 运动平均]
---

# Motion Averaging

## 定义
从多个噪声相对位姿估计恢复全局一致的绝对位姿的方法，包括旋转平均和平移三角化。

## 数学形式

$$$\hat{R}_q = \text{median}\{R_{d_i} \cdot \hat{R}_{q,d_i}\}$$$

## 核心要点
1. 旋转平均：聚合多个绝对旋转估计
2. 相机中心三角化：从平移方向恢复度量位置
3. Reloc3r 使用无参数 motion averaging 替代神经网络学习度量尺度

## 代表工作
- [[Reloc3r]]: 核心模块，从相对位姿恢复绝对位姿

## 相关概念
- [[Rotation Averaging]]
- [[Camera Center Triangulation]]
