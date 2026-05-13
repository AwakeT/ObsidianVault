---
type: concept
aliases: [Trajectory Smoothness, 轨迹平滑性]
---

# Trajectory Smoothness

## 定义
约束相邻帧相机位姿变化平滑的正则化项，防止优化中出现不自然的位姿跳变。

## 数学形式

$$$\mathcal{L}_{\text{smooth}} = \sum_t (\|(R^t)^\top R^{t+1} - I\|_F + \|(R^t)^\top(T^{t+1} - T^t)\|_2)$$$

## 核心要点
1. 同时约束旋转和平移的平滑性
2. MonST3R 用于全局位姿优化

## 代表工作
- [[MonST3R]]: 全局优化的平滑正则项

## 相关概念
- [[Global Optimization]]
