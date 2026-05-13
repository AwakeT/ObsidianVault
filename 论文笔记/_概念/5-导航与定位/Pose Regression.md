---
type: concept
aliases: [Pose Regression, 位姿回归]
---

# Pose Regression

## 定义
使用神经网络从图像直接回归 6-DoF 相机位姿（旋转+平移）的方法，分为绝对位姿回归 (APR) 和相对位姿回归 (RPR)。

## 核心要点
1. APR: 场景专用，快速但泛化差
2. RPR: 跨场景泛化但精度曾不及 APR
3. Reloc3r 证明 RPR 可以超越 APR

## 代表工作
- [[Reloc3r]]: RPR + motion averaging 实现高精度定位

## 相关概念
- [[Relative Pose Regression]]
- [[Visual Place Recognition]]
