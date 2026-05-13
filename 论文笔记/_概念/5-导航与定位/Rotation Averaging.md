---
type: concept
aliases: [Rotation Averaging, 旋转平均]
---

# Rotation Averaging

## 定义
从多个相对旋转估计中恢复全局一致绝对旋转的问题，通常使用中位数或 L1 优化。

## 核心要点
1. Reloc3r 中取四元数表示的中位数
2. 比均值更鲁棒

## 代表工作
- [[Reloc3r]]: 用于从多个相对旋转恢复查询图像的绝对旋转

## 相关概念
- [[Motion Averaging]]
