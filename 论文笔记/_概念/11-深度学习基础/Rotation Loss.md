---
type: concept
aliases: [Rotation Loss, 旋转损失]
---

# Rotation Loss

## 定义
衡量预测旋转与真值旋转之间差异的损失函数，常用测地距离。

## 数学形式

$$$\ell_R = \arccos\frac{\text{tr}(\hat{R}^{-1}R) - 1}{2}$$$

## 核心要点
1. 测地距离：旋转矩阵间的角度差
2. 以弧度为单位，与平移方向损失天然平衡

## 代表工作
- [[Reloc3r]]: 旋转损失 + 平移方向损失

## 相关概念
- [[Translation Loss]]
- [[9D Rotation Representation]]
