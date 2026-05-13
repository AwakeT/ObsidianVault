---
type: concept
aliases: [Translation Loss, 平移损失]
---

# Translation Loss

## 定义
衡量预测平移方向与真值平移方向之间夹角的损失函数。

## 数学形式

$$$\ell_t = \arccos\frac{\hat{t} \cdot t}{\|\hat{t}\| \|t\|}$$$

## 核心要点
1. 仅约束方向，不约束度量尺度
2. 与旋转损失单位一致（弧度），无需权重调优

## 代表工作
- [[Reloc3r]]: 仅学习平移方向，度量尺度由 motion averaging 恢复

## 相关概念
- [[Rotation Loss]]
- [[Motion Averaging]]
