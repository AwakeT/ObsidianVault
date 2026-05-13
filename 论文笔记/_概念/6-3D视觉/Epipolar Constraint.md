---
type: concept
aliases: [Epipolar Constraint, 对极约束, Epipolar Geometry]
---

# Epipolar Constraint

## 定义
两视图几何中的基本约束：对应点必须满足 $x'^T F x = 0$，其中 $F$ 为基础矩阵。运动物体违反此约束。

## 数学形式

$$$x'^T F x = 0$$$

## 核心要点
1. 静态场景中两视图匹配点的基本约束
2. 动态物体违反对极约束，给 SfM 和 SLAM 带来困难
3. MonST3R 通过 pointmap 表示绕过了对极约束的假设

## 代表工作
- [[MonST3R]]: 处理违反对极约束的动态场景

## 相关概念
- [[Structure from Motion]]
- [[Bundle Adjustment]]
