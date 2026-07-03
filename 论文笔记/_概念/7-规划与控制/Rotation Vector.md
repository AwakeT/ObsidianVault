---
type: concept
aliases: [旋转向量, Axis-Angle]
---

# Rotation Vector

## 定义

Rotation Vector 是轴角旋转表示，将三维旋转写成 $r=\theta k$，其中 $\theta$ 是旋转角，$k$ 是单位旋转轴。

## 数学形式

$$
r=\theta k,\qquad r\in\mathbb{R}^3,\quad \theta\in[0,\pi],\quad \|k\|=1
$$

## 核心要点

1. 相比 Euler angles 可避免部分奇异性和角度顺序问题。
2. 相比 rotation matrix / quaternion 更紧凑，适合动作向量输出。
3. 在精细旋转操作中有助于提高动作预测稳定性。

## 代表工作

- [[ABot-M0]]: 将多源 orientation 表示统一转换为 rotation vector。

## 相关概念

- [[Delta Action]]
- [[End-Effector Frame]]
- [[6D Rotation]]
