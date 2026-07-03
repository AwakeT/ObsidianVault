---
type: concept
aliases: [动作流形假设]
---

# Action Manifold Hypothesis

## 定义

Action Manifold Hypothesis 指有效机器人动作序列并非任意分布在高维动作空间，而是受物理规律、任务约束和 embodiment 结构限制，位于低维平滑流形上。

## 核心要点

1. 噪声和速度目标通常位于 off-manifold 区域，直接回归会增加学习负担。
2. clean action 本身更贴近可执行结构，因此更适合作为高维控制的预测目标。
3. 该假设解释了为什么长 action chunk 下 [[Action Manifold Learning]] 比 noise prediction 更稳定。

## 代表工作

- [[ABot-M0]]: 将该假设用于 VLA action expert 的目标设计。

## 相关概念

- [[Action Manifold Learning]]
- [[Flow Matching]]
- [[Diffusion Policy]]
