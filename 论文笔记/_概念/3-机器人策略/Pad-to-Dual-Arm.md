---
type: concept
aliases: [pad-to-dual-arm, 双臂零填充]
---

# Pad-to-Dual-Arm

## 定义

Pad-to-Dual-Arm 是一种统一单臂和双臂机器人数据的动作格式策略：单臂轨迹保留执行臂动作，未使用手臂的动作维度用零填充。

## 核心要点

1. 让一个策略网络始终输出双臂动作空间。
2. 单臂数据可被视为右臂执行，左臂维度置零。
3. 允许模型在指令条件下隐式学习何时单臂执行、何时双臂协作。

## 代表工作

- [[ABot-M0]]: 用 Pad-to-Dual-Arm 将单臂和双臂数据统一到 14 维动作输出。

## 相关概念

- [[Delta Action]]
- [[End-Effector Frame]]
- [[Vision-Language-Action]]
