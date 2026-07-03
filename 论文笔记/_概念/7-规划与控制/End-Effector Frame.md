---
type: concept
aliases: [EEF Frame, 末端执行器坐标系]
---

# End-Effector Frame

## 定义

End-Effector Frame 是以机器人末端执行器为参考的局部坐标系，用来描述夹爪或工具相对自身姿态的位移、旋转和接触动作。

## 核心要点

1. 比关节空间更容易跨 embodiment 对齐。
2. 常用于 manipulation policy 的动作输出。
3. 与 [[Delta Action]] 结合时，动作表示为局部位姿增量和 gripper 状态。

## 代表工作

- [[ABot-M0]]: 将多源机器人数据统一为 EEF delta action。

## 相关概念

- [[Delta Action]]
- [[Rotation Vector]]
