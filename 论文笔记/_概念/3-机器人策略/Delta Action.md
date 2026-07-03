---
type: concept
aliases: [相对动作, Delta Actions]
---

# Delta Action

## 定义

Delta Action 是以当前状态为参考的相对动作表示，通常预测末端执行器位移、旋转增量和夹爪状态变化，而不是绝对关节或绝对位姿。

## 核心要点

1. 相对表示能降低跨机器人平台的动作尺度差异。
2. 在操作任务中常与 [[End-Effector Frame]] 一起使用。
3. 对 VLA 预训练更友好，因为它强调局部可执行变化而非平台特定绝对坐标。

## 代表工作

- [[ABot-M0]]: 将所有动作统一为 EEF delta action。

## 相关概念

- [[End-Effector Frame]]
- [[Rotation Vector]]
- [[Vision-Language-Action]]
