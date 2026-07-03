---
type: concept
aliases: [动作规划, 子任务规划]
---

# Action Planning

## 定义

Action Planning 是将高层目标分解为可执行动作或子任务序列的过程，在机器人系统中常由 VLM planner 预测文本子任务，再交给低层 actor 执行。

## 数学形式

$$
a_t=\pi(o_{\le t},a_{<t},g)
$$

## 核心要点

1. 长程任务通常是非 Markov 的，必须依赖历史观测和历史动作。
2. 输出可以是自然语言子任务、导航 waypoint、stop 信号或低层控制目标。
3. 在层级机器人系统中，action planning 的粒度需要适配 actor 的可控能力。

## 代表工作

- [[Vesta]]: 用 memory-conditioned VLM planner 预测真实机器人下一条子任务。

## 相关概念

- [[Subgoal Planning]]
- [[Vision-Language-Action]]
- [[Memory Harness]]
