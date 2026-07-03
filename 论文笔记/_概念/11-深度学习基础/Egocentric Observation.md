---
type: concept
aliases: [第一人称观测, 自我中心观测, Egocentric View]
---

# Egocentric Observation

## 定义

Egocentric Observation 是从机器人或人自身视角采集的视觉观测，反映当前执行者能看到的局部场景。

## 数学形式

$$
o_t = \operatorname{Camera}(s_t)
$$

## 核心要点

1. 与全局第三人称视角不同，egocentric 观测通常局部、遮挡多、依赖历史。
2. 在导航和机器人操作中，它是 planner 与 actor 的主要视觉输入。
3. 长程任务需要把当前 egocentric view 与历史记忆结合。

## 代表工作

- [[Vesta]]: 使用 egocentric video 进行 memory-conditioned planning 与真实机器人评估。

## 相关概念

- [[Vision-Language Navigation]]
- [[Memory Harness]]
- [[Visual Perception]]
