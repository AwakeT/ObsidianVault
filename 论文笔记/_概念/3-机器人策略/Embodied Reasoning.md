---
type: concept
aliases: [具身推理, Embodied Reasoning]
---

# Embodied Reasoning

## 定义

Embodied Reasoning 指模型在具身场景中结合视觉、空间关系、可供性、任务进度和动作后果进行推理的能力。

## 数学形式

$$
y=f(o_t,g,h_t)
$$

## 核心要点

1. 不只回答“图中有什么”，还要判断“在哪里、能否操作、下一步如何影响任务”。
2. 常见任务包括 affordance prediction、placement prediction、egocentric VQA、task progress estimation。
3. 是高层机器人 planner 和低层 VLA actor 之间的重要语义桥梁。

## 代表工作

- [[Vesta]]: 把 embodied reasoning 作为 generalist planner 的四类核心能力之一。

## 相关概念

- [[Vision-Language-Action]]
- [[VLM]]
- [[Action Planning]]
