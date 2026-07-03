---
type: concept
aliases: [CoT, 思维链]
---

# Chain-of-Thought

## 定义

Chain-of-Thought 是让语言模型在给出最终答案前显式生成中间推理步骤的 prompting 或训练范式。

## 数学形式

$$
p(y,r\mid x)=p(r\mid x)\,p(y\mid x,r)
$$

## 核心要点

1. 中间推理 $r$ 可以帮助模型分解复杂问题。
2. 在具身任务中，CoT 常被结构化为 observation、progress、reasoning、action 等字段。
3. 部署时不一定需要把完整 CoT 传给下游执行器。

## 代表工作

- [[Vesta]]: 在每个子任务前生成 Observation、Progress、Reasoning、Action 四段推理。

## 相关概念

- [[LLM]]
- [[Action Planning]]
- [[Embodied Reasoning]]
