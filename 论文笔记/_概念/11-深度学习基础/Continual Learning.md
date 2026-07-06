---
type: concept
aliases: [CL, 持续学习, 终身学习]
---

# Continual Learning

## 定义
Continual Learning 是模型按任务或数据流顺序持续学习新知识，同时尽量保持旧知识不遗忘的学习范式。

## 核心要点
1. 核心矛盾是 stability-plasticity trade-off。
2. 常见方法包括正则化、回放、参数隔离、动态架构和专家化模型。
3. 评估常使用 backward transfer、forward transfer 和平均任务性能。

## 代表工作
- [[M3E]]: 研究 replay-free continual VLN，并用动态 MoE 动量更新缓解遗忘。

## 相关概念
- [[Dynamic MoE Momentum Update]]
- [[Mixture-of-Experts]]
- [[Vision-and-Language Navigation]]
