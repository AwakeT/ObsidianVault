---
type: concept
aliases: [Navigation LLM, LLM Navigation Agent]
---

# NaviLLM

## 定义
NaviLLM 是一种将大语言模型能力引入视觉语言导航的 agent 框架，用 LLM 参与指令理解、历史建模和动作决策。

## 核心要点
1. 以语言模型作为导航策略的高层推理核心。
2. 通常需要把视觉、历史轨迹、候选动作和指令编码成 LLM 可消费的上下文。
3. 可通过参数高效微调方法适配 VLN 数据集与新环境。

## 代表工作
- [[M3E]]: 以 NaviLLM 作为持续 VLN 的骨干，并在其可训练模块中引入 MoE-LoRA。

## 相关概念
- [[Vision-and-Language Navigation]]
- [[LLM]]
- [[LoRA]]
