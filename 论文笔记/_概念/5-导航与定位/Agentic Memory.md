---
type: concept
aliases: [智能体记忆, agentic memory]
---

# Agentic Memory

## 定义
由 LLM/VLM agent 以迭代循环（常形式化为有限状态机）方式主动检索、评估外部记忆库并决定是否继续检索或生成答案的记忆使用范式。

## 核心要点
1. agent 维护工作记忆上下文，判断证据是否充分
2. 不足则调用检索工具追加记忆，形成闭环推理
3. 缩短喂给模型的注意力窗口，避免全库穷举

## 代表工作
- [[RAVEN]]: VLM agent 迭代调用 text/time/position/image 四种检索工具

## 相关概念
- [[Retrieval-Augmented Generation]]
- [[Visuo-Spatio-Temporal Memory]]
- [[ReMEmbR]]
