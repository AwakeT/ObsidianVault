---
type: concept
aliases: [EMV, 情景记忆言语化, 经历问答]
---

# Episodic Memory Verbalization

## 定义
让机器人或智能体把过去亲历的感知、动作、任务和交互检索出来，并以自然语言回答“何时、何地、做了什么、为什么失败”等问题的任务。

## 数学形式

$$
\hat a = \operatorname{Verbalize}(q,\operatorname{Retrieve}(M,q))
$$

其中 $q$ 是关于过去的问题，$M$ 是情景记忆，$\hat a$ 是基于检索证据生成的答案。

## 核心要点
1. 同时依赖记忆构建、时间/空间检索、证据 grounding 和语言生成
2. 长历史中需要层级索引或 agentic retrieval，避免把全部经历塞入上下文
3. “信息已忘”和“信息仍在但没检索到”必须区分，否则反馈会错误更新保留策略

## 代表工作
- [[H2-EMV]]: 在线 History Tree + 选择性遗忘 + 用户反馈学习
- [[RAVEN]]: 视觉-空间-时间外部记忆上的 agentic 机器人问答

## 相关概念
- [[Episodic Memory]]
- [[Agentic Memory]]
- [[History Tree]]
