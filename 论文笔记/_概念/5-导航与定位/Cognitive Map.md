---
type: concept
aliases: [认知图, Cognitive Graph]
---

# Cognitive Map

## 定义
Cognitive Map 是智能体在导航过程中维护的环境拓扑或空间记忆，用于记录已访问节点、候选前沿节点及其连接关系。

## 核心要点
1. 连接在线观测与长期空间记忆，使 agent 能在部分可观测环境中推理。
2. 常以图结构表达房间、视点、前沿位置和可通行边。
3. 在持续 VLN 中可作为宏观路由器判断场景结构和任务阶段的依据。

## 代表工作
- [[M3E]]: Macro Router 从 cognitive map 中提取 visited/frontier nodes 并构造稀疏邻接图。

## 相关概念
- [[Graph Neural Network]]
- [[Macro Router]]
- [[Vision-and-Language Navigation]]
