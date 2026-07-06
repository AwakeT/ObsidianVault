---
type: concept
aliases: [宏观路由器, Scene-Level Router]
---

# Macro Router

## 定义
Macro Router 是在场景或拓扑层面选择专家的路由模块，关注全局空间结构、已探索区域和导航阶段。

## 核心要点
1. 输入通常来自认知图、拓扑邻接关系和指令表示。
2. 适合捕捉房间布局、前沿探索和长程规划相关的专家偏好。
3. 在双路由 MoE 中与 token 级 Micro Router 互补。

## 代表工作
- [[M3E]]: 使用 GNN 聚合 cognitive map，再通过指令注意力得到 scene-level expert weights。

## 相关概念
- [[Cognitive Map]]
- [[Graph Neural Network]]
- [[Micro Router]]
- [[Dual-Router Fusion]]
