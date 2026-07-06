---
type: concept
aliases: [微观路由器, Token-Level Router]
---

# Micro Router

## 定义
Micro Router 是在 token 或局部语义层面选择专家的路由模块，关注指令词、视觉语义和细粒度 grounding。

## 核心要点
1. 通常从 LLM hidden state 生成 token-wise expert weights。
2. 适合区分动作词、地标词、物体词和空间关系词所需的不同专家。
3. 与 Macro Router 融合后，可同时利用局部语义和全局拓扑信息。

## 代表工作
- [[M3E]]: 使用 MLP/Softmax 为每个 token 生成 micro routing weights。

## 相关概念
- [[LLM]]
- [[Macro Router]]
- [[Dual-Router Fusion]]
- [[MoE-LoRA]]
