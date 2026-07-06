---
type: concept
aliases: [双路由融合, Macro-Micro Fusion]
---

# Dual-Router Fusion

## 定义
Dual-Router Fusion 是把两个不同粒度或不同信息源的路由权重融合为统一专家权重的机制。

## 数学形式
$$
w = \beta w_{\mathrm{macro}} + (1-\beta) w_{\mathrm{micro}}
$$

其中 $\beta$ 控制宏观路由与微观路由的相对权重。

## 核心要点
1. 可同时利用全局上下文和局部 token 语义。
2. 融合权重需要避免某一路由器长期主导，导致专家退化。
3. 在导航中适合把拓扑级规划与语言 grounding 合并。

## 代表工作
- [[M3E]]: 用固定系数融合 Macro Router 与 Micro Router 的专家权重。

## 相关概念
- [[Macro Router]]
- [[Micro Router]]
- [[MoE-LoRA]]
