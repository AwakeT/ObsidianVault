---
type: concept
aliases: [Mixture-of-LoRA Experts, MoE LoRA]
---

# MoE-LoRA

## 定义
MoE-LoRA 是将多个 LoRA 适配器作为专家，并通过路由器动态选择或加权这些低秩专家的参数高效 MoE 结构。

## 数学形式
$$
\Delta W(x) = \sum_{i=1}^{N} g_i(x) B_i A_i
$$

其中 $g_i(x)$ 是第 $i$ 个 LoRA 专家的路由权重。

## 核心要点
1. 保留 LoRA 的参数效率，同时引入专家专精能力。
2. 可让不同专家学习不同任务、域或语义模式。
3. 路由策略会直接影响专家利用率和持续学习稳定性。

## 代表工作
- [[M3E]]: 使用 Macro Router 与 Micro Router 的融合权重驱动 MoE-LoRA。

## 相关概念
- [[LoRA]]
- [[Mixture-of-Experts]]
- [[Dual-Router Fusion]]
