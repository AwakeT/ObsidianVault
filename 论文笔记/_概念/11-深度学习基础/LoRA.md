---
type: concept
aliases: [Low-Rank Adaptation, 低秩适配]
---

# LoRA

## 定义
LoRA 是一种参数高效微调方法，通过在冻结权重旁加入低秩增量矩阵来适配下游任务。

## 数学形式
$$
W' = W + \Delta W,\quad \Delta W = BA
$$

其中 $A$ 和 $B$ 是低秩可训练矩阵，原始权重 $W$ 通常保持冻结。

## 核心要点
1. 显著减少可训练参数量和存储成本。
2. 适合大模型在多任务或多域场景中的快速适配。
3. 可与 MoE 结合，让不同 LoRA 专家学习不同能力。

## 代表工作
- [[M3E]]: 在 NaviLLM 的可训练部分采用 MoE-LoRA 专家。

## 相关概念
- [[MoE-LoRA]]
- [[LLM]]
- [[Mixture-of-Experts]]
