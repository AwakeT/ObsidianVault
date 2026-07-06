---
type: concept
aliases: [MoE, Mixture of Experts, 混合专家]
---

# Mixture-of-Experts

## 定义
Mixture-of-Experts (MoE) 是用路由器为不同输入动态选择或加权多个专家子模块的模型结构。

## 数学形式
$$
y = \sum_{i=1}^{N} g_i(x) E_i(x)
$$

其中 $E_i$ 是第 $i$ 个专家，$g_i(x)$ 是路由器给出的专家权重。

## 核心要点
1. 通过条件计算提高模型容量或任务适配能力。
2. 路由器可以是样本级、token 级、层级或任务级。
3. 在持续学习中，专家专精有助于隔离不同任务知识。

## 代表工作
- [[M3E]]: 使用 Macro/Micro 双路由选择 MoE-LoRA 专家。

## 相关概念
- [[Top-K Routing]]
- [[MoE-LoRA]]
- [[Dual-Router Fusion]]
