---
type: concept
aliases: [动态 MoE 动量更新, Dynamic Momentum Update]
---

# Dynamic MoE Momentum Update

## 定义
Dynamic MoE Momentum Update 是根据专家在当前任务中的贡献度，为不同专家分配不同更新强度的持续学习参数更新策略。

## 数学形式
$$
\theta_i \leftarrow \alpha_i \theta_i^{\mathrm{old}} + (1-\alpha_i)\theta_i^{\mathrm{new}}
$$

其中 $\alpha_i$ 随专家重要性变化；重要专家更偏向当前任务更新，非重要专家更保守。

## 核心要点
1. 利用路由统计判断哪些专家主要承担当前任务。
2. 对关键专家增强可塑性，对非关键专家增强稳定性。
3. 适合 replay-free continual learning，因为不依赖旧样本缓存。

## 代表工作
- [[M3E]]: 根据 expert contribution 选择 Top-K 重要专家，并采用差异化动量更新 MoE 参数。

## 相关概念
- [[Continual Learning]]
- [[Expert Contribution]]
- [[Top-K Routing]]
- [[MoE-LoRA]]
