---
type: concept
aliases: [Top-K 路由, Sparse Routing]
---

# Top-K Routing

## 定义
Top-K Routing 是在 MoE 中只选择权重最高的 K 个专家参与计算或更新的稀疏路由策略。

## 数学形式
$$
\mathcal{S}_K(x) = \operatorname{TopK}(g(x), K)
$$

其中 $g(x)$ 是路由器输出的专家分数。

## 核心要点
1. 降低计算开销，并鼓励专家形成差异化分工。
2. 需要处理负载均衡、专家塌缩和路由不稳定问题。
3. 在持续学习中可用于识别当前任务最相关的专家集合。

## 代表工作
- [[M3E]]: 在动态 MoE 动量更新中选择贡献度 Top-K 的重要专家。

## 相关概念
- [[Mixture-of-Experts]]
- [[Expert Contribution]]
- [[Dynamic MoE Momentum Update]]
