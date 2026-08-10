---
type: concept
aliases: [选择性遗忘, Learned Forgetting, Intentional Forgetting]
---

# Selective Forgetting

## 定义
对外部记忆条目进行有目的的保留、压缩、降级或删除：系统依据时间、任务价值、用户偏好、置信度或新颖性决定哪些细节继续可访问。

## 数学形式

$$
\tau_n \gets \tau_n + \alpha_n \Delta t_\ell
$$

其中 $\tau_n$ 是记忆节点到期时间，$\alpha_n$ 是相关性因子，$\Delta t_\ell$ 是层级相关的基础寿命。

## 核心要点
1. 与灾难性遗忘不同：它治理外部显式记忆，不是模型参数无意丢失旧知识
2. 目标是在问答质量、存储规模、查询延迟和治理风险之间取得平衡
3. 删除可以分为热存→摘要→占位符→冷存→物理删除等可审计层级
4. 用户反馈可把固定 TTL 升级为个性化、可检查的保留策略

## 代表工作
- [[H2-EMV]]: 到期衰减 + LLM 相关性估计 + 用户反馈学习自然语言规则

## 相关概念
- [[Episodic Memory]]
- [[Relevance Estimation]]
- [[Hierarchical Memory]]
- [[Continual Learning]]
