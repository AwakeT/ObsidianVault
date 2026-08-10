---
type: concept
aliases: [相关性估计, Memory Relevance Estimation, Importance Estimation]
---

# Relevance Estimation

## 定义
根据记忆内容、上下文、任务和用户规则估计某条记忆对未来使用的价值，用于排序检索、延长寿命或触发遗忘。

## 数学形式

$$
\alpha_n = f(n,\operatorname{parent}(n),R,u)
$$

其中 $n$ 是记忆节点，$R$ 是保留规则集，$u$ 是用户/任务上下文，$\alpha_n$ 是保留相关性。

## 核心要点
1. 可以由固定打分、学习模型或 LLM 规则推理实现
2. 相关性不是全局常数，应显式考虑用户、任务、时间和作用域
3. 错误高估会浪费存储，错误低估会造成不可恢复的信息丢失
4. 实际系统需要 provenance、版本、审计和回滚机制

## 代表工作
- [[H2-EMV]]: LLM 读取 node + parent + natural-language rules，输出整数延寿因子或 `inf`

## 相关概念
- [[Selective Forgetting]]
- [[Episodic Memory]]
- [[LLM]]
