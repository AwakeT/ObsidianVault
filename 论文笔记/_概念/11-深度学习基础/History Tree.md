---
type: concept
aliases: [历史树, Hierarchical History Tree]
---

# History Tree

## 定义
把时间序列经历按 scene、event、goal 和更高层活动摘要组织成树的层级外部记忆结构；高层节点用于粗粒度检索，子节点用于回溯证据。

## 数学形式

$$
H=(V,E),\qquad
n=(S_n,\operatorname{children}(n),[t_s,t_e],\tau_n)
$$

其中节点保存摘要 $S_n$、子节点、时间范围和可选到期时间 $\tau_n$。

## 核心要点
1. 查询可先搜索高层摘要，再逐层展开，降低长历史 token 成本
2. 在线更新只重算受新节点影响的时间尾部，避免每次全树重建
3. 摘要必须保留到低层证据的可追溯关系
4. 生命周期治理应保证父节点不会早于仍存活的子节点过期

## 代表工作
- [[H2-EMV]]: 时间感知的在线 History Tree，并使用相关性规则控制摘要与遗忘

## 相关概念
- [[Hierarchical Memory]]
- [[Episodic Memory]]
- [[Selective Forgetting]]
- [[Scene Graph]]
