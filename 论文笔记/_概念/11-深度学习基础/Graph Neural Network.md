---
type: concept
aliases: [GNN, 图神经网络]
---

# Graph Neural Network

## 定义
Graph Neural Network (GNN) 是在图结构数据上进行消息传递和节点/图表示学习的神经网络。

## 数学形式
$$
h_v^{(k+1)} = \phi\left(h_v^{(k)}, \operatorname{AGG}_{u \in \mathcal{N}(v)} \psi(h_v^{(k)}, h_u^{(k)}, e_{uv})\right)
$$

其中 $h_v$ 是节点表示，$\mathcal{N}(v)$ 是邻居集合。

## 核心要点
1. 适合处理拓扑、关系和空间连接信息。
2. 通过邻居聚合把局部结构传播到节点表示中。
3. 在导航任务中可用于建模环境图、路径图或认知图。

## 代表工作
- [[M3E]]: Macro Router 用 GNN 在认知图上聚合 visited/frontier nodes。

## 相关概念
- [[Cognitive Map]]
- [[Macro Router]]
