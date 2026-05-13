---
type: concept
aliases: [前沿探索, frontier-based exploration]
---

# Frontier Exploration

## 定义
基于自由空间与未知空间边界的探索策略，agent 优先前往 frontier 以最大化信息增益。

## 数学形式
$$\mathcal{F}^t = \{v \in \mathcal{V}_{\text{free}}^t \mid \exists v' \in \mathcal{N}(v), v' \in \mathcal{V}_{\text{unk}}^t\}$$

## 核心要点
1. Frontier 是自由空间与未知空间的边界
2. 选择策略可结合语义、距离、历史等多因素
3. 广泛用于 zero-shot 导航和 ObjectNav

## 代表工作
- [[MetaNav]]: 统一效用函数选择 frontier
- [[OVAL]]: 概率地图多因素 frontier 评分
- [[VLFM]]: 视觉语言 frontier maps

## 相关概念
- [[Semantic Map]]
- [[TSDF]]
- [[Object Navigation]]
