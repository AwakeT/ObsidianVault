---
type: concept
aliases: [锚点视图, anchor view node]
---

# Anchor View

## 定义
在视觉记忆图中表示"物体丰富的已探索观测区域"的视图节点，存储原始图像证据用于目标验证与多视图 Stop 决策。

## 数学形式
$$v_t^A = (I_t^{rgb},\ p_t,\ O_t^{vis},\ r_t^{room})$$

## 核心要点
1. 与[[Frontier View|前沿视图]]互补，共同构成视图节点集 $\mathcal{V}_t = \mathcal{V}_{A,t} \cup \mathcal{V}_{F,t}$
2. 存储原始 RGB、位姿、可见物体标签 $O_t^{vis}$、房间标签 $r_t^{room}$
3. 在粗阶段通过房间感知分桶被压缩为紧凑候选集 $C_t^A$
4. Stop 验证基于锚点视图的多视图图像证据，避免过早停止与同类错误实例

## 代表工作
- [[EvoMemNav]]: 在 VSMGraph 中区分锚点视图（验证）与前沿视图（探索）

## 相关概念
- [[Visual-Semantic Memory Graph]]
- [[Frontier View]]
- [[Object Navigation]]
