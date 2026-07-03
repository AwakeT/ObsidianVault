---
type: concept
aliases: [线性注意力, Linear Attention]
---

# Linear Attention

## 定义
用核特征映射替代 softmax，把注意力从 $O(N^2)$ 降到**线性复杂度 $O(N)$** 的一类机制：通过结合律先聚合 key-value 外积成一个定长状态矩阵，再与 query 交互，从而可写成**常数显存的循环形式**。

## 数学形式
$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{K}_t)\mathbf{V}_t^\top,\qquad
\mathbf{O}_t = \phi(\mathbf{Q}_t)^\top \mathbf{S}_t
$$

## 核心要点
1. 定长状态 $\mathbf{S}$ 压缩全部历史，推理时计算与显存 $O(1)$，天然适合长序列 / 在线流。
2. 代价是有限记忆容量 → 超训练长度易 **遗忘 / 退化**。
3. 变体：[[Gated Linear Attention]]（门控遗忘）、[[DeltaNet]]（delta rule 修正）、RetNet、Mamba 等。
4. 可统一到 [[Test-Time Training]] 视角：状态更新即在线学习。

## 代表工作
- Linear Transformers / RetNet / GLA / Mamba：语言建模线性序列模型。
- [[CUT3R]] / [[TTT3R]]：把线性注意力式定长状态用于在线 3D 重建。

## 相关概念
- [[Attention Mechanism]]
- [[Gated Linear Attention]]
- [[DeltaNet]]
- [[Test-Time Training]]
