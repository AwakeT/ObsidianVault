---
type: concept
aliases: [Gated Linear Attention, GLA, 门控线性注意力]
---

# Gated Linear Attention

## 定义
线性注意力的门控变体，用数据相关的门控机制控制状态更新，以线性复杂度高效编码长序列，同时保持较强表达力。

## 核心要点
1. 相比 softmax 注意力，复杂度从 $O(n^2)$ 降到 $O(n)$，适合长多模态序列。
2. 在 Qwen3.5 的混合注意力设计中占多数层，间隔层保留 [[Grouped-Query Attention|分组查询 softmax 注意力]]做全精度全局推理。

## 代表工作
- [[Qwen3.5-VL]]: 混合注意力主干

## 相关概念
- [[Grouped-Query Attention]]
- [[Self-Attention]]
- [[Qwen3.5-VL]]
