---
type: concept
aliases: [Grouped-Query Attention, GQA, 分组查询注意力]
---

# Grouped-Query Attention

## 定义
介于 multi-head 与 multi-query 注意力之间的折中：多个查询头共享同一组 key/value 头，减少 KV cache 与内存带宽，同时保持接近 MHA 的质量。

## 核心要点
1. 显著降低推理时 KV cache 开销，常用于大语言/多模态模型。
2. 在 Qwen3.5 混合注意力中作为间隔层，提供全精度全局推理。

## 代表工作
- Ainslie et al. 2023: 原始提出
- [[Qwen3.5-VL]]: 混合注意力主干

## 相关概念
- [[Gated Linear Attention]]
- [[Self-Attention]]
- [[Qwen3.5-VL]]
