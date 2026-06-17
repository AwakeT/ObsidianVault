---
type: concept
aliases: [Qwen3.5-VL, Qwen3.5, Qwen3.5-4B]
---

# Qwen3.5-VL

## 定义
Qwen 系列的原生多模态视觉语言模型，采用早期视觉语言融合与混合注意力设计；Qwen-VLA 以其 4B 版本为主干。

## 核心要点
1. **早期融合**：ViT（含空间合并）产生的视觉 token 直接插入文本 token 流，单一 Transformer 统一处理图像/视频/语言。
2. **混合注意力**：多数层用 [[Gated Linear Attention|门控线性注意力]]，间隔层用 [[Grouped-Query Attention|分组查询 softmax 注意力]]，兼顾长序列效率与全精度全局推理。
3. 提供细粒度视觉感知、指代 grounding、多语言指令跟随、结构化推理能力，是具身任务的关键基础。

## 代表工作
- Qwen Team 2026: 原始提出
- [[Qwen-VLA]]: 以 Qwen3.5-4B 为主干

## 相关概念
- [[Gated Linear Attention]]
- [[Grouped-Query Attention]]
- [[Vision-Language-Action]]
- [[RoPE]]
