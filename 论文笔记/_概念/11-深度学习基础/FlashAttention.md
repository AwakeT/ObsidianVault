---
type: concept
aliases: [高效注意力]
---

# FlashAttention

## 定义
通过 IO-aware 的分块计算策略实现的高效注意力实现，在不改变数学等价性的前提下显著降低 GPU 内存占用和提升速度。

## 核心要点
1. 通过 tiling 避免在 HBM 中存储完整注意力矩阵
2. 利用 online softmax 技巧实现精确等价的分块计算
3. FlashAttention-2 进一步优化了并行度和工作分配

## 代表工作
- [[FoundationStereo]]: Disparity Transformer 使用 FlashAttention

## 相关概念
- [[Transformer]]
- [[Self-Attention]]
