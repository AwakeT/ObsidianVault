---
type: concept
aliases: [Alternating Attention, 交替注意力]
---

# Alternating Attention

## 定义
VGGT 提出的注意力机制，在帧内自注意力和全局自注意力之间交替，平衡局部特征归一化和跨帧信息融合。

## 核心要点
1. 帧内注意力：各帧 token 独立 attend
2. 全局注意力：所有帧 token 联合 attend
3. 不使用 cross-attention，仅使用 self-attention
4. 消融实验证明优于纯 cross-attention 和纯全局 self-attention

## 代表工作
- [[VGGT]]: 核心注意力机制

## 相关概念
- [[Self-Attention]]
- [[Cross-Attention]]
- [[Transformer]]
