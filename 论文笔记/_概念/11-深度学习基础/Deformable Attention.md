---
type: concept
aliases: [Deformable Cross-Attention, 可变形注意力, Deformable DETR]
---

# Deformable Attention

## 定义
Zhu et al. (2020, Deformable DETR) 提出的注意力机制：每个 query 只关注参考点周围少量可学习采样点，而非全部空间位置，大幅降低计算量并加速 DETR 收敛。

## 核心要点
1. 稀疏采样 K 个点，复杂度从 O(N²) 降到 O(NK)
2. 加速 DETR 训练收敛、支持多尺度特征
3. [[RF-DETR]] 解码器用 deformable cross-attention，并对 projector 输出双线性插值保持空间一致

## 代表工作
- [[RF-DETR]]: 解码器交叉注意力
- [[DETR]]: Deformable DETR 是其重要改进

## 相关概念
- [[Self-Attention]]
- [[Cross-Attention]]
- [[DETR]]
