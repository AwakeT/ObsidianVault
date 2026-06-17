---
type: concept
aliases: [LayerNorm, 层归一化]
---

# Layer Normalization

## 定义
对单个样本的特征维度做归一化（而非跨 batch），不依赖 batch 统计量，对 batch size 鲁棒。

## 核心要点
1. 与 [[Batch Normalization]] 不同，归一化沿特征维而非 batch 维
2. 对小 batch / 梯度累积训练友好
3. [[RF-DETR]] 在多尺度 projector 用 LayerNorm 替 BatchNorm，以支持消费级 GPU 上的梯度累积训练（虽轻微降精度但必要）

## 代表工作
- [[RF-DETR]]: projector 用 LayerNorm 支持梯度累积

## 相关概念
- [[Batch Normalization]]
- [[Transformer]]
- [[梯度累积]]
