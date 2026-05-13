---
type: concept
aliases: [实例归一化, IN]
---

# Instance Normalization

## 定义
对每个样本的每个通道独立归一化的归一化方法，不跨 batch 也不跨通道，常用于风格迁移和密集预测任务。

## 核心要点
1. 归一化维度: 单个样本 + 单个通道的空间维度（$H \times W$）
2. 相比 BN 不依赖 batch 统计量，推理时行为一致
3. RAFT-Stereo 的 feature encoder 使用 IN（提升泛化能力）

## 代表工作
- [[RAFT-Stereo]]: Feature Encoder 使用 Instance Normalization

## 相关概念
- [[Batch Normalization]]
