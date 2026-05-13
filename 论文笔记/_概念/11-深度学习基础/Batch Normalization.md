---
type: concept
aliases: [批归一化, BN, BatchNorm]
---

# Batch Normalization

## 定义
对 mini-batch 内每个通道的激活值进行归一化的技术，加速训练收敛并起正则化作用。

## 数学形式
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$$

## 核心要点
1. 训练时用 batch 统计量，推理时用 running average
2. batch size 太小时统计量不准确
3. RAFT-Stereo 的 Context Encoder 使用 BN

## 代表工作
- [[RAFT-Stereo]]: Context Encoder 使用 BN

## 相关概念
- [[Instance Normalization]]
