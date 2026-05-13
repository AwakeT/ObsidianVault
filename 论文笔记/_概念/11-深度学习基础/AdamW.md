---
type: concept
aliases: [AdamW 优化器, Decoupled Weight Decay]
---

# AdamW

## 定义
在 Adam 基础上将权重衰减从梯度更新中解耦的优化器，修正了 Adam 中 L2 正则化与自适应学习率的不一致问题。

## 核心要点
1. Adam + weight decay 不等价于 L2 正则化（Loshchilov & Hutter 2019）
2. AdamW 将 weight decay 直接应用于参数而非梯度
3. 在大模型训练中几乎成为标准优化器

## 代表工作
- [[RAFT-Stereo]]: AdamW + one-cycle 学习率调度
- [[FoundationStereo]]: AdamW lr=1e-4

## 相关概念
- [[Adam]]
