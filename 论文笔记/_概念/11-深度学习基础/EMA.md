---
type: concept
aliases: [Exponential Moving Average, 指数移动平均]
---

# EMA

## 定义
指数移动平均：对模型权重做滑动平均得到一份平滑的"影子权重"用于推理，常提升泛化与训练稳定性。

## 核心要点
1. $\theta_{ema} \leftarrow \alpha\,\theta_{ema} + (1-\alpha)\,\theta$
2. [[RF-DETR]] 用 EMA 调度替代 cosine 学习率调度（cosine 假设固定优化 horizon，不适合多样数据集），并省略 warmup
3. 与 [[DINOv3]] 类似采用 EMA 调度

## 代表工作
- [[RF-DETR]]: scheduler-free 训练的关键

## 相关概念
- [[Adam]]
- [[Knowledge Distillation]]
