---
type: concept
aliases: [平均池化, AvgPool]
---

# Average Pooling

## 定义
对输入特征图的局部区域取平均值的下采样操作。

## 核心要点
1. 相比最大池化保留更多上下文信息
2. RAFT-Stereo 中用于构建 correlation pyramid

## 代表工作
- [[RAFT-Stereo]]: Correlation Pyramid 各层通过 average pooling 构建

## 相关概念
- [[Correlation Volume]]
