---
type: concept
aliases: [Multi-Task Learning, MTL, 多任务学习]
---

# Multi-Task Learning

## 定义
同时训练一个模型完成多个相关任务，利用任务间的共享表示和互补信息提升各任务性能。

## 核心要点
1. 共享 backbone + 任务专用 head
2. 损失函数加权是关键设计决策
3. VGGT 证明了多任务训练带来的互补增益

## 代表工作
- [[VGGT]]: 四任务联合训练（相机、深度、点云、追踪）互补增益

## 相关概念
- [[Transformer]]
