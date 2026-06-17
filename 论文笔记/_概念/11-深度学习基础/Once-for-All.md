---
type: concept
aliases: [OFA, Once for All]
---

# Once-for-All

## 定义
Cai et al. (2019) 提出的权重共享 [[Neural Architecture Search|NAS]] 方法：训练一个超网络（supernet）后，可针对不同硬件直接抽取专用子网络部署，无需重训。

## 核心要点
1. 通过 progressive shrinking 同时优化数千子网络，共享权重
2. 解耦训练与搜索，换硬件无需重新训练
3. 是 [[RF-DETR]] 端到端权重共享 NAS 的核心思想来源

## 代表工作
- [[RF-DETR]]: 受 OFA 启发将其用于检测，但不分阶段训练、不用 per-stage 调度器

## 相关概念
- [[Neural Architecture Search]]
