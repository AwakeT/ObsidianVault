---
type: concept
aliases: [DFINE]
---

# D-FINE

## 定义
Peng et al. (2024) 提出的实时 [[DETR]]，将边框回归任务重新定义为细粒度分布精化（Fine-grained Distribution Refinement），构建于 [[RT-DETR]] 之上。

## 核心要点
1. 把回归当作分布预测并逐层精化，提升定位精度
2. COCO 上 SOTA 实时检测器之一，但被指可能过拟合 COCO 超参
3. 在 [[RF100-VL]] 上 AP50 反被 [[RT-DETR]] 超过，暗示其超参对 COCO 过拟合

## 代表工作
- [[RF-DETR]]: 主要对比对象，RF-DETR (nano) 超 D-FINE (nano) 5.3 AP

## 相关概念
- [[DETR]]
- [[RT-DETR]]
