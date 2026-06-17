---
type: concept
aliases: [CAE v2, Context Autoencoder v2]
---

# CAEv2

## 定义
Zhang et al. (2022) 提出的 Context Autoencoder v2，以 CLIP 为目标的上下文自编码器自监督预训练方法，[[LW-DETR]] 用其作为 backbone。

## 核心要点
1. 自监督预训练，编码器 10 层、patch size 16，省略 class token
2. 是 [[LW-DETR]] 的原始 backbone
3. [[RF-DETR]] 用 [[DINOv2]] 替换 CAEv2，提升约 2.4 AP

## 代表工作
- [[LW-DETR]]: 使用 CAEv2 backbone
- [[RF-DETR]]: 用 DINOv2 替换之

## 相关概念
- [[DINOv2]]
- [[CLIP]]
