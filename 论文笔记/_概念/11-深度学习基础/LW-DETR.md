---
type: concept
aliases: [Light-Weight DETR]
---

# LW-DETR

## 定义
Chen et al. (2024) 提出的轻量级实时 [[DETR]]，用 ViT backbone + 窗口注意力替代 YOLO，证明预训练能提升实时 DETR 性能。

## 核心要点
1. 用 [[CAEv2]] ViT backbone（10 层，patch 16），窗口注意力块放在 layer {0,1,3,6,7,9}
2. 解码器各层独立监督，支持推理时层裁剪
3. 是 [[RF-DETR]] 的直接基线与起点

## 代表工作
- [[RF-DETR]]: 在 LW-DETR 基础上换 DINOv2 backbone + 权重共享 NAS

## 相关概念
- [[DETR]]
- [[Windowed Attention]]
- [[Vision Transformer]]
