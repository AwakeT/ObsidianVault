---
type: concept
aliases: [Mask DINO]
---

# MaskDINO

## 定义
Li et al. (2023) 提出的统一 Transformer 框架，将目标检测与实例/全景分割统一在 DINO 架构下，用 query embedding 与像素嵌入图点积生成掩码。

## 核心要点
1. 检测与分割共享 query，融合多尺度 backbone 特征
2. 精度高但延迟大（R50 约 242ms）
3. [[RF-DETR]]-Seg 受其启发但不融合多尺度特征以省延迟

## 代表工作
- [[RF-DETR]]: RF-DETR-Seg (medium) 逼近 MaskDINO 而延迟仅其零头

## 相关概念
- [[DETR]]
- [[Open-Vocabulary Segmentation]]
