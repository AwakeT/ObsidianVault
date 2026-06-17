---
type: concept
aliases: [Flexi ViT]
---

# FlexiViT

## 定义
Beyer et al. (2023) 提出的 ViT 变体，"一个模型适配所有 patch size"——通过 patch embedding 插值，使单个 ViT 能在不同 patch size 下工作。

## 核心要点
1. 对 patch embedding 权重做插值，支持训练/推理时切换 patch size
2. patch size 越小 token 越多、越准但越慢
3. [[RF-DETR]] 借用其 patch 插值机制作为 NAS 的一个可调旋钮，并能泛化到训练时未见的 patch size（如 27、18）

## 代表工作
- [[RF-DETR]]: patch size 插值与对未见 patch 的泛化

## 相关概念
- [[Vision Transformer]]
- [[Neural Architecture Search]]
