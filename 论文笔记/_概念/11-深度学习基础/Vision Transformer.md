---
type: concept
aliases: [Vision Transformer, ViT]
---

# Vision Transformer

## 定义
将 Transformer 架构应用于图像处理的方法，将图像分割为 patch 序列后输入标准 Transformer。

## 核心要点
1. 图像被分割为固定大小的 patch（如 14×14 或 16×16）
2. ViT-B/L/H 分别对应不同规模
3. 是 DINOv2、DUSt3R、VGGT 等方法的基础架构

## 代表工作
- [[VGGT]]: 基于 ViT-L 架构的 1.2B 参数模型
- [[Reloc3r]]: ViT-Large encoder + ViT-Base decoder

## 相关概念
- [[Transformer]]
- [[DINOv2]]
- [[DPT]]
