---
type: concept
aliases: []
---

# LVIS

## 定义
Large Vocabulary Instance Segmentation 数据集，含超过 1000 类物体、呈长尾（long-tail）分布，用于评测大词汇量、长尾目标检测与分割能力。

## 核心要点
1. **长尾分布**: 大量罕见类别，考验模型对稀有物体的检测能力。
2. **大词汇量**: 类别数远超 COCO，适合评测开集/泛化检测。
3. **评测**: 常用 F1@IoU（0.5 / 0.95 / mean）或 AP。

## 代表工作
- LVIS (Gupta et al., 2019): 长尾实例分割 benchmark。
- [[LocateAnything]]: 在 LVIS 上 mean F1 +3.8% 优于 Rex-Omni。

## 相关概念
- [[Object Detection]]
- [[COCO]]
