---
type: concept
aliases: [RoMa, Robust Dense Feature Matching]
---

# RoMa

## 定义
鲁棒稠密特征匹配方法，在两视图匹配基准上取得 SOTA。

## 核心要点
1. 稠密匹配后通过几何估计恢复位姿
2. 在 ScanNet-1500 上 AUC@5=28.9

## 代表工作
- [[VGGT]]: 在匹配基准上超越 RoMa
- [[Reloc3r]]: 精度相当但速度快 7-20 倍

## 相关概念
- [[Efficient LoFTR]]
