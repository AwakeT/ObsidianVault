---
type: concept
aliases: [Depth Anything V2, DAV2]
---

# DepthAnythingV2

## 定义
Yang et al. 2024 提出的单目深度估计基础模型，基于 DPT (Vision Transformer) 架构，在大规模标注+伪标注数据上训练，实现强零样本泛化。

## 核心要点
1. 提供 ViT-S/B/L 多种尺寸
2. 在单目深度估计任务上实现互联网级泛化
3. FoundationStereo 通过 STA 将其先验注入立体匹配

## 代表工作
- [[FoundationStereo]]: 通过 Side-Tuning Adapter 适配到立体匹配

## 相关概念
- [[Side-Tuning]]
- [[DINOv2]]
