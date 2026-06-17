---
type: concept
aliases: [Segment Anything 2, SAM 2]
---

# SAM2

## 定义
Ravi et al. (2024) 提出的 Segment Anything 2，可对图像和视频做可提示的分割，是 [[SAM]] 的视频扩展。

## 核心要点
1. 支持图像与视频的统一可提示分割，带记忆机制
2. 常用作伪标注工具生成实例掩码
3. [[RF-DETR]] 用 SAM2 在 [[Objects365]] 上伪标注实例掩码，预训练 RF-DETR-Seg

## 代表工作
- [[RF-DETR]]: 用 SAM2 伪标注 Objects365 实例掩码

## 相关概念
- [[SAM]]
- [[Open-Vocabulary Segmentation]]
