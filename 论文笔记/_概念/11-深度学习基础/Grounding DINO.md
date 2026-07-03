---
type: concept
aliases: [GroundingDINO]
---

# Grounding DINO

## 定义
将 DINO 检测器与 grounded 预训练结合的开集（open-set）目标检测器，可根据文本短语检测任意类别物体，是 grounding/detection 领域的强专用基线。

## 核心要点
1. **开集检测**: 支持检测训练中未见的类别，由文本 prompt 驱动。
2. **专用架构**: 基于 DETR/DINO 的检测头，非生成式 token 解码。
3. **基线地位**: 在高 IoU（0.95）上常领先生成式 VLM，但泛化任务覆盖窄。

## 代表工作
- Grounding DINO (Liu et al., 2023): 开集 grounding 检测。
- [[LocateAnything]]: 作为开集专用检测器对比基线。

## 相关概念
- [[Object Detection]]
- [[Visual Grounding]]
- [[Grounded-SAM]]
