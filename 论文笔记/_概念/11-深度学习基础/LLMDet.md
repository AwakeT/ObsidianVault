---
type: concept
aliases: []
---

# LLMDet

## 定义
Fu et al. (2025) 提出的开放词表目标检测器，在大语言模型监督下学习强开放词表检测能力。

## 核心要点
1. 用 LLM 监督增强开放词表检测
2. 在 [[RF100-VL]] 上与 [[Grounding-DINO|GroundingDINO]] (tiny) 同为 62.3 AP，但延迟约 308ms
3. 被 [[RF-DETR]] (2XL) 以约 1/20 延迟超越

## 代表工作
- [[RF-DETR]]: RF100-VL 对比对象

## 相关概念
- [[Grounding-DINO]]
- [[VLM]]
- [[LLM]]
