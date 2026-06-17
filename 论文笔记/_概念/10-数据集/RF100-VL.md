---
type: concept
aliases: [Roboflow100-VL, RF100VL]
---

# RF100-VL

## 定义
Robicheaux et al. (2025) 提出的 Roboflow100-VL，由 100 个多样真实世界数据集组成的多域目标检测基准，专为评测视觉-语言模型/检测器的泛化而设计。

## 核心要点
1. 100 个分布各异的数据集，覆盖 OOD 类别/成像模态，作为"任意目标域迁移性"的代理
2. 揭示 [[VLM]]（如 [[Grounding-DINO|GroundingDINO]]）对预训练外类别泛化差
3. [[RF-DETR]] (2XL) 在此超 GroundingDINO/LLMDet 且快约 20×；同作者团队

## 代表工作
- [[RF-DETR]]: 真实世界泛化主评测基准

## 相关概念
- [[COCO]]
- [[Grounding-DINO]]
