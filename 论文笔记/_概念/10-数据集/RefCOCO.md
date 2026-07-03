---
type: concept
aliases: [RefCOCOg, RefCOCO+]
---

# RefCOCO

## 定义
基于 COCO 图像构建的指代表达理解（Referring Expression Comprehension, REC）数据集系列（RefCOCO / RefCOCO+ / RefCOCOg），每条样本含一段自然语言描述及其对应的目标边界框。

## 核心要点
1. **REC 任务**: 根据指代表达定位唯一目标区域。
2. **变体**: RefCOCOg 描述更长更自然，RefCOCO+ 禁用绝对方位词。
3. **评测**: F1@IoU / Accuracy@0.5。

## 代表工作
- RefCOCO (Yu et al., 2016) / RefCOCOg。
- [[LocateAnything]]: RefCOCOg val/test 上与顶级模型保持高度竞争（76.7 / 77.6 mean）。

## 相关概念
- [[Visual Grounding]]
- [[COCO]]
