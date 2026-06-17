---
type: concept
aliases: [MS COCO, Microsoft COCO, COCO Detection]
---

# COCO

## 定义
Lin et al. (2014) 的 Microsoft Common Objects in Context，目标检测/实例分割/关键点的旗舰基准（80 类，约 118k 训练图）。

## 核心要点
1. 检测领域标准评测，用 mAP@50:95 等指标
2. [[RF-DETR]] 指出近期 specialist 检测器**隐式过拟合 COCO**（架构、调度器都为 COCO 调优）
3. RF-DETR (2XL) 60.1 AP 首次让实时检测器破 60

## 代表工作
- [[RF-DETR]]: COCO 检测/分割主评测
- 几乎所有现代检测器

## 相关概念
- [[Objects365]]
- [[RF100-VL]]
