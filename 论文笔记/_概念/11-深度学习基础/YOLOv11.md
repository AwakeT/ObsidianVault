---
type: concept
aliases: [YOLO11, Ultralytics YOLOv11]
---

# YOLOv11

## 定义
Ultralytics (2024) 发布的 YOLO 系列实时目标检测/分割模型，依赖 NMS 的单阶段检测器。

## 核心要点
1. 单阶段、anchor-free，依赖 NMS 后处理
2. COCO 上高效，但在 [[RF100-VL]] 等真实数据集上泛化弱于 DETR 系列
3. 放大模型在 RF100-VL 上不显著提升

## 代表工作
- [[RF-DETR]]: 对比对象，RF-DETR (nano) 与 YOLOv11 (medium) 性能持平

## 相关概念
- [[YOLOv8]]
- [[DETR]]
