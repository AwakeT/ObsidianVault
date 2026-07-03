---
type: concept
aliases: [目标检测, 物体检测]
---

# Object Detection

## 定义
在图像中定位并分类所有感兴趣物体的任务，输出每个物体的边界框及类别标签。

## 核心要点
1. **范式**: 专用检测器（DETR、Faster R-CNN、DINO）vs 生成式 VLM（坐标 token 生成）。
2. **开集 vs 闭集**: 开集检测器（[[Grounding DINO]]）可检测训练未见类别；闭集仅限固定类别。
3. **密集检测**: 单图大量物体（如 VisDrone、Dense200、SKU110K）对串行解码是严峻挑战。
4. **评测**: COCO、LVIS（长尾），F1@IoU / mAP。

## 代表工作
- [[LocateAnything]]: 生成式 VLM 检测，PBD 并行解码。
- [[Grounding DINO]]: 开集专用检测器。

## 相关概念
- [[Visual Grounding]]
- [[Bounding Box]]
- [[COCO]]
- [[LVIS]]
