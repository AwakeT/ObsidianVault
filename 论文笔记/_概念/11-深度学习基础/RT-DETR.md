---
type: concept
aliases: [Real-Time DETR]
---

# RT-DETR

## 定义
Zhao et al. (2024, "DETRs beat YOLOs") 提出的首个实时 [[DETR]]，通过高效混合编码器与不确定性最小 query 选择，在精度和速度上同时超越 YOLO。

## 核心要点
1. 证明 DETR 可在实时场景下击败 YOLO（无需 NMS）
2. 支持推理时调整解码器层数以权衡速度/精度
3. 是 [[D-FINE]] 的基础架构

## 代表工作
- [[RF-DETR]]: 对比基线
- [[D-FINE]]: 构建于 RT-DETR 之上

## 相关概念
- [[DETR]]
- [[Query Selection]]
