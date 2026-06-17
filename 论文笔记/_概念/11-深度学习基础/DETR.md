---
type: concept
aliases: [Detection Transformer, DEtection TRansformer]
---

# DETR

## 定义
Carion et al. (2020) 提出的端到端目标检测 Transformer，用一组可学习的 object query 与二分图匹配损失直接预测目标集合，去除了 NMS、anchor 等手工组件。

## 核心要点
1. encoder-decoder Transformer 架构，object query 直接对应检测框
2. 用匈牙利匹配（二分图匹配）做集合预测损失，无需 NMS
3. 早期 DETR 收敛慢、推理慢，后续工作（[[Deformable Attention|Deformable DETR]]、[[RT-DETR]]、[[LW-DETR]]）改善之

## 代表工作
- [[RF-DETR]]: 用权重共享 NAS 现代化实时 DETR
- [[LW-DETR]] / [[RT-DETR]] / [[D-FINE]]: 实时 DETR 系列

## 相关概念
- [[Transformer]]
- [[Self-Attention]]
- [[Query Selection]]
