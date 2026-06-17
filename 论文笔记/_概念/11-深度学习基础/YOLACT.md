---
type: concept
aliases: [分割原型, Prototype Mask]
---

# YOLACT

## 定义
Bolya et al. (2019) 提出的实时实例分割方法，通过预测一组"原型掩码（prototype）"与每个实例的线性组合系数生成掩码。

## 核心要点
1. 用 prototype + 系数线性组合的方式高效生成实例掩码
2. "分割原型"概念被后续工作借鉴
3. [[RF-DETR]]-Seg 的像素嵌入图可解释为分割原型

## 代表工作
- [[RF-DETR]]: 像素嵌入解释为 prototype
- [[FastInst]]: 实时实例分割

## 相关概念
- [[MaskDINO]]
- [[Open-Vocabulary Segmentation]]
