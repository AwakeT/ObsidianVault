---
type: concept
aliases: [空间记忆, Spatial Memory]
---

# Spatial Memory

## 定义
一种用于增量式 3D 重建的外部记忆机制：网络维护一组随时间累积的历史 3D/视觉特征（key/value），通过查询该记忆把新观测帧对齐并预测到统一全局坐标系，从而无需基于优化的全局对齐。

## 核心要点
1. 通常由**稠密工作记忆（最近若干帧）+ 稀疏长期记忆（稀疏化保留的历史 token）** 组成。
2. 通过 [[Cross-Attention]] 做记忆读写：query 来自当前帧，key/value 来自记忆。
3. 记忆 value 同时携带**几何 + 外观**特征，使检索可基于外观与距离双重线索。
4. 长期记忆借鉴视频分割中的 [[XMem]]，按累积注意力权重稀疏化。

## 代表工作
- [[Spann3R]]：首次把空间记忆用于 DUSt3R 式在线增量重建。
- [[CUT3R]]：用持久 state token 作为隐式空间记忆的推广。

## 相关概念
- [[Memory Mechanism]]
- [[Cross-Attention]]
- [[Global Scene Representation]]
- [[XMem]]
