---
type: concept
aliases: [Multi-modal 3D Scene Graph Navigation]
---

# MSGNav

## 定义
MSGNav（Unleashing the power of multi-modal 3D scene graph for zero-shot embodied navigation, arXiv:2511.10376, 2025）：利用多模态 3D 场景图进行零样本具身导航的方法，是 GOAT-Bench 上的强 training-free 基线。

## 核心要点
1. 基于多模态 [[Scene Graph|3D 场景图]] 表示进行零样本导航
2. 在 GOAT-Bench VAL-UNSEEN 上达 52.0 SR / 29.6 SPL（被 [[EvoMemNav]] 的 59.6/38.9 超越）
3. 属于检测器/场景图中心的记忆范式

## 代表工作
- [[EvoMemNav]]: 将 MSGNav 作为对比基线，论证图像锚定记忆相对场景图记忆的优势

## 相关概念
- [[Scene Graph]]
- [[GOAT-Bench]]
- [[Object Navigation]]
