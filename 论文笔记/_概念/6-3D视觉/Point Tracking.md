---
type: concept
aliases: [Point Tracking, TAP, Tracking Any Point, 点追踪]
---

# Point Tracking

## 定义
给定视频和 2D 查询点，预测该点在所有帧中的 2D 对应位置及可见性。

## 核心要点
1. TAP-Vid 提出标准基准
2. 需要处理遮挡和重新出现
3. VGGT 将其与 3D 重建统一在一个前馈框架中

## 代表工作
- [[VGGT]]: 端到端联合训练点追踪与 3D 重建
- [[CoTracker]]: 利用点间相关性的追踪方法

## 相关概念
- [[CoTracker]]
- [[Optical Flow]]
