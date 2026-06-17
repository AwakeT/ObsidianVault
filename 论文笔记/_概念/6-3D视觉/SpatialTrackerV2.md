---
type: concept
aliases: [STv2, SpatialTrackerV2]
---

# SpatialTrackerV2

## 定义
SpatialTrackerV2 是用于动态视频中 3D 点跟踪的强基线方法，能够估计动态点的空间轨迹。

## 核心要点
1. 能处理动态点对应，比纯 3D 重建方法更适合 motion tracking。
2. 依赖迭代 refinement，速度较慢。
3. 通常从单帧出发跟踪，完整稠密 4D 重建会留下遮挡空洞。

## 代表工作
- [[D4RT]]: 在吞吐和完整 4D 重建上优于 SpatialTrackerV2。

## 相关概念
- [[Point Tracking]]
- [[4D Reconstruction]]
