---
type: concept
aliases: [DepthCrafter]
---

# DepthCrafter

## 定义
基于视频扩散模型的视频深度估计方法，生成时间一致的深度图，但输出 scale/shift-invariant 深度。

## 核心要点
1. 时间一致性优于单帧方法
2. 输出为 scale/shift-invariant，不适合 3D 应用
3. MonST3R 在 metric 条件下显著超越

## 代表工作
- [[MonST3R]]: metric 深度条件下全面超越 DepthCrafter

## 相关概念
- [[Optical Flow]]
