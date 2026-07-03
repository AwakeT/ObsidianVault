---
type: concept
aliases: [MonST3R, Motion DUSt3R]
---

# MonST3R

## 定义
DUSt3R 谱系面向**动态场景**的 pointmap 重建方法：在动态视频上估计逐帧几何与相机运动，通过对动态内容的处理与全局对齐优化，得到时序一致的 4D 重建。

## 核心要点
1. 把 DUSt3R 的静态成对重建扩展到含运动物体的动态视频。
2. 仍依赖**全局对齐优化（global alignment）**，属于离线方法、非在线。
3. 常作为动态视频深度/位姿估计的强基线。

## 代表工作
- MonST3R 原文：动态场景 pointmap 重建。
- 被 [[CUT3R]]、[[TTT3R]] 作为动态场景对比基线（MonST3R-GA）。

## 相关概念
- [[DUSt3R]]
- [[Point Map]]
- [[Global Optimization]]
- [[Video Depth Estimation]]
