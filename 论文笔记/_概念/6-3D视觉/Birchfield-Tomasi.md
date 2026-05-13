---
type: concept
aliases: [BT 代价, Birchfield-Tomasi Metric]
---

# Birchfield-Tomasi

## 定义
一种对采样位置不敏感的亚像素匹配代价度量，通过考虑像素间的线性插值值来降低对精确像素对齐的敏感性。

## 核心要点
1. 比简单的绝对差值（AD）对亚像素偏移更鲁棒
2. OpenCV 的 StereoSGBM 用 BT 代价替代了原始 SGM 的互信息代价
3. 计算效率高，适合实时应用

## 代表工作
- [[SGBM]]: OpenCV SGBM 的默认匹配代价

## 相关概念
- [[Census Transform]]
- [[Mutual Information]]
- [[Stereo Matching]]
