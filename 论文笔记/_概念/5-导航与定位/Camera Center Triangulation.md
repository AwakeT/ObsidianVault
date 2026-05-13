---
type: concept
aliases: [Camera Center Triangulation, 相机中心三角化]
---

# Camera Center Triangulation

## 定义
从两条或多条已知方向的射线交汇确定相机中心位置的方法，最小化点到射线的距离。

## 数学形式

$$$c_q = \arg\min_c \sum_k \|(I - \hat{d}_k \hat{d}_k^\top)(c - c_{d_k})\|^2$$$

## 核心要点
1. 通过 SVD 求解最小二乘问题
2. Reloc3r 用此恢复查询相机的绝对位置

## 代表工作
- [[Reloc3r]]: 从相对平移方向三角化绝对相机位置

## 相关概念
- [[Motion Averaging]]
- [[PnP]]
