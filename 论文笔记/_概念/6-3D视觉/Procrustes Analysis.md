---
type: concept
aliases: [普氏分析, Procrustes, Procrustes Analysis, 正交普氏问题]
---

# Procrustes Analysis

## 定义
求解两组对应点云之间最优刚体（或相似）变换的经典方法：给定源点集与目标点集，闭式求出使对齐残差最小的旋转、平移（及可选尺度）。正交 Procrustes 问题通过对协方差矩阵做 SVD 得到最优旋转。

## 数学形式
$$
R^\star = \arg\min_{R \in SO(3)} \| RA - B \|_F^2 = UV^\top,\quad \text{s.t. } A B^\top = U\Sigma V^\top
$$

## 核心要点
1. 闭式解，无需迭代，比 [[PnP]] 更简单更快。
2. 与 [[Umeyama Algorithm]]（含尺度）密切相关，是相似变换估计的基础。
3. 在 pointmap 类方法中，可由同一帧的**局部坐标 pointmap $X_{i,i}$** 与**全局坐标 pointmap $X_{i,1}$** 的点对应直接恢复相机位姿。

## 代表工作
- [[MUSt3R]]：用 Procrustes 分析从局部/全局 pointmap 恢复相对位姿，替代 PnP 加速推理。

## 相关概念
- [[Umeyama Algorithm]]
- [[Point Cloud Alignment]]
- [[PnP]]
- [[Camera Pose Estimation]]
