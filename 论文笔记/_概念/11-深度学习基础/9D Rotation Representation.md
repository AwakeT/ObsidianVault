---
type: concept
aliases: [9D Rotation Representation, 9D 旋转表示]
---

# 9D Rotation Representation

## 定义
使用 9 维向量（3×3 矩阵展平）表示旋转，通过 SVD 正交化映射到 SO(3) 旋转矩阵，具有连续性避免万向锁等问题。

## 数学形式

$$$R = U V^T$，其中 $U\Sigma V^T = \text{SVD}(M_{3\times3})$$$

## 核心要点
1. 比四元数和欧拉角更适合神经网络回归
2. 通过 SVD 确保输出为正交矩阵
3. Zhou et al. (CVPR 2019) 证明 6D 表示的连续性

## 代表工作
- [[Reloc3r]]: 使用 9D 表示回归旋转矩阵

## 相关概念
- [[Rotation Loss]]
