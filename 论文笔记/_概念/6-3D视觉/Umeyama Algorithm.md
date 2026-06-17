---
type: concept
aliases: [Umeyama, Umeyama Alignment, Umeyama算法]
---

# Umeyama Algorithm

## 定义
Umeyama Algorithm 是一种基于 SVD 的点集相似变换估计算法，用于求解两组对应点之间的旋转、平移和可选尺度。

## 核心要点
1. 输入为两组有对应关系的点。
2. 输出通常是 Sim(3) 或 SE(3) 对齐变换。
3. 常用于点云对齐、相机位姿估计和视频 chunk 拼接。

## 代表工作
- [[D4RT]]: 用 Umeyama 从不同相机参考系下的同一点坐标恢复相对 pose，并用于长视频 chunk 对齐。

## 相关概念
- [[Point Cloud Alignment]]
- [[Camera Pose Estimation]]
