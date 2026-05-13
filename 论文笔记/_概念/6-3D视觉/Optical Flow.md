---
type: concept
aliases: [Optical Flow, 光流]
---

# Optical Flow

## 定义
估计视频相邻帧间每个像素的运动向量场，描述图像间的像素级对应关系。

## 数学形式

$$$F: \mathbb{R}^{H \times W} \to \mathbb{R}^{H \times W \times 2}$$$

## 核心要点
1. 稠密运动估计的基础任务
2. MonST3R 用光流与相机运动光流的差异检测动态物体
3. RAFT 是代表性的深度学习光流方法

## 代表工作
- [[MonST3R]]: 光流用于动态/静态分割和优化约束

## 相关概念
- [[Point Tracking]]
- [[Motion Segmentation]]
