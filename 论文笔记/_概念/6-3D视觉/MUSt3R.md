---
type: concept
aliases: [MUSt3R, Multi-view DUSt3R]
---

# MUSt3R

## 定义
DUSt3R 谱系把**两视图**扩展为**对称多视图（N-view）** 的重建网络：用共享权重对称解码器 + 多层记忆机制，一次性预测所有视图在统一坐标系下的 pointmap，支持离线与在线，在无标定 VO / 位姿 / 深度上达 SOTA。

## 核心要点
1. 对称共享权重 Siamese 解码器 + 可学习参考嵌入 $B$（否则崩溃）。
2. 类 KV-cache 的 [[Memory Mechanism]]，把 $O(N^2)$ 成对推理 + 全局对齐消除。
3. 同时输出局部 $X_{i,i}$ 与全局 $X_{i,1}$ pointmap，用 [[Procrustes Analysis]] 恢复位姿。
4. Global 3D Feedback 从倒数第二层回注；log-space 损失在多视图下更稳。

## 代表工作
- [[MUSt3R]]（本篇，CVPR 2025）：Multi-view Network for Stereo 3D Reconstruction（Naver Labs Europe）。

## 相关概念
- [[DUSt3R]]
- [[Memory Mechanism]]
- [[Procrustes Analysis]]
- [[Visual Odometry]]
- [[Point Map]]
