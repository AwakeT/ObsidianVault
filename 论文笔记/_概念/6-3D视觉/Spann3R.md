---
type: concept
aliases: [Spann3R, Spanner]
---

# Spann3R

## 定义
DUSt3R 谱系首个把**外部空间记忆**用于在线增量式稠密 3D 重建的方法：逐帧回归全局坐标系下的 pointmap，去掉基于优化的全局对齐，实现约 65 fps 的实时重建。

## 核心要点
1. 复用 [[DUSt3R]] 权重与两个交织解码器，仅加轻量记忆编码器。
2. [[Spatial Memory]] = 稠密工作记忆（5 帧）+ 稀疏长期记忆（约 4000 token，借鉴 [[XMem]]）。
3. 每帧 pointmap 直接在首帧全局坐标系，消除 [[Global Optimization]]。
4. 无 BA → 长序列漂移、闭环误差；镜面场景离群。

## 代表工作
- [[Spann3R]]（本篇，3DV 2025）：3D Reconstruction with Spatial Memory。

## 相关概念
- [[DUSt3R]]
- [[Spatial Memory]]
- [[3D Reconstruction]]
- [[Point Map]]
