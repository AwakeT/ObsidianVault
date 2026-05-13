---
type: concept
aliases: [DPT, Dense Prediction Transformer]
---

# DPT

## 定义
基于 ViT 的稠密预测架构，通过多尺度 token 上采样和融合实现像素级输出（深度、分割等）。

## 核心要点
1. 从 ViT 不同层提取 token 进行多尺度融合
2. 广泛用于单目深度估计
3. VGGT 和 DUSt3R 使用 DPT 作为稠密预测头

## 代表工作
- [[VGGT]]: 使用 DPT 输出深度图、点云图和追踪特征
- [[DUSt3R]]: 使用 DPT 输出 pointmap

## 相关概念
- [[Vision Transformer]]
- [[Transformer]]
