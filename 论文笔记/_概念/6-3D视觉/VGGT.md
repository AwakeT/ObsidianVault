---
type: concept
aliases: [VGGT, Visual Geometry Grounded Transformer]
---

# VGGT

## 定义
基于大规模 Transformer 的前馈式多视图 3D 重建网络（~1.2B 参数），无需几何后处理即可从 1 到数百张图像预测相机参数、深度图、点云和 3D 点轨迹。CVPR 2025 Best Paper Award。

## 核心要点
1. Alternating-Attention 机制（帧内+全局自注意力交替）
2. DINOv2 tokenizer + DPT 稠密预测头
3. 多任务联合训练（相机、深度、点云、追踪）互补增益
4. 前馈 0.2 秒即超越需要 7-10 秒后处理的 DUSt3R/MASt3R

## 代表工作
- [[VGGT]]: 本方法的完整论文笔记
- [[PROSPECT]]: 使用 VGGT 特征的下游工作
- [[DyGeoVLN]]: 使用 VGGT 特征的导航工作

## 相关概念
- [[DUSt3R]]
- [[MASt3R]]
- [[Alternating Attention]]
- [[Point Map]]
- [[3D Reconstruction]]
