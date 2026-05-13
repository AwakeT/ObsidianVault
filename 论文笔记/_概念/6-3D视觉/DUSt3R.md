---
type: concept
aliases: [DUSt3R, Dense and Unconstrained Stereo 3D Reconstruction]
---

# DUSt3R

## 定义
基于 ViT 的两视图 3D 重建方法，通过 pointmap 表示直接从图像对回归稠密 3D 点云，隐式推断相机内参和相对位姿。

## 核心要点
1. 使用 CroCo 预训练的 ViT encoder-decoder 架构
2. 输出 pointmap 表示：将每个像素映射到 3D 空间
3. 需要测试时全局优化来融合成对预测
4. 开创了基于 Transformer 的几何基础模型范式

## 代表工作
- [[VGGT]]: 将 DUSt3R 的 pointmap 表示扩展到前馈多视图重建
- [[MonST3R]]: 微调 DUSt3R 以处理动态场景
- [[Reloc3r]]: 利用 DUSt3R 预训练权重进行位姿回归

## 相关概念
- [[MASt3R]]
- [[CroCo]]
- [[Point Map]]
