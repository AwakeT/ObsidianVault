---
type: concept
aliases: [Video Masked Autoencoder, VideoMAE]
---

# VideoMAE

## 定义
VideoMAE 是面向视频的 masked autoencoder 预训练方法，通过遮挡视频 token 并重建来学习时空表示。

## 核心要点
1. 可为视频 Transformer 提供强初始化。
2. 对需要长程时空理解的视频几何任务有帮助。
3. D4RT 的消融显示，VideoMAE 初始化显著优于随机初始化。

## 代表工作
- [[D4RT]]: 使用 VideoMAE 初始化 encoder，显著改善深度和 pose 指标。

## 相关概念
- [[Vision Transformer]]
- [[Masked Autoencoder]]
