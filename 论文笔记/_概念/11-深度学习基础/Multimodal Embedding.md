---
type: concept
aliases: [多模态嵌入, 多模态编码]
---

# Multimodal Embedding

## 定义
将图像、文本等多模态输入映射到同一对齐潜空间的嵌入表示，使跨模态检索（如文本查图、图查图）可通过向量相似度实现。

## 核心要点
1. 现代多模态嵌入随 scaling law 与文本高度对齐
2. 对比学习类（CLIP/SigLIP/QQMM）优于自回归 VLM 隐藏态
3. 直接保留细粒度视觉语义，避免图像转文字的信息损失

## 代表工作
- [[RAVEN]]: 用多模态嵌入构建视觉记忆，评测多种嵌入模型

## 相关概念
- [[CLIP]]
- [[SigLIP]]
- [[Visuo-Spatio-Temporal Memory]]
