---
type: concept
aliases: [MAE, Masked Autoencoder, 掩码自编码器]
---

# Masked Autoencoder

## 定义
Masked Autoencoder 是一种自监督预训练范式：随机遮挡输入的一部分，只用可见 token 编码，再让解码器重建被遮挡内容。

## 核心要点
1. 常用于视觉 Transformer 预训练。
2. 通过高遮挡率迫使模型学习全局语义和结构。
3. 视频版本通常进一步建模时间维度上的运动与对应关系。

## 代表工作
- [[VideoMAE]]: 将 masked autoencoder 预训练扩展到视频。
- [[D4RT]]: 使用 VideoMAE 初始化视频 encoder。

## 相关概念
- [[Vision Transformer]]
- [[Self-Supervised Learning]]
