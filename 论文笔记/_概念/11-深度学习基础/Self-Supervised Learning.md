---
type: concept
aliases: [SSL, 自监督学习]
---

# Self-Supervised Learning

## 定义
Self-Supervised Learning 是从数据自身构造监督信号的学习范式，常通过预测被遮挡内容、对比不同视图或恢复扰动输入来学习通用表示。

## 核心要点
1. 不依赖人工标签，适合大规模预训练。
2. 视觉领域常见路线包括 contrastive learning、masked image modeling 和 masked autoencoding。
3. 视频自监督学习通常还会利用时间连续性、运动和跨帧对应。

## 代表工作
- [[Masked Autoencoder]]: 通过遮挡和重建学习视觉表示。
- [[VideoMAE]]: 面向视频的 masked autoencoder 预训练。

## 相关概念
- [[Contrastive Loss]]
- [[Vision Transformer]]
