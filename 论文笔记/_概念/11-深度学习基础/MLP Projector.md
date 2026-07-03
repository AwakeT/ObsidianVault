---
type: concept
aliases: [MLP投影层, Vision-Language Projector]
---

# MLP Projector

## 定义
VLM 中连接视觉编码器与语言模型的投影模块，通常为一或两层 MLP，把视觉特征映射到语言模型的 embedding 空间，实现模态对齐。

## 核心要点
1. **桥接模态**: 将视觉 token 投影为语言模型可接受的输入。
2. **常见结构**: 两层 MLP（带激活），结构简单但是 VLM 对齐的关键组件。
3. **训练**: 常在第一阶段单独训练 projector 以对齐视觉-文本特征。

## 代表工作
- [[LocateAnything]]: 用两层 MLP projector 桥接 [[Moon-ViT]] 与 [[Qwen2.5]]。

## 相关概念
- [[Moon-ViT]]
- [[CLIP]]
