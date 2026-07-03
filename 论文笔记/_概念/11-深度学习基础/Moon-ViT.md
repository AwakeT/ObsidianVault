---
type: concept
aliases: [MoonViT]
---

# Moon-ViT

## 定义
Kimi Team 提出的原生分辨率（native-resolution）视觉 Transformer 编码器，可直接处理任意分辨率图像并保留细粒度空间细节，常用作 VLM 的视觉 backbone。

## 核心要点
1. **native resolution**: 不强制 resize 到固定尺寸，保留高精度定位所需的细粒度空间信息。
2. **VLM 视觉编码器**: 提取视觉 token 后经 projector 送入语言模型。

## 代表工作
- [[LocateAnything]]: 用 Moon-ViT 作视觉编码器以支持高精度定位。

## 相关概念
- [[Vision Transformer]]
- [[MLP Projector]]
- [[Qwen2.5]]
