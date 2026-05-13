---
type: concept
aliases: [感受野]
---

# Receptive Field

## 定义
网络某一层的单个神经元在输入图像上所能"看到"的区域大小。

## 核心要点
1. 通过堆叠卷积层、使用空洞卷积或下采样扩大感受野
2. 感受野不足时难以处理大面积无纹理区域
3. RAFT-Stereo 的 Multi-Level GRU 通过多分辨率操作有效扩大感受野

## 代表工作
- [[RAFT-Stereo]]: Multi-Level GRU 跨分辨率传播扩大有效感受野

## 相关概念
- [[Dilated Convolution]]
- [[U-Net]]
