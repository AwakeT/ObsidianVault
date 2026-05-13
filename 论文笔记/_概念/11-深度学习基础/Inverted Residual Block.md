---
type: concept
aliases: [倒残差块, MBConv, MobileNetV2 Block]
---

# Inverted Residual Block

## 定义
MobileNetV2 提出的轻量化模块：先用 pointwise 卷积扩展通道（expansion），再 depthwise 卷积，最后 pointwise 投影回低维，配合残差连接。

## 核心要点
1. "倒残差"：bottleneck 在低维，中间层在高维（与 ResNet bottleneck 相反）
2. 扩展因子 $t$ 控制中间层宽度
3. 3D 扩展后用于立体匹配的 cost volume encoder-decoder

## 代表工作
- [[MobileStereoNet]]: $v_2$ block 的 2D/3D 扩展
- [[MobileNetV2]]: 原始提出

## 相关概念
- [[Depthwise Separable Convolution]]
- [[MobileNetV1]]
- [[MobileNetV2]]
