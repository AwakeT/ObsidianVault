---
type: concept
aliases: [边界框, bbox]
---

# Bounding Box

## 定义
用于框定图像中物体位置的矩形区域，通常表示为四个坐标 $(x_1, y_1, x_2, y_2)$（左上角与右下角）或 $(x, y, w, h)$（中心 + 宽高）。

## 核心要点
1. **几何耦合**: 四个坐标强相关，构成紧耦合的原子单元；逐 token 解码会破坏这种耦合。
2. **VLM 序列化**: 归一化到 $[0,1000]$ 后离散为坐标 token，存在 Textual Digits / Quantized Tokens 两种表示。
3. **质量度量**: IoU（交并比）衡量预测框与 gt 框的重叠程度。

## 代表工作
- [[LocateAnything]]: 把整个 box 作为不可分割原子单元一次并行解码（[[Parallel Box Decoding]]）。

## 相关概念
- [[Object Detection]]
- [[Visual Grounding]]
- [[Parallel Box Decoding]]
