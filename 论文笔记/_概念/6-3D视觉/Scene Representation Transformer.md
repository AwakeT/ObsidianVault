---
type: concept
aliases: [SRT, Scene Representation Transformer]
---

# Scene Representation Transformer

## 定义
Scene Representation Transformer 是一种先编码观测为场景表示，再通过 query 解码目标视角或目标位置属性的 Transformer 场景表示框架。

## 核心要点
1. 强调 latent scene representation 与 query-based decoding。
2. 适合把渲染、几何或点属性预测表述成条件查询。
3. D4RT 将这种思想扩展到动态视频 4D 重建。

## 代表工作
- [[D4RT]]: 受 SRT-style decoder 启发，设计 pointwise 4D query interface。

## 相关概念
- [[Query-Based Decoder]]
- [[Global Scene Representation]]
