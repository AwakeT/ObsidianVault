---
type: concept
aliases: [位置嵌入, 位置编码]
---

# Positional Embedding

## 定义
向 Transformer token 注入位置信息的可学习或固定向量，使模型感知 token 的空间/序列顺序。详见 [[Positional Encoding]]。

## 核心要点
1. ViT 中为每个 patch token 分配位置嵌入
2. 可通过插值适配不同输入分辨率
3. [[RF-DETR]] 预分配对应（最大分辨率 / 最小 patch size）的 N 个位置嵌入，对较小分辨率或较大 patch 做插值，使所有分辨率作为 NAS 旋钮可用

## 代表工作
- [[RF-DETR]]: 位置嵌入插值支持多分辨率 NAS

## 相关概念
- [[Positional Encoding]]
- [[Vision Transformer]]
