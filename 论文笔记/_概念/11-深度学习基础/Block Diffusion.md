---
type: concept
aliases: [块扩散, Semi-Autoregressive Diffusion]
---

# Block Diffusion

## 定义
一类半自回归扩散语言模型范式：把序列切分为固定大小的 token 块，块内并行去噪解码、块间保持因果依赖，从而兼顾并行加速与 KV-cache 兼容性。

## 核心要点
1. **固定块并行**: 在块内做非自回归扩散解码，块间因果。
2. **structure-agnostic**: 块边界按固定大小切分，与边界框等结构单元不对齐，会学到跨边界模式。
3. **速度-精度 trade-off**: 增大块大小吞吐增益边际、F1 持续下降（被 LocateAnything 消融验证）。

## 代表工作
- Block Diffusion (Arriola et al., 2025): 半自回归块扩散。
- [[LocateAnything]]: 作为 structure-agnostic MTP 对比对象，box-aligned PBD 优于它。

## 相关概念
- [[Multi-Token Prediction]]
- [[Parallel Box Decoding]]
- [[Diffusion Model]]
