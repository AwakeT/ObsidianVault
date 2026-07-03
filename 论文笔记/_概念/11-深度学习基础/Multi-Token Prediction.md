---
type: concept
aliases: [MTP, 多token预测]
---

# Multi-Token Prediction (MTP)

## 定义
一类在单个前向步内并行预测多个未来 token 的解码技术，用于减少自回归生成的解码步数、提升吞吐，常与投机解码（speculative decoding）结合。

## 核心要点
1. **常见形式**: (i) 随机选位置训练模型预测后续 span（next-block prediction）；(ii) 掩码部分 token 让模型重建原文（类似掩码扩散建模）。
2. **structure-agnostic（结构无关）的弊端**: 通用 MTP 把输入当作泛型 token 流，主要捕获共现相关性；对边界框等紧耦合单元会学到跨边界、跨类别的不可靠 token 组合。
3. **改进方向**: 将预测块与结构化单元对齐（如 [[Parallel Box Decoding]] 的 box-aligned MTP）。

## 代表工作
- [[LocateAnything]]: 提出 box-aligned MTP（即 PBD），优于 structure-agnostic 的 SDLM、[[Block Diffusion]]。

## 相关概念
- [[Next-Token Prediction]]
- [[Parallel Box Decoding]]
- [[Block Diffusion]]
