---
type: concept
aliases: [位置编码, PE]
---

# Positional Encoding

## 定义
为 Transformer 输入添加位置信息的编码方式，使模型能区分不同位置的 token。

## 数学形式
$$PE(pos, 2i) = \sin(pos / 10000^{2i/d})$$
$$PE(pos, 2i+1) = \cos(pos / 10000^{2i/d})$$

## 核心要点
1. 正弦/余弦位置编码（绝对位置）
2. RoPE（旋转位置编码，相对位置）
3. 可学习位置编码

## 代表工作
- [[CREStereo]]: 特征图位置编码
- [[FoundationStereo]]: DT 中使用 Cosine 位置编码

## 相关概念
- [[Transformer]]
- [[Self-Attention]]
