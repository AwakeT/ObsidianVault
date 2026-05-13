---
type: concept
aliases: [RoPE, Rotary Position Embedding, 旋转位置编码]
---

# RoPE

## 定义
通过旋转矩阵将位置信息编码到 query 和 key 向量中的位置编码方法，具有相对位置感知和外推能力。

## 数学形式

$$$f(x_m, m) = R_m x_m$，其中 $R_m$ 为旋转矩阵$$

## 核心要点
1. 将绝对位置编码转化为相对位置感知
2. 支持序列长度外推
3. LLaMA 和 Reloc3r 等方法均采用

## 代表工作
- [[Reloc3r]]: ViT encoder 使用 RoPE 位置编码

## 相关概念
- [[Transformer]]
- [[Vision Transformer]]
