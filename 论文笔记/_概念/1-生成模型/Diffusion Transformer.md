---
type: concept
aliases: [DiT, Diffusion Transformer, 扩散 Transformer]
---

# Diffusion Transformer

## 定义
用 Transformer 替代 U-Net 作为扩散/流匹配模型骨架的架构，通过 [[AdaLN]] 等机制注入 timestep 与条件信息，可扩展性强。

## 核心要点
1. 每个 DiT block 含 [[Self-Attention|自注意力]] + Feed-Forward MLP，timestep 通过 [[AdaLN]] (Adaptive LayerNorm) 调制。
2. 相比 U-Net 更易随参数/数据规模扩展，是 SD3、Sora 等大模型的基础。
3. 在 VLA 中作为动作专家：把 VLM 隐状态与含噪动作块拼接做联合注意力，输出连续动作。

## 代表工作
- [[Qwen-VLA]]: 16-block 单流 DiT 动作专家（约 1.15B 参数）
- DiT (Peebles & Xie, 2023): 原始提出
- SD3 / Esser et al. 2024: rectified flow transformer

## 相关概念
- [[Flow Matching]]
- [[AdaLN]]
- [[Self-Attention]]
- [[RoPE]]
