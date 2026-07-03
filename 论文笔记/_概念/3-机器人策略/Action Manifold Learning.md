---
type: concept
aliases: [AML, 动作流形学习]
---

# Action Manifold Learning

## 定义

Action Manifold Learning 是 [[ABot-M0]] 提出的动作生成范式：模型直接预测低维动作流形上的 clean action chunk，而不是预测高维噪声或速度。

## 数学形式

$$
\hat{A}_t = V_\theta(\phi_t, A_t^\tau, q_t)
$$

## 核心要点

1. 基于 [[Action Manifold Hypothesis]]，认为有效动作序列位于低维、平滑、物理可行的流形上。
2. 模型输出 action，但训练 loss 可以映射到 velocity space，保留 [[Flow Matching]] 的噪声级重加权优势。
3. 对高维动作空间和长 action chunk 更稳，适合双臂、灵巧手和全身控制。

## 代表工作

- [[ABot-M0]]: 用 AML 替代 noise prediction action expert，在 LIBERO-Plus 和 RoboCasa 上取得更好表现。

## 相关概念

- [[Action Manifold Hypothesis]]
- [[Diffusion Transformer]]
- [[Flow Matching]]
- [[Action Chunking]]
