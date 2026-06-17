---
type: concept
aliases: [AdaLN, Adaptive Layer Normalization, 自适应层归一化, AdaLN-Zero]
---

# AdaLN

## 定义
自适应层归一化：用条件信息（如 timestep embedding）回归出 LayerNorm 的缩放与平移参数（以及残差门控），把条件注入 [[Diffusion Transformer|DiT]] 等网络。

## 数学形式
$$
\text{AdaLN}(h, c) = \gamma(c)\odot \text{LayerNorm}(h) + \beta(c)
$$

其中 $\gamma, \beta$ 由条件 $c$（如 timestep）通过 MLP 回归。

## 核心要点
1. 是 DiT 注入扩散/流匹配 timestep 的标准机制。
2. AdaLN-Zero 变体初始化为恒等，训练更稳定。
3. Qwen-VLA 动作专家用 AdaLN 做 timestep 条件，输出 AdaLN modulation 占 4.7M 参数。

## 代表工作
- Peebles & Xie 2023 (DiT): 提出 AdaLN-Zero
- [[Qwen-VLA]]: 动作专家 timestep 条件

## 相关概念
- [[Diffusion Transformer]]
- [[Flow Matching]]
