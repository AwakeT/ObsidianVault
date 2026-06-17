---
type: concept
aliases: [Next-Token Prediction, 下一 token 预测, NTP, 自回归损失]
---

# Next-Token Prediction

## 定义
自回归语言/多模态建模的标准训练目标：给定前文 token 预测下一个 token，用交叉熵最大化条件似然。

## 数学形式
$$
\mathcal{L} = -\sum_i \log p_\theta(w_i \mid w_{<i}, o_{1:t})
$$

## 核心要点
1. 在 VLA 联合训练中保留 NTP 损失于视觉语言数据，稳定语言 grounding，防止灾难性遗忘。
2. Qwen-VLA SFT 阶段把 VL NTP 权重设为 0.1，动作流匹配损失为 1.0。

## 代表工作
- [[Qwen-VLA]]: 视觉语言损失保留主干能力

## 相关概念
- [[Vision-Language-Action]]
- [[Supervised Fine-Tuning]]
