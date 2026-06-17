---
type: concept
aliases: [Action Chunking, 动作块, action chunk]
---

# Action Chunking

## 定义
一次预测并执行未来多步动作（一个动作块）而非单步动作的策略输出方式，减少推理频率、提升时序一致性、抑制误差累积。

## 数学形式
预测时域 $H$ 步的动作序列 $a_{t:t+H-1}$，组织为张量 $Y\in\mathbb{R}^{H\times K}$。

## 核心要点
1. 减少决策频率，降低 compounding error 与抖动。
2. 与[[Flow Matching|流匹配]]/扩散动作专家天然契合（一次生成整块）。
3. Qwen-VLA 用 $H=16$（导航为 8 waypoint）；RL 中每块一个标量奖励与优势。

## 代表工作
- [[Qwen-VLA]]: 统一动作-轨迹空间的动作块预测
- [[Diffusion Policy]] / [[Pi0]]: 动作块生成

## 相关概念
- [[Flow Matching]]
- [[Diffusion Transformer]]
