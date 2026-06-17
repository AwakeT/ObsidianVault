---
type: concept
aliases: [GAE, Generalized Advantage Estimation, 广义优势估计]
---

# GAE

## 定义
通过对多步 TD 残差做指数加权平均来估计优势函数，用偏差-方差权衡参数 $\lambda$ 平衡估计质量。

## 数学形式
$$
\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

## 核心要点
1. $\gamma$ 折扣因子、$\lambda$ 迹衰减（Qwen-VLA 用 $\gamma=0.99, \lambda=0.95$）。
2. 把稀疏的 episode 级奖励通过价值基线传播到各步/各动作块，解决信用分配。
3. 常与 [[PPO]] 配合。

## 代表工作
- [[Qwen-VLA]]: 把 SimplerEnv 的稀疏二值奖励经 GAE 传播到 action chunk
- Schulman et al. 2015: 原始提出

## 相关概念
- [[PPO]]
- [[Reinforcement Learning]]
