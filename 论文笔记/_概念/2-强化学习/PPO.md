---
type: concept
aliases: [PPO, Proximal Policy Optimization, 近端策略优化]
---

# PPO

## 定义
一种 on-policy 策略梯度算法，用剪裁的重要性比代理目标限制每步策略更新幅度，兼顾稳定性与样本效率。

## 数学形式
$$
\mathcal{L}_{actor}(\theta) = -\,\mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

其中 $r_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{old}}(a_t\mid s_t)$，$\hat{A}_t$ 为 [[GAE]] 优势估计。

## 核心要点
1. 剪裁阈值 $\epsilon$（常用 0.2）防止策略更新过大。
2. 常与 [[GAE]] 和价值函数回归项联合优化。
3. 应用到流匹配策略时需把隐式密度转为显式高斯以计算 log 概率。

## 代表工作
- [[Qwen-VLA]]: RL 阶段用 PPO+GAE，action-chunk 级奖励/优势
- Schulman et al. 2017: 原始提出

## 相关概念
- [[GAE]]
- [[Reinforcement Learning]]
- [[RLinf]]
