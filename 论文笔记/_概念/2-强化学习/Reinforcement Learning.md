---
type: concept
aliases: [强化学习, Reinforcement Learning, RL]
---

# Reinforcement Learning

## 定义
智能体通过与环境交互、根据奖励信号优化策略以最大化累积回报的学习范式。

## 核心要点
1. 核心要素：状态、动作、奖励、策略、价值函数。
2. 在 VLA 后训练中用于把模仿学习 (SFT) 优化的似然目标，进一步对齐到**闭环任务成功**这一真正目标。
3. 稀疏二值成功奖励 + 仿真 rollout 是机器人 RL 常见设置。

## 代表工作
- [[Qwen-VLA]]: SFT 后用 RL 优化闭环成功率，产出 Instruct 版本
- [[PPO]] / [[GAE]]: 常用算法组件

## 相关概念
- [[PPO]]
- [[GAE]]
- [[RLinf]]
- [[Supervised Fine-Tuning]]
