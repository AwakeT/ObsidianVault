---
type: concept
aliases: [RLinf]
---

# RLinf

## 定义
一个灵活高效的大规模强化学习框架，通过 macro-to-micro flow transformation 支持大模型 RL 训练；Qwen-VLA 的 RL 阶段基于此实现。

## 核心要点
1. 采用解耦的 client-server 架构：远端 benchmark server 托管仿真环境，训练进程经网络接口通信，使仿真负载与 GPU 策略优化独立扩展。
2. 支持大规模并行环境 rollout（Qwen-VLA 用 N=128 并行实例）。

## 代表工作
- [[Qwen-VLA]]: RL 后训练框架
- Yu et al. 2025 (arXiv:2509.15965): 原始提出

## 相关概念
- [[Reinforcement Learning]]
- [[PPO]]
