---
type: concept
aliases: [记忆框架, 多模态记忆框架]
---

# Memory Harness

## 定义

Memory Harness 是一种为长程具身任务构造压缩历史上下文的机制，通常把历史图像、时间、动作和文本摘要组织后重新注入 planner prompt。

## 数学形式

$$
m_i=\langle i,\tau_i,o_i,a_i,g\rangle,\quad
s_t=\Phi(o_t,\mathcal{M}_t,g)
$$

## 核心要点

1. 用有限历史代替完整视频上下文，降低长程任务的 token 与延迟压力。
2. 可以混合图像历史和文本子任务日志。
3. 关键设计包括保留哪些帧、如何采样、哪些推理内容写入记忆。

## 代表工作

- [[Vesta]]: 使用 image-text memory harness 支持长程真实机器人 planning。

## 相关概念

- [[Action Planning]]
- [[Egocentric Observation]]
- [[Memory Mechanism]]
