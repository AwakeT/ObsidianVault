---
type: concept
aliases: [欧拉积分, Euler Integration, Euler step]
---

# Euler Integration

## 定义
求解常微分方程 (ODE) 的最简单数值方法，用当前导数线性外推下一步；在流匹配/扩散采样中用于从噪声迭代生成样本。

## 数学形式
对速度场 $v_\theta$，从 $\tau=1$ 到 $\tau=0$ 分若干步：

$$
Y^{\tau - \Delta\tau} = Y^\tau - \Delta\tau \cdot v_\theta(Y^\tau, \tau)
$$

## 核心要点
1. [[Flow Matching|流匹配]]推理用少量欧拉步即可生成高质量动作，低延迟。
2. 步数越多越精确但越慢；机器人实时控制偏好少步。
3. 可把确定性 probability-flow ODE 转为 SDE（每步注入受控噪声），使每步成为显式高斯，便于计算 log 概率（Qwen-VLA RL 阶段所用）。

## 代表工作
- [[Qwen-VLA]]: 动作生成与 PPO log 概率估计

## 相关概念
- [[Flow Matching]]
- [[Diffusion Transformer]]
