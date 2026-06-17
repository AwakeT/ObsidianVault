---
type: concept
aliases: [流匹配, Flow Matching, Conditional Flow Matching, CFM]
---

# Flow Matching

## 定义
一种生成式建模方法，通过学习一个时间相关的速度场，将简单噪声分布连续地"流"到目标数据分布；训练时直接回归条件速度场，无需模拟 ODE。

## 数学形式
线性插值路径 $Y^\tau = (1-\tau)Y^0 + \tau Y^1$（$Y^0$ 为干净目标，$Y^1\sim\mathcal{N}(0,I)$），模型 $v_\theta$ 回归条件速度 $(Y^1-Y^0)$：

$$
\mathcal{L} = \mathbb{E}_{\tau, Y^0, Y^1}\left[\left\| v_\theta(Y^\tau, \tau) - (Y^1 - Y^0)\right\|^2\right]
$$

推理时用少量 [[Euler Integration|欧拉积分]]步从 $\tau=1$ 积分到 $\tau=0$。

## 核心要点
1. 相比 diffusion 的得分匹配，流匹配直接监督速度场，训练目标更简洁。
2. 推理只需少量积分步，低延迟，适合实时机器人控制。
3. timestep 采样分布 $p(\tau)$ 影响梯度集中位置（Beta 偏向干净端，Sigmoid-Normal 偏向中间噪声）。

## 代表工作
- [[Qwen-VLA]]: 动作专家用流匹配生成连续动作块
- [[Pi0]] / [[Pi05]]: 流匹配 VLA 策略
- [[Diffusion Policy]]: 扩散式动作生成的前身

## 相关概念
- [[Diffusion Transformer]]
- [[Euler Integration]]
- [[Diffusion Policy]]
