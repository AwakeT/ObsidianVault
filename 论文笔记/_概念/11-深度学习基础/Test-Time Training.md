---
type: concept
aliases: [测试时训练, TTT, Test-Time Training]
---

# Test-Time Training

## 定义
一类在**推理阶段**继续用当前输入数据更新模型部分参数的范式：把序列模型的隐状态（state）视为在测试时从 in-context tokens 学到的 **fast weight**，通过（显式或隐式的）梯度下降在线更新，而模型主参数（slow weights）冻结，充当 meta-learner。

## 数学形式
$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t\,\nabla_{\mathbf{S}}\,\ell(\mathbf{S}_{t-1}, \mathbf{X}_t)
$$
其中 $\mathbf{S}$ 是 fast weight（state），$\beta_t$ 是学习率，$\ell$ 是自监督/关联记忆重建目标。

## 核心要点
1. 把 RNN / [[Linear Attention]] 的状态更新统一看作在线学习：state 是在测试时优化的快权重。
2. 学习率 $\beta_t$ 的建模是关键：ScalarLR（标量）、ConditionLR（输入相关标量）、TokenLR（逐-token）。
3. 提供对 **state overfitting / forgetting** 的原理性解释，指导长度泛化。
4. 与 [[DeltaNet]]（linear TTT）等价：最小化 KV 重建误差得到解析梯度更新。

## 代表工作
- TTT（Sun et al.）：Learning to (Learn at Test Time)。
- [[TTT3R]]：把 3D 重建的循环状态更新解释为 TTT，用对齐置信度当闭式学习率。

## 相关概念
- [[DeltaNet]]
- [[Linear Attention]]
- [[Memory Mechanism]]
- [[Gated Linear Attention]]
