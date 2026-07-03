---
type: concept
aliases: [TTT3R]
---

# TTT3R

## 定义
把 [[CUT3R]] 的循环状态更新重新解释为 **[[Test-Time Training]]** 的在线学习问题，用记忆状态与新观测的**对齐置信度**推导出闭式自适应学习率来门控状态更新的 training-free 干预，显著提升长序列长度泛化。

## 核心要点
1. 诊断：CUT3R 更新隐含常数学习率 $\beta_t=1.0$ + softmax 归一化 → 偏向新观测 → 灾难性遗忘。
2. 用 state query × observation key 的 [[Cross-Attention]] 对齐置信度作为逐-token 学习率（TokenLR 范式）。
3. training-free、plug-and-play、无额外参数/计算，保持 CUT3R 速度与显存。
4. 长序列位姿 2× 于 CUT3R；>1000 帧需可选 State Reset。

## 数学形式
$$
\beta_t = \sigma\!\left(\tfrac{1}{m}\sum_m \mathbf{Q}_{\mathbf{S}_{t-1}}\mathbf{K}^{\top}_{\mathbf{X}_t}\right),\qquad
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t\nabla(\mathbf{S}_{t-1},\mathbf{X}_t)
$$

## 代表工作
- [[TTT3R]]（本篇，ICLR 2026）：3D Reconstruction as Test-Time Training。

## 相关概念
- [[CUT3R]]
- [[Test-Time Training]]
- [[DeltaNet]]
- [[Linear Attention]]
