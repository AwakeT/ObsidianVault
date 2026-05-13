---
type: concept
aliases: [L1 损失, MAE Loss, 平均绝对误差]
---

# L1 Loss

## 定义
预测值与真值之间绝对差的均值，对异常值比 L2 Loss 更鲁棒。

## 数学形式
$$\mathcal{L}_{L1} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

## 核心要点
1. 梯度恒定（不随误差大小变化），对大误差不过度惩罚
2. 在零点不可微，实践中用 smooth L1 替代
3. RAFT-Stereo、CREStereo 均使用带指数加权的 L1 损失

## 代表工作
- [[RAFT-Stereo]]: $\gamma^{N-i}$ 加权的序列 L1 损失
- [[CREStereo]]: 级联加权 L1 损失

## 相关概念
- [[Smooth L1 Loss]]
- [[Robust Loss]]
