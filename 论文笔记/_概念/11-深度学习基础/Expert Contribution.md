---
type: concept
aliases: [专家贡献度, Expert Importance]
---

# Expert Contribution

## 定义
Expert Contribution 是衡量某个专家在当前任务、样本或训练阶段中被使用和产生影响程度的指标。

## 数学形式
$$
c_i = \frac{\sum_{t} w_{t,i}}{\sum_{j=1}^{N}\sum_{t} w_{t,j}}
$$

其中 $w_{t,i}$ 表示第 $t$ 个输入上第 $i$ 个专家的路由权重。

## 核心要点
1. 可由路由概率、梯度贡献、激活频率或损失敏感度估计。
2. 在持续学习中可用来区分当前任务关键专家和应保守保护的专家。
3. 贡献度统计通常与 Top-K 选择或动量更新结合使用。

## 代表工作
- [[M3E]]: 用当前任务累计融合路由权重计算 expert contribution。

## 相关概念
- [[Top-K Routing]]
- [[Dynamic MoE Momentum Update]]
- [[Mixture-of-Experts]]
