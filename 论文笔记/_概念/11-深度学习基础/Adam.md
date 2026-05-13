---
type: concept
aliases: [Adam 优化器]
---

# Adam

## 定义
结合一阶矩（动量）和二阶矩（RMSProp）的自适应学习率优化器。

## 数学形式
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

## 核心要点
1. 默认参数 $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$
2. 自适应学习率，每个参数有独立的学习率
3. CREStereo 使用 Adam，学习率 0.0004

## 代表工作
- [[CREStereo]]: Adam lr=0.0004

## 相关概念
- [[AdamW]]
