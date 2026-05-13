---
type: concept
aliases: [鲁棒损失, 截断损失]
---

# Robust Loss

## 定义
对异常值不敏感的损失函数族，通过截断或缓增长抑制大误差的影响。

## 核心要点
1. Huber Loss：小误差 L2、大误差 L1
2. 截断损失：超过阈值后梯度为零
3. Generalized robust loss (Barron 2019) 统一多种鲁棒损失

## 代表工作
- [[HITNet]]: 传播阶段使用截断鲁棒损失

## 相关概念
- [[L1 Loss]]
- [[Smooth L1 Loss]]
