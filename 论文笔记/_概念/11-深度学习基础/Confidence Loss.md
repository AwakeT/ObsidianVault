---
type: concept
aliases: [置信度损失]
---

# Confidence Loss

## 定义
监督网络预测的置信度/不确定性估计的损失函数，使正确预测的置信度高、错误预测的置信度低。

## 数学形式
$$L^w(w) = \max(1-w, 0) \cdot \mathbb{1}[|d^{diff}| < C_1] + \max(w, 0) \cdot \mathbb{1}[|d^{diff}| > C_2]$$

## 核心要点
1. 正确预测推向高置信度，错误预测推向低置信度
2. 中间误差区域不施加损失（dead zone）
3. HITNet 用于 tile 假设选择和传播

## 代表工作
- [[HITNet]]: tile 更新置信度监督

## 相关概念
- [[Contrastive Loss]]
