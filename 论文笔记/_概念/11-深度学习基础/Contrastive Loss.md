---
type: concept
aliases: [对比损失]
---

# Contrastive Loss

## 定义
使正样本对的距离减小、负样本对的距离增大的损失函数，常用于度量学习和匹配任务。

## 数学形式
$$L = \psi(d^{gt}) + \max(\beta - \psi(d^{nm}), 0)$$

## 核心要点
1. 正样本对拉近，负样本对推远至 margin $\beta$ 之外
2. HITNet 初始化阶段用对比损失训练匹配代价
3. 与 triplet loss、InfoNCE 等属于同一家族

## 代表工作
- [[HITNet]]: 初始化阶段的匹配代价监督

## 相关概念
- [[L1 Loss]]
