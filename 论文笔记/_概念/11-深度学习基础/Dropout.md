---
type: concept
aliases: [随机失活]
---

# Dropout

## 定义
Srivastava et al. (2014) 提出的正则化方法，训练时随机置零部分神经元，等价于训练大量共享权重子网络的集成。

## 核心要点
1. 防止过拟合，提升泛化
2. 可视为对子网络做集成学习
3. [[RF-DETR]] 的权重共享 NAS（每次随机采样配置）类比 dropout 的集成效应，起"架构增强"正则作用

## 代表工作
- [[RF-DETR]]: 权重共享 NAS 类比 dropout 集成

## 相关概念
- [[正则化]]
- [[Neural Architecture Search]]
