---
type: concept
aliases: [可变形卷积, DCN, Deformable Conv]
---

# Deformable Convolution

## 定义
通过学习采样位置的额外偏移量使卷积核的感受野自适应变形的卷积操作。

## 数学形式
$$y(p) = \sum_{k} w_k \cdot x(p + p_k + \Delta p_k)$$

其中 $\Delta p_k$ 为学习到的偏移量。

## 核心要点
1. 常规卷积的采样位置固定在规则网格上，可变形卷积的采样位置可自适应
2. 偏移量由额外的卷积层从输入特征预测
3. 通过双线性插值实现对非整数位置的采样

## 代表工作
- [[CREStereo]]: AGCL 中的可变形搜索窗口

## 相关概念
- [[Dilated Convolution]]
- [[Correlation Volume]]
