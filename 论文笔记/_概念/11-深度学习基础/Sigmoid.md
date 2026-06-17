---
type: concept
aliases: [Sigmoid 函数, Logistic Function]
---

# Sigmoid

## 定义
$\sigma(x)=1/(1+e^{-x})$，将实数映射到 (0,1)，常用作二分类概率或多标签置信度。

## 核心要点
1. 输出可解释为置信度/概率
2. 多标签检测中对每类 logit 独立做 sigmoid
3. [[RF-DETR]] 用 query token 最大类别 logit 的 sigmoid 值排序做 [[Query Selection|query 丢弃]]

## 数学形式
$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

## 代表工作
- [[RF-DETR]]: query 置信度排序

## 相关概念
- [[Query Selection]]
