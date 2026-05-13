---
type: concept
aliases: [MobileNet V1]
---

# MobileNetV1

## 定义
Howard et al. 2017 提出的轻量级 CNN，核心是用深度可分离卷积替代标准卷积，大幅降低计算量。

## 核心要点
1. 深度可分离卷积 = depthwise + pointwise
2. 计算量降低约 $k^2$ 倍
3. MobileStereoNet 的 $v_1$ block 即基于此

## 代表工作
- [[MobileStereoNet]]: $v_1$ block 的来源

## 相关概念
- [[Depthwise Separable Convolution]]
- [[MobileNetV2]]
