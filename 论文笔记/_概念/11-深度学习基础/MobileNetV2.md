---
type: concept
aliases: [MobileNet V2]
---

# MobileNetV2

## 定义
Sandler et al. 2018 提出的改进版轻量 CNN，核心是倒残差块（Inverted Residual Block）。

## 核心要点
1. 倒残差块：低维输入 → 高维展开 → depthwise → 低维投影 + 残差
2. 相比 MobileNetV1 增加了残差连接和 expansion
3. MobileStereoNet 的 $v_2$ block 即基于此

## 代表工作
- [[MobileStereoNet]]: $v_2$ block 的来源

## 相关概念
- [[Inverted Residual Block]]
- [[MobileNetV1]]
