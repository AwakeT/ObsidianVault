---
type: concept
aliases: [Huber Loss, 平滑 L1 损失]
---

# Smooth L1 Loss

## 定义
结合 L1 和 L2 损失的优点：小误差区域使用 L2（平滑），大误差区域使用 L1（鲁棒）。

## 数学形式
$$\text{SmoothL1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

## 核心要点
1. 在零点处可微（L1 不可微）
2. 对异常值鲁棒（L2 会过度惩罚）
3. MobileStereoNet、FoundationStereo 初始视差监督使用

## 代表工作
- [[MobileStereoNet]]: 训练损失函数
- [[FoundationStereo]]: 初始视差的 smooth L1 监督

## 相关概念
- [[L1 Loss]]
- [[Robust Loss]]
