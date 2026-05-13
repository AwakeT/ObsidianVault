---
type: concept
aliases: [软最小值, Differentiable Argmin]
---

# Soft-Argmin

## 定义
通过对 cost volume 沿视差维度做 softmax 加权求和实现可微的亚像素视差回归。

## 数学形式
$$d = \sum_{d=0}^{D-1} d \cdot \text{Softmax}(-C(d))$$

## 核心要点
1. 替代不可微的 argmin/WTA 操作，支持端到端训练
2. 输出连续值，天然支持亚像素精度
3. 在多模态分布时可能回归到错误的中间值（已知缺陷）

## 代表工作
- [[GCNet]]: 首次在立体匹配中使用 soft-argmin
- [[FoundationStereo]]: 初始视差预测

## 相关概念
- [[Correlation Volume]]
- [[Winner-Takes-All]]
