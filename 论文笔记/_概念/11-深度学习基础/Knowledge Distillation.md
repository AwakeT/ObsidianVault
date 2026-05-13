---
type: concept
aliases: [知识蒸馏, KD]
---

# Knowledge Distillation

## 定义
将大模型（教师）的知识迁移到小模型（学生）的训练策略，学生模型学习教师的软标签或中间表示。

## 核心要点
1. 软标签包含类别间关系的暗知识（dark knowledge）
2. 可用于模型压缩、域适配
3. MobileStereoNet 的潜在改进方向之一

## 代表工作
- [[MobileStereoNet]]: 论文建议的改进方向

## 相关概念
- [[Depthwise Separable Convolution]]
