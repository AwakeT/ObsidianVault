---
type: concept
aliases: [Success weighted by Path Length]
---

# SPL

## 定义
VLN/ObjectNav 中的标准评估指标，在 SR 基础上惩罚多余路径。

## 数学形式
$$\text{SPL} = \frac{1}{N}\sum_{i=1}^{N} S_i \cdot \frac{l_i}{\max(p_i, l_i)}$$

## 核心要点
1. SR 只看成功率，SPL 同时考虑路径效率
2. VLN-NF 扩展为 REV-SPL

## 代表工作
- [[MetaNav]]: 主要评估指标
- [[VLN-NF]]: 扩展为 REV-SPL
- [[OVAL]]: 主要评估指标
- [[UcON]]: SR/SPL 评测

## 相关概念
- [[Object Navigation]]
