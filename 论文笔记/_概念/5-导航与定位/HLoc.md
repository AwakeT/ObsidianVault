---
type: concept
aliases: [HLoc, Hierarchical Localization]
---

# HLoc

## 定义
分层视觉定位框架，先通过图像检索缩小搜索范围，再通过特征匹配和 PnP 精确定位。

## 核心要点
1. 结合 SuperPoint+SuperGlue 等现代特征
2. 在大规模场景上精度最高
3. 推理时间较长（~700ms）

## 代表工作
- [[Reloc3r]]: 在大场景上精度不及 HLoc 但速度快 3 倍

## 相关概念
- [[Visual Place Recognition]]
- [[PnP]]
