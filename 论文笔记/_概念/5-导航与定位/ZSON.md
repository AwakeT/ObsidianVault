---
type: concept
aliases: [ZSON, Zero-Shot Object-Goal Navigation]
---

# ZSON

## 定义
开词表、端到端的零样本物体目标导航方法：用 CLIP 编码目标物体、ResNet 编码观测，喂给策略网络预测动作。

## 核心要点
1. 用多模态目标嵌入实现零样本开词表导航
2. 端到端策略网络直接出动作
3. 习惯驱动放置下同样失配

## 代表工作
- [[UcON]]: 作为对比基线，SR 仅 1.0%

## 相关概念
- [[Object Navigation]]
- [[CLIP]]
- [[VTN]]
