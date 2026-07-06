---
type: concept
aliases: [L3MVN]
---

# L3MVN

## 定义
一种基于大语言模型的视觉目标导航方法：构建语义地图，用 RoBERTa 判断哪个前沿点最可能出现目标物体来引导导航。

## 核心要点
1. 用 RoBERTa（encoder-only）预测物体出现概率
2. 基于前沿探索选择最高概率方向
3. encoder-only 架构推理能力弱，难利用长篇自然语言习惯

## 代表工作
- [[UcON]]: 作为对比基线，无法从用户习惯获益

## 相关概念
- [[Object Navigation]]
- [[Frontier Exploration]]
