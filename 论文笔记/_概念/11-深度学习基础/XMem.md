---
type: concept
aliases: [XMem]
---

# XMem

## 定义
一种受 Atkinson–Shiffrin 记忆模型启发的**长时视频对象分割**方法，用**多级记忆**（感官 / 工作 / 长期记忆）管理海量历史特征：工作记忆保留近期稠密特征，达到容量时按重要性稀疏化并合并进长期记忆，从而在超长视频上保持稳定分割。

## 核心要点
1. 分层记忆：sensory / working / long-term，逐级稀疏化。
2. 按累积注意力权重（使用频率）决定哪些 token 进入长期记忆。
3. 用定长/稀疏记忆避免显存随视频长度爆炸。

## 代表工作
- XMem（Cheng & Schwing, ECCV 2022）：长时视频对象分割。
- [[Spann3R]]：借鉴 XMem 的长期记忆稀疏化管理 [[Spatial Memory]]。

## 相关概念
- [[Memory Mechanism]]
- [[Spatial Memory]]
- [[Hierarchical Memory]]
