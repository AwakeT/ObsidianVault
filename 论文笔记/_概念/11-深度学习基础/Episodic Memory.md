---
type: concept
aliases: [情景记忆]
---

# Episodic Memory

## 定义
存储 agent 在特定时间、地点和上下文中亲历事件的外部记忆结构，可包含位置、动作、感知、对话、任务结果和证据引用，并支持跨会话检索、摘要与选择性遗忘。

## 数学形式
N/A

## 核心要点
1. 保留“何时、何地、发生了什么”的经历证据，而非只保存一般语义知识
2. 可按 scene / event / goal / episode 等粒度组织并压缩为长期摘要
3. 长期部署需要显式管理写入、检索、巩固、衰减、遗忘与审计
4. 认知科学中源于 Tulving 的情景记忆理论

## 代表工作
- [[MetaNav]]: 滑动窗口+长期摘要双层结构
- [[H2-EMV]]: 连续多模态经历的在线层级历史树，并通过用户反馈学习选择性遗忘规则

## 相关概念
- [[Hierarchical Memory]]
- [[Selective Forgetting]]
- [[Episodic Memory Verbalization]]
- [[元认知]]
