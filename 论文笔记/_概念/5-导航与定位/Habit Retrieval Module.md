---
type: concept
aliases: [HRM, 习惯检索模块]
---

# Habit Retrieval Module

## 定义
受 RAG 启发的模块：用 BGE-M3 计算查询模板与用户习惯知识库中所有习惯的相似度，取 top-p 最相关习惯注入 prompt，缓解 LLM 长上下文推理退化。

## 数学形式
查询模板："Please retrieve the habit about {TargetObject}."，取 top-$p$ 相似习惯。

## 核心要点
1. 选择性检索目标相关习惯，避免整库长 prompt
2. 检索到的间接相关习惯提供地标线索，常优于 GT 习惯
3. 机制简单，核心贡献是证明"相关习惯增强 prompt"有效

## 代表工作
- [[UcON]]: 提出 HRM 提升基于 LLM 的习惯导航

## 相关概念
- [[Retrieval-Augmented Generation]]
- [[BGE-M3]]
- [[User Habit Knowledge Base]]
