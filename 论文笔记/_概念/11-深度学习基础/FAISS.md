---
type: concept
aliases: [FAISS, Faiss]
---

# FAISS

## 定义
Facebook AI 开发的高效相似度搜索与稠密向量聚类库，支持 GPU 加速的十亿级最近邻检索。

## 核心要点
1. 支持欧氏/余弦等距离度量的近似最近邻搜索
2. 优化到 sub-linear 复杂度，可扩展到大规模数据集
3. 常作为 RAG / 视觉记忆检索的后端

## 代表工作
- [[RAVEN]]: 作为视觉-空间-时间记忆的检索后端之一

## 相关概念
- [[Milvus]]
- [[Retrieval-Augmented Generation]]
