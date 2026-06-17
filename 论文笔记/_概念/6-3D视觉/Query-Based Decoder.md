---
type: concept
aliases: [Query Decoder, Pointwise Decoder, 查询式解码器]
---

# Query-Based Decoder

## 定义
Query-Based Decoder 是一种按需解码机制：模型先得到全局表示，再用显式 query 指定要读取的空间、时间或任务位置，并只输出该 query 对应的结果。

## 核心要点
1. query 可以是坐标、时间、参考帧、类别或任务条件。
2. 独立 query 便于稀疏监督、稀疏推理和大规模并行。
3. 在 D4RT 中，query 为 $(u,v,t_{src},t_{tgt},t_{cam})$，用于预测任意时空点的 3D 坐标。

## 代表工作
- [[D4RT]]: 用 pointwise query decoder 统一点云、深度、轨迹和相机参数。

## 相关概念
- [[Cross-Attention]]
- [[Global Scene Representation]]
- [[4D Reconstruction]]
