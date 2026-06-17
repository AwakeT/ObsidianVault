---
type: concept
aliases: [全局场景表示, Latent Scene Representation]
---

# Global Scene Representation

## 定义
Global Scene Representation 是由 encoder 从整段观测中提取的全局潜表示，用于承载场景几何、外观、时间对应和相机运动等信息。

## 核心要点
1. 通常由 Transformer token 表示。
2. 下游任务通过 decoder 或 query 从该表示中读取所需信息。
3. 在动态场景中，需要同时编码空间结构和时间变化。

## 代表工作
- [[D4RT]]: 先将视频编码成 $F$，再对 $F$ 做任意点查询。

## 相关概念
- [[Scene Representation Transformer]]
- [[Vision Transformer]]
- [[Query-Based Decoder]]
