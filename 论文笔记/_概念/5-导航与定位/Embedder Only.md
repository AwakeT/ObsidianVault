---
type: concept
aliases: [纯嵌入检索基线]
---

# Embedder Only

## 定义
物体检索/导航中的一种基线：用多模态嵌入模型编码图像，返回与用户查询嵌入相似度最高的图像，全程不使用语言模型。

## 核心要点
1. 无 LLM/VLM 参与，纯向量最近邻检索
2. 计算轻量，适合离线无网、端侧部署
3. 中等规模开源 VLM 下反而可能优于含 VLM 的方法

## 代表工作
- [[RAVEN]]: 作为对比基线；发现开源 VLM 较弱时 embedder-only 更可取

## 相关概念
- [[Multimodal Embedding]]
- [[VLM Only]]
