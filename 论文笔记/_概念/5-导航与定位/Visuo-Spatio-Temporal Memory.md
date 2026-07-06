---
type: concept
aliases: [视觉-空间-时间记忆, 视觉空间时间记忆]
---

# Visuo-Spatio-Temporal Memory

## 定义
将每帧观测的视觉嵌入、机器人位姿、时间戳组成三元组存入向量数据库的记忆表示，支持按语义相似度、空间邻近、时间窗口灵活检索。

## 数学形式
$$\{m_i\}_N := \{(p_1, t_1, z_1, o_1), \ldots, (p_N, t_N, z_N, o_N)\}$$

## 核心要点
1. 直接存视觉嵌入 $z_i$ 而非图像转文字 caption，绕过有损的 caption 瓶颈
2. 位姿 $p_i$ 与时间戳 $t_i$ 提供空间和时间的 grounding
3. 借向量数据库实现 sub-$O(N)$ 检索，可扩展到超长轨迹

## 代表工作
- [[RAVEN]]: 提出该记忆结构作为长时程机器人问答与导航的核心

## 相关概念
- [[Multimodal Embedding]]
- [[Retrieval-Augmented Generation]]
- [[Agentic Memory]]
