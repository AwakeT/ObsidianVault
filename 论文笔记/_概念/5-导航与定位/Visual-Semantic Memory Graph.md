---
type: concept
aliases: [VSMGraph, 视觉语义记忆图]
---

# Visual-Semantic Memory Graph

## 定义
一种以原始 RGB 视图为"一等记忆"的图像锚定记忆结构，用轻量语义线索与拓扑关系将记忆组织成 room-view-object 层级，保留细粒度视觉证据用于消歧与 Stop 验证。

## 数学形式
$$\mathcal{G}_t = (\mathcal{R}_t,\ \mathcal{V}_t,\ \mathcal{O}_t,\ \mathcal{E}_t), \quad \mathcal{V}_t = \mathcal{V}_{A,t} \cup \mathcal{V}_{F,t}$$

## 核心要点
1. **视图为一等公民**: 视图节点存储原始图像证据与位姿，物体节点仅作索引/分桶，不压缩记忆
2. **room-view-object 层级**: 房间节点支持区域级检索，视图节点分为 [[Anchor View|锚点视图]] 与 [[Frontier View|前沿视图]]，物体节点存实例假设
3. **两类边**: 可导航边（navigability，连续无碰撞视点）与可见性边（visibility，视图↔物体）
4. **轻量语义标注**: 每视图用 [[CLIP]] argmax 打离散房间标签，用于房间感知分桶
5. **决策 grounded 在多视图图像**: 区别于检测器中心 [[Scene Graph|场景图]]（信息压缩）与稠密 3D 重建（计算昂贵）

## 代表工作
- [[EvoMemNav]]: 提出 VSMGraph 并配合粗到精策略与 RDCMA

## 相关概念
- [[Topological Map]]
- [[Scene Graph]]
- [[Anchor View]]
- [[Frontier View]]
- [[Occupancy Grid]]
