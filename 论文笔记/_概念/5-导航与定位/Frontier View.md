---
type: concept
aliases: [前沿视图, frontier view node]
---

# Frontier View

## 定义
在视觉记忆图中表示"可达探索边界"的视图节点，用于驱动探索；与[[Anchor View|锚点视图]]互补。

## 数学形式
$$F_f = (r_f,\ p_f,\ I_f^{front},\ f^{room})$$

## 核心要点
1. 在自顶向下[[Occupancy Grid|占用栅格]]上，自由格邻接至少一个未知格即为前沿格，聚类成前沿区域
2. 对每个区域选可达入口位姿 $p_f$ 并拍摄前沿朝向图像 $I_f^{front}$
3. 每个前沿视图连接到最近的已探索视图（及其房间节点），未探索区域作为边界节点出现在图外围
4. 是 [[Frontier Exploration|前沿探索]] 思想在视觉记忆图上的结构化实现

## 代表工作
- [[EvoMemNav]]: VSMGraph 中前沿视图用于粗阶段探索路由

## 相关概念
- [[Visual-Semantic Memory Graph]]
- [[Anchor View]]
- [[Frontier Exploration]]
- [[Occupancy Grid]]
