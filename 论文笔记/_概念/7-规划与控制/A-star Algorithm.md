---
type: concept
aliases: [A*, A-star, A星算法]
---

# A-star Algorithm

## 定义
经典启发式最短路径搜索算法，用代价函数 $f(n)=g(n)+h(n)$（已走代价 + 启发式估计）引导在图上寻找最小代价路径。

## 数学形式
$$f(n) = g(n) + h(n)$$

## 核心要点
1. 启发式 $h(n)$ 可采（admissible）时保证最优
2. 常用于占据栅格地图上的机器人路径规划
3. 配合 PD 控制器跟踪 waypoints 执行导航

## 代表工作
- [[RAVEN]]: 用 A* 轨迹规划器执行检索到目标位置的导航

## 相关概念
- [[Frontier Exploration]]
- [[Occupancy Grid]]
