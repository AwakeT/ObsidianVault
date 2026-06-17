---
type: concept
aliases: [占用栅格, occupancy map, 占据栅格地图]
---

# Occupancy Grid

## 定义
将环境离散为规则网格、每格标记为自由/占用/未知的二维（或多维）度量地图，是机器人导航中表示可通行空间与边界的经典度量表示。

## 核心要点
1. 每个格单元状态通常为 {free, occupied, unknown}
2. 自由格邻接未知格即构成 [[Frontier Exploration|frontier]]（探索边界）
3. 常作为度量骨架：最短路径在栅格上计算，再沿拓扑/视图边执行
4. 与 [[Topological Map|拓扑图]] 结合时，栅格负责度量层、图负责高层推理与检索

## 代表工作
- [[EvoMemNav]]: 占用栅格作为 VSMGraph 的度量骨架，前沿视图在栅格边界生成
- [[VLFM]]: 基于占用/价值地图的前沿选择

## 相关概念
- [[Frontier Exploration]]
- [[Topological Map]]
- [[Frontier View]]
