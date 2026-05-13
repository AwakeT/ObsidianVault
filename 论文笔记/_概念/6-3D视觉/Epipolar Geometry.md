---
type: concept
aliases: [极线几何, 对极几何]
---

# Epipolar Geometry

## 定义
描述两个视图之间几何关系的数学框架。在校正后的立体图像对中，对应点约束在同一水平线（极线）上，将 2D 搜索简化为 1D 搜索。

## 数学形式
$$\mathbf{x}'^T \mathbf{F} \mathbf{x} = 0$$

基础矩阵 $\mathbf{F}$ 约束左图点 $\mathbf{x}$ 与右图对应点 $\mathbf{x}'$ 的关系。

## 核心要点
1. 校正后的立体对中，极线为水平线，对应点仅在水平方向有偏移（视差）
2. 极线约束是立体匹配算法的基础假设
3. 实际相机校正不完美时，残余垂直偏移影响匹配精度

## 代表工作
- [[RAFT-Stereo]]: 利用极线约束将 4D 关联体积降为 3D
- [[CREStereo]]: AGCL 处理非理想极线校正

## 相关概念
- [[Correlation Volume]]
- [[Stereo Matching]]
