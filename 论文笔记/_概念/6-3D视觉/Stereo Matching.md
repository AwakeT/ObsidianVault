---
type: concept
aliases: [立体匹配, 双目匹配, 双目深度估计]
---

# Stereo Matching

## 定义
从校正后的左右立体图像对中估计逐像素视差（disparity）图的任务。视差与深度成反比：$Z = \frac{f \cdot B}{d}$。

## 数学形式
$$d(x,y) = x_L - x_R$$
$$Z = \frac{f \cdot B}{d}$$

## 核心要点
1. 经典流程：匹配代价计算 → 代价聚合 → 视差选择 → 后处理
2. 深度学习方法：特征提取 → cost volume 构建 → 3D 卷积/迭代精化 → 回归视差
3. 主要 benchmark：[[Middlebury]]、[[ETH3D]]、[[KITTI-2015]]、[[SceneFlow]]

## 代表工作
- [[SGBM]]: 经典半全局匹配方法
- [[RAFT-Stereo]]: 迭代精化范式
- [[CREStereo]]: 自适应关联 + 级联精化
- [[FoundationStereo]]: 零样本基础模型

## 相关概念
- [[Epipolar Geometry]]
- [[Correlation Volume]]
- [[Cost Aggregation]]
