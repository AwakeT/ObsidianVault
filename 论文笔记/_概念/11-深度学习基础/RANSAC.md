---
type: concept
aliases: [RANSAC, Random Sample Consensus]
---

# RANSAC

## 定义
从含有大量离群值的数据中鲁棒估计模型参数的迭代算法。

## 核心要点
1. 随机采样最小点集拟合模型
2. 计算 inlier 数量评估模型质量
3. 广泛用于位姿估计、匹配过滤

## 代表工作
- [[MonST3R]]: RANSAC+PnP 从 pointmap 恢复相对位姿

## 相关概念
- [[PnP]]
- [[Bundle Adjustment]]
