---
type: concept
aliases: [Map-free Relocalization, Map-free]
---

# Map-free Relocalization

## 定义
无需预建地图的相机重定位方法，直接从数据库图像回归相对位姿。

## 核心要点
1. 支持零样本泛化
2. 需要分别训练室内/室外模型
3. 精度不及有地图方法

## 代表工作
- [[Reloc3r]]: 不分场景类型的统一模型全面超越

## 相关概念
- [[Relative Pose Regression]]
- [[Visual Place Recognition]]
