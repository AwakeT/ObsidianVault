---
type: concept
aliases: [Correlation Map, 相关性图, Cost Volume]
---

# Correlation Map

## 定义
两个特征图之间逐像素计算的相似度矩阵，用于匹配、光流和追踪等密集对应任务。

## 数学形式

$$$C(i,j) = F_1(i)^T F_2(j)$$$

## 核心要点
1. 光流和立体匹配的核心组件
2. 4D cost volume: H×W×H×W
3. VGGT 追踪模块中用于计算查询点与目标帧的对应

## 代表工作
- [[VGGT]]: 追踪模块中计算特征相关性

## 相关概念
- [[Optical Flow]]
- [[Point Tracking]]
