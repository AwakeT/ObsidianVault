---
type: concept
aliases: [光线图, Ray Map, Raymap]
---

# Raymap

## 定义
一种把相机（内参 + 外参）编码为逐像素图像的表示：每个像素用 6 个通道分别编码该像素对应光线的**起点（ray origin）** 与**方向（ray direction）**，即一张 $H\times W\times 6$ 的图像。

## 数学形式
每像素 $p$：$\text{raymap}(p) = [\mathbf{o}_p,\ \mathbf{d}_p] \in \mathbb{R}^6$，其中 $\mathbf{o}_p$ 为光线起点（相机中心），$\mathbf{d}_p$ 为经内外参决定的光线方向。

## 核心要点
1. 用统一的图像形式表达任意相机位姿与内参，便于与视觉 token 一起送入 transformer。
2. 可作为**虚拟相机查询**：在未观测视角构造 raymap，查询模型状态以"幻想"出未见区域的几何与颜色。
3. 与像素对齐，天然适配 pointmap / DPT 类稠密预测头。

## 代表工作
- [[CUT3R]]：用 raymap 查询持久状态，推断未观测区域结构并解码颜色。

## 相关概念
- [[Pinhole Camera Model]]
- [[Point Map]]
- [[Camera Pose Estimation]]
