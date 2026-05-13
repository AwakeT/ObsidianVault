---
type: concept
aliases: [关联体积, Cost Volume, 代价体积]
---

# Correlation Volume

## 定义
通过计算左右图（或参考图与源图）特征之间的相似度构建的多维张量，编码所有像素在各候选视差/位移下的匹配代价。

## 数学形式
$$\mathbf{C}_{ijk} = \sum_h \mathbf{f}_{ijh} \cdot \mathbf{g}_{ikh}$$

3D 立体关联体积：左图像素 $(i,j)$ 与右图同行像素 $(i,k)$ 的特征内积。

## 核心要点
1. 3D 关联体积（立体匹配）仅沿极线方向计算，维度 $H \times W \times D$
2. 4D 关联体积（光流）在全 2D 位移空间计算，维度 $H \times W \times H \times W$
3. 通过 average pooling 构建金字塔可捕获多尺度匹配信息
4. Group-wise correlation 将特征分组独立计算，保留更丰富的匹配信息

## 代表工作
- [[RAFT-Stereo]]: 3D correlation volume + 4 级金字塔
- [[CREStereo]]: 自适应局部关联（AGCL），可变形搜索窗口
- [[FoundationStereo]]: Hybrid cost volume（group-wise + concatenation）
- [[GwcNet]]: Group-wise correlation 的提出者

## 相关概念
- [[Cost Aggregation]]
- [[Soft-Argmin]]
- [[Epipolar Geometry]]
