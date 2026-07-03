---
type: concept
aliases: [PBD, 并行框解码]
---

# Parallel Box Decoding (PBD)

## 定义
一种将边界框作为不可分割的原子单元、在单个并行前向步内一次性预测完整坐标集 $(x_1,y_1,x_2,y_2)$ 的解码范式，用于 VLM 的视觉检测与 grounding，取代逐 token 自回归生成。

## 数学形式
块级半自回归联合概率（块内并行、块间因果）：

$$
P(\mathbf{B} \mid \mathcal{Z}, \mathcal{E}) = \prod_{i=1}^{N} P\big(b_i \mid b_{<i}, \mathcal{Z}, \mathcal{E}\big)
$$

## 核心要点
1. **box-aligned（框对齐）**: 把 [[Multi-Token Prediction|MTP]] 的预测块与结构化几何单元对齐，避免对坐标 token 任意分块。
2. **块内双向、块间因果**: 块内 token 共享双向注意力捕获坐标几何依赖，块间严格因果保留自回归推理能力。
3. **双形式化训练**: NTP 流（token-level）+ block-level MTP 流联合训练，保留因果推理同时解锁并行。
4. **速度-精度双赢**: 相比 [[Next-Token Prediction|NTP]] 的逐 token 串行，吞吐提升约 2.5×，同时高 IoU 定位质量更优。

## 代表工作
- [[LocateAnything]]: 首次将 PBD 应用到 VLM 检测/grounding，配合 Hybrid NTP fallback。

## 相关概念
- [[Multi-Token Prediction]]
- [[Next-Token Prediction]]
- [[Bounding Box]]
- [[Visual Grounding]]
