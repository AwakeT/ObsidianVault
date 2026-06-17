---
type: concept
aliases: [Window Attention, 窗口注意力, 窗口化注意力]
---

# Windowed Attention

## 定义
将自注意力限制在固定数量的邻近 token（窗口）内计算，以降低注意力的二次复杂度，常与全局（非窗口）注意力交错使用以平衡精度与延迟。

## 核心要点
1. 减少计算量，但牺牲全局信息混合
2. [[RF-DETR]] 在 ViT backbone 中交错排列窗口块与非窗口块，窗口块放在连续 layer 以省去 reshape
3. 与 class token 不兼容时需为每个窗口复制 class token（RF-DETR 用 2 窗口最优）

## 代表工作
- [[RF-DETR]]: backbone 交错窗口/非窗口注意力
- [[LW-DETR]]: 窗口注意力放在 layer {0,1,3,6,7,9}

## 相关概念
- [[Self-Attention]]
- [[Vision Transformer]]
