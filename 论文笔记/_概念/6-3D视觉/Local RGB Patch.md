---
type: concept
aliases: [Local Appearance Patch, 局部RGB Patch]
---

# Local RGB Patch

## 定义
Local RGB Patch 是围绕 query 像素位置截取的小局部图像块，用作解码器的外观条件，帮助模型恢复细粒度视觉边界和局部对应。

## 核心要点
1. 可为纯坐标 query 补充低层纹理和边缘信息。
2. 在 D4RT 中默认使用 $9\times9$ patch。
3. 高分辨率输出时，patch 也需要从原始分辨率抽取才能保留高频细节。

## 代表工作
- [[D4RT]]: local RGB patch 显著改善深度边界、pose 和高分辨率细节。

## 相关概念
- [[Query-Based Decoder]]
- [[Video Depth Estimation]]
