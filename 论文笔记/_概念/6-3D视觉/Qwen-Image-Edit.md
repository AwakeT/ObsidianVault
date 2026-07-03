---
type: concept
aliases: [Qwen Image Edit]
---

# Qwen-Image-Edit

## 定义

Qwen-Image-Edit 是图像编辑/生成模型，在 ABot-M0 中被用来从原始视角合成额外视角，提供隐式多视图 3D 信息。

## 核心要点

1. 可为 VLA 策略提供 viewpoint variation 下的额外视觉线索。
2. 在 [[ABot-M0]] 中与 VLM final-layer feature 融合后进入 action expert。
3. 对 LIBERO-Plus 的 camera perturbation 子集提升明显。

## 代表工作

- [[ABot-M0]]: 使用 Qwen-Image-Edit 合成 1 view / 2 views 并测试 3D information injection。

## 相关概念

- [[VGGT]]
- [[3D Reconstruction]]
- [[Vision-Language-Action]]
