---
type: concept
aliases: [PixelNav, Pixel-Guided Navigation]
---

# PixelNav

## 定义
桥接零样本物体导航与基础模型的像素引导导航方法：设计 prompt 让 LLM/VLM 从多张图像中选方向、并在 RGB 上标注目标相关像素，再由低层策略走到该像素描述的位置。

## 核心要点
1. 多模态 LLM 选导航方向 + 目标像素
2. 低层策略执行到像素位置
3. 有直接输入图像 / 输入物体标签两个变体

## 代表工作
- [[UcON]]: 作为真机唯一对比基线

## 相关概念
- [[Object Navigation]]
- [[LGX]]
