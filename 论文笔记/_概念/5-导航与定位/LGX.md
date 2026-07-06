---
type: concept
aliases: [LGX]
---

# LGX

## 定义
LLM-based 零样本物体导航方法：从提取的物体标签或场景描述生成 prompt，用 LLM 的常识知识选择无障碍方向，再用视觉语言模型做物体检测。

## 核心要点
1. LLM 推理目标物体可能方向
2. 依赖常识先验，习惯驱动放置下失配
3. 更强 LLM（GPT-4）更善用检索到的习惯

## 代表工作
- [[UcON]]: 作为对比基线，GPT-4 版在习惯检索下 SR 最高

## 相关概念
- [[Object Navigation]]
- [[PixelNav]]
