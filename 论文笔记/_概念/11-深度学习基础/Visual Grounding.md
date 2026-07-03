---
type: concept
aliases: [视觉定位, 视觉grounding, Referring Expression Comprehension, REC]
---

# Visual Grounding

## 定义
根据自然语言描述（指代表达）在图像中定位对应区域（边界框或点）的任务，要求将语言意图与具体空间区域对齐。

## 核心要点
1. **任务形式**: 输入图像 + 文本查询，输出匹配区域的边界框/点。
2. **子任务**: 指代表达理解（REC）、短语 grounding（Phrase Grounding）、GUI grounding、文本 grounding 等。
3. **VLM 范式**: 把坐标作为 token 序列生成（Textual Digits / Quantized Tokens），存在串行解码瓶颈。
4. **评测**: RefCOCO/RefCOCOg、HumanRef 等，常用 F1@IoU 度量。

## 代表工作
- [[LocateAnything]]: 用 [[Parallel Box Decoding]] 统一 grounding 与检测。
- [[Grounding DINO]]: 开集专用 grounding 检测器。
- [[Rex-Omni]]: VLM 统一 grounding/detection。

## 相关概念
- [[Object Detection]]
- [[RefCOCO]]
- [[Bounding Box]]
