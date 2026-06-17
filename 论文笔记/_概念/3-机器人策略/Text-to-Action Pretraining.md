---
type: concept
aliases: [T2A, Text-to-Action Pretraining, 文本到动作预训练]
---

# Text-to-Action Pretraining

## 定义
VLA 训练的第一阶段：冻结 VLM 主干，只训练动作解码器，**仅用语言指令 + 具身提示而刻意不给图像**，让解码器学会作为"语言→动作解压器"，建立结构化动作先验。

## 核心要点
1. 动机源于"压缩视角"：语言指令 + 具身提示用少量 token 紧凑编码任务意图，而动作轨迹是高维稠密序列，桥接二者是结构化解压问题。
2. 屏蔽视觉可防止解码器走视觉捷径，迫使其建立可靠的语言→动作映射。
3. 消融关键结论：全序列预测 > chunk 预测；20% 合成 + 80% 真机最佳；T2A 用 Sigmoid-Normal timestep 分布；2,000 步为甜点（过长过拟合）。

## 代表工作
- [[Qwen-VLA]]: 阶段 I 关键创新，比无 T2A 基线提升 +10.2pp

## 相关概念
- [[Flow Matching]]
- [[Continued Pretraining]]
- [[Diffusion Transformer]]
