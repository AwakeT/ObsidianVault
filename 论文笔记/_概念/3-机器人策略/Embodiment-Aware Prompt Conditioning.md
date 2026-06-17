---
type: concept
aliases: [具身感知提示条件, Embodiment-Aware Prompt Conditioning, 具身提示]
---

# Embodiment-Aware Prompt Conditioning

## 定义
通过在每个训练/推理样本前置一段描述当前机器人平台、臂构型、控制约定、控制频率与预测时域的文本提示，作为唯一的平台特定接口，使单一模型服务多种机器人形态。

## 核心要点
1. 提示模板示例："The robot is {robot_tag} with {single/dual arms}... control frequency {FPS} Hz. Please predict the next {chunk_size} control actions..."
2. 配合统一动作表示（共享张量接口 + 掩码），同一套 DiT 参数处理不同控制模式/动作维度，**无需 per-embodiment 输出头**。
3. 部署时只需替换提示为目标物理平台描述即可零样本迁移。

## 代表工作
- [[Qwen-VLA]]: 跨形态统一的核心机制

## 相关概念
- [[Vision-Language-Action]]
- [[Action Chunking]]
- [[Quantile Normalization]]
