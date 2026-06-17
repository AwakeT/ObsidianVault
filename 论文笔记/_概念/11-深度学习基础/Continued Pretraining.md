---
type: concept
aliases: [CPT, Continued Pretraining, 继续预训练, 持续预训练]
---

# Continued Pretraining

## 定义
在已有预训练模型基础上，用领域/任务相关的大规模数据继续预训练，使模型适配新模态或新分布。

## 核心要点
1. 在 Qwen-VLA 中为阶段 II：解冻 VLM 主干与 [[Diffusion Transformer|DiT]] 动作专家，把 [[Text-to-Action Pretraining|T2A]] 建立的动作先验 grounding 到视觉观测。
2. 刻意混合仿真与真机轨迹，使 checkpoint 同时见过两域，便于后续 SFT/RL 专精任一域。
3. 暴露模型于多样视觉分布，使主干视觉表示不依赖特定渲染器，支撑零样本 sim-to-real。

## 代表工作
- [[Qwen-VLA]]: 阶段 II 视觉 grounding

## 相关概念
- [[Text-to-Action Pretraining]]
- [[Supervised Fine-Tuning]]
- [[Vision-Language-Action]]
