---
type: concept
aliases: [SFT, Supervised Fine-Tuning, 监督微调]
---

# Supervised Fine-Tuning

## 定义
在预训练模型上用高质量标注/示范数据做监督微调，使模型对齐目标任务分布。

## 核心要点
1. 在 Qwen-VLA 中为阶段 III，从 CPT checkpoint 分两路：多任务 SFT（VQA/空间 grounding/操作/导航，形态与任务平衡采样）与真机 ALOHA SFT。
2. 联合优化 VL 下一 token 损失（权重 0.1）与动作流匹配损失（权重 1.0）。
3. SFT 优化的是示范似然，无法直接优化闭环成功，故后接 [[Reinforcement Learning|RL]]。

## 代表工作
- [[Qwen-VLA]]: 阶段 III 任务专精

## 相关概念
- [[Continued Pretraining]]
- [[Reinforcement Learning]]
- [[Next-Token Prediction]]
