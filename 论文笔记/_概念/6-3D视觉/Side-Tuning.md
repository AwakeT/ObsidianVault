---
type: concept
aliases: [侧接微调, Side Network]
---

# Side-Tuning

## 定义
通过在冻结的预训练网络旁附加一个可训练的侧网络，并将两者输出融合，实现对基础模型的高效适配。

## 核心要点
1. 冻结主网络保留预训练知识，侧网络学习任务特定信息
2. 比全量微调更节省参数，比 LoRA 等更灵活
3. 融合方式可以是拼接、加法或注意力机制

## 代表工作
- [[FoundationStereo]]: STA 用 EdgeNeXt-S 侧接冻结的 DepthAnythingV2

## 相关概念
- [[DepthAnythingV2]]
- [[EdgeNeXt]]
