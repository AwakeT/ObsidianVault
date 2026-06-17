---
type: concept
aliases: [VLN-CE, Vision-and-Language Navigation in Continuous Environments, 连续环境视觉语言导航]
---

# VLN-CE

## 定义
视觉-语言导航的连续环境版本：智能体在连续动作空间（而非离散导航图节点）中按自然语言指令移动，更贴近真实机器人导航。

## 核心要点
1. 常用 benchmark：R2R、RxR 的 Val-Unseen split。
2. 评测指标：NE（导航误差）、OS/OSR（Oracle Success）、SR（成功率）、SPL（路径加权成功率）、nDTW。
3. Qwen-VLA 用 sliding-window waypoint 动作与其轨迹预测对齐。

## 代表工作
- Krantz et al. 2020 / Ku et al. 2020 (R2R, RxR): 基础
- [[StreamVLN]]: 导航 SOTA 基线
- [[Qwen-VLA]]: R2R OS 69.0 / RxR SR 59.6

## 相关概念
- [[StreamVLN]]
- [[Vision-Language-Action]]
