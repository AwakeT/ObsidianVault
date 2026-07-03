---
type: concept
aliases: [UniACT, UniACT Dataset]
---

# UniACT-dataset

## 定义

UniACT-dataset 是 [[ABot-M0]] 构建的统一机器人操作预训练数据集，由六个公开 VLA 数据源清洗、标准化和重采样后组成。

## 核心要点

1. 规模约 600 万+ trajectories、9500+ 小时，覆盖 20+ robot embodiments。
2. 数据源包括 OXE、OXE-AugE、AgiBot-Beta、RoboCOIN、RoboMind 和 Galaxea。
3. 统一为 [[Delta Action]]、[[End-Effector Frame]]、[[Rotation Vector]] 和 [[Pad-to-Dual-Arm]] 格式。

## 代表工作

- [[ABot-M0]]: 构建 UniACT-dataset 并用于跨 embodiment VLA 预训练。

## 相关概念

- [[Vision-Language-Action]]
- [[Delta Action]]
- [[Pad-to-Dual-Arm]]
