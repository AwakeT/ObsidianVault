---
type: concept
aliases: [MANO, hand Model with Articulated and Non-rigid defOrmations]
---

# MANO

## 定义
参数化人手模型，用低维姿态与形状参数描述铰接手部的关节运动与非刚性形变，广泛用于手部姿态估计与动作捕捉。

## 核心要点
1. 提供紧凑的手部姿态参数化（关节轴角 + 形状）。
2. 第一视角人体数据常以 MANO 表示手部，作为机器人操作的可扩展先验来源。
3. Qwen-VLA 中真人数据的末端动作即"EEF (from MANO)"。

## 代表工作
- Romero et al. 2017: 原始提出
- [[Qwen-VLA]]: 第一视角人体手部表示

## 相关概念
- [[Eigengrasp]]
- [[PCA]]
