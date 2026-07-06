---
type: concept
aliases: [SR, 成功率]
---

# Success Rate

## 定义
导航任务的核心评估指标：智能体在规定条件下成功到达目标的 episode 比例。通常以"停止时距目标实例 1m 以内"为成功判据。

## 核心要点
1. $\mathrm{SR} = \frac{1}{N}\sum_i \mathbb{1}[\text{episode } i \text{ success}]$
2. 不考虑路径效率，常与 [[SPL]]（按路径长度加权的成功分数）配合报告
3. 多模态导航中可按目标模态（object / language / image）分别统计
4. 越高越好

## 代表工作
- [[EvoMemNav]]: 在 GOAT-Bench / HM3D 上报告 SR/SPL
- [[GOAT-Bench]]: 多模态终身导航的 SR/SPL 评估协议
- [[RAVEN]]: 真机导航成功率评测
- [[UcON]]: 物体导航成功率评测

## 相关概念
- [[SPL]]
- [[Object Navigation]]
- [[GOAT-Bench]]
