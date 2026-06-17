---
type: concept
aliases: [粗到精, 粗到精策略, Coarse-to-Fine Policy]
---

# Coarse-to-Fine

## 定义
（在零样本导航语境下）一种预算化两阶段决策过程：粗阶段压缩候选空间并路由，精阶段仅对小候选集调用昂贵推理（如 VLM）做选择与验证，从而将计算预算集中在少量候选上。

## 数学形式
$$(a_t, \tau_t) = \mathrm{VLM}(g, C_t), \quad C_t = C_t^A \cup C_t^F, \quad |C_t^A| \le K_A,\ |C_t^F| \le K_F$$

## 核心要点
1. **粗阶段（Explore）**: 房间感知分桶 + Top-K 压缩候选，路由到前沿（探索）或锚点（证据）；候选为空时无需 VLM 直接前往前沿
2. **精阶段（Search + Verify）**: 仅对紧凑集 $C_t$ 查询 VLM 选择，并对到达的锚点视图做结构化 Stop 验证（STOP / RESELECT）
3. **置信门控**: 低置信度（uncertain/unknown）时回退到探索，减少同类错误实例
4. **效率收益**: 显著降低 VLM 调用次数与运行时，同时提升 SR

## 代表工作
- [[EvoMemNav]]: 在 VSMGraph 上实现预算化粗到精导航策略
- [[3D-Mem]]: 对比的多视图记忆方法（VLM 调用更多）

## 相关概念
- [[Visual-Semantic Memory Graph]]
- [[VLM]]
- [[Frontier Exploration]]
