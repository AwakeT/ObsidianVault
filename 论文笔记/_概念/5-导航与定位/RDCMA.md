---
type: concept
aliases: [Reflection-Driven Continual Memory Adaptation, 反思驱动持续记忆适应]
---

# RDCMA

## 定义
Reflection-Driven Continual Memory Adaptation：一种免训练机制，将导航交互结果反思后回写为目标条件、图附着的轻量先验，使记忆图成为逐步自适应的决策基底，无需更新模型参数。

## 核心要点
1. **两个时间尺度**: [[Episodic Memory|Episode-STM]]（episode 内短期抑制/防回环，结束重置）与 Scene-LTM（按场景持久化，跨子任务复用，保守更新）
2. **图附着先验**: 关联到房间/锚点/前沿节点，存为紧凑统计量（支持计数、可靠性/风险估计），由目标签名 $s_g$ 索引
3. **反思来源**: 在 Stop 验证时调用（STOP vs RESELECT），子任务后聚合轨迹事件与验证结果回写
4. **两个使用通道**: 探索排序先验（Explore 阶段 tie-breaker）+ 停止校准先验（Verify 阶段 stop-memory 提示）
5. **维护**: 指数衰减、每签名 Top-K 上限、对反复被反驳先验的保守抑制
6. **无参数更新**: 区别于需重训练的方法，是终身导航中"免训练自进化"的实现

## 代表工作
- [[EvoMemNav]]: 提出 RDCMA 实现自进化导航
- [[Reflexion]]: LLM 智能体反思机制的相关思想

## 相关概念
- [[Episodic Memory]]
- [[Reflexion]]
- [[Visual-Semantic Memory Graph]]
- [[Coarse-to-Fine]]
