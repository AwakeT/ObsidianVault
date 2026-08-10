---
type: concept
aliases: [层次化记忆]
---

# Hierarchical Memory

## 定义
在多个时间或语义粒度上组织历史信息的记忆架构；层级既可对应步/子目标/指令，也可对应 scene/event/goal/长期活动摘要。

## 数学形式
N/A

## 核心要点
1. 用高层摘要缩短查询路径并防止上下文窗口溢出
2. 允许从高层粗定位，再按需展开到底层证据
3. 可在任务边界、时间间隔或语义变化处形成不同粒度节点
4. 摘要错误和底层噪声可能沿层级传播，需要证据回链与生命周期治理

## 代表工作
- [[FineCog-Nav]]: 三级记忆（步/子目标/指令）
- [[H2-EMV]]: scene graph instant → event → goal → higher-level summary 的在线历史树

## 相关概念
- [[Episodic Memory]]
- [[History Tree]]
- [[Selective Forgetting]]
- [[元认知]]
