# H²-EMV 论文分析与双记忆调研归类

> 论文：Learning to Forget -- Hierarchical Episodic Memory for Lifelong Robot Deployment  
> Bärmann et al., arXiv:2604.11306v2, 2026  
> 完整逐图逐表笔记：[[论文笔记/VLA与具身智能/H2-EMV|H2-EMV]]  
> 架构图：[[H2-EMV_层级情景记忆与自适应遗忘_一页总览.drawio]]

---

## 1. 归类结论

本文在当前调研体系中的主类应是：

**长期记忆驱动的具身智能体 → 外部情景/事件记忆 → 记忆生命周期治理 → 用户自适应选择性遗忘**。

它适合作为本目录 RAVEN + UcON 双记忆方案的 **M1 情景记忆治理扩展**，但不应被误写成第三种内容记忆，也不应被直接标为“双记忆方法”。H²-EMV 原论文只维护一棵多层级 episodic history tree；自然语言 relevance rule store 是“如何治理记忆”的元记忆。

### 排除项

- **不是 Continual Learning 的灾难性遗忘**：没有修改模型参数，也不研究旧任务能力保持；它主动删除外部记忆条目。
- **不是习惯/语义记忆获取方法**：不会从多次情景中统计“物体通常在哪”；规则描述的是“什么值得保留”。
- **不是 VLN 策略方法**：没有 SR/SPL 或闭环导航实验，主要评估过去经历问答、树规模和 LLM token。

---

## 2. 方法核心

H²-EMV 将长期情景记忆拆成四个闭环阶段：

1. **在线建树**：连续多模态观测被组织成 `scene → event → goal → higher-level summary` 的 History Tree。
2. **生命周期治理**：节点按层级获得默认寿命；到期后由 LLM 结合节点、父上下文与规则估计相关性 $\alpha$。
3. **问答使用**：Dialog Manager 把关于过去的问题交给 agentic EMV，后者逐层搜索/展开 History Tree。
4. **反馈学习**：若用户指出某个细节不该被忘，规则学习 LLM 更新自然语言 relevance rules；新规则同时影响后续摘要与遗忘。

遗忘不是简单删除。仍然过期的节点会变成 `forgotten placeholder`：只留时间范围和摘要首行，删除完整内容与子节点；EMV 看不到该摘要，但后续建树 LLM 可以用它维持上层摘要和时间连续性。

### 生命周期公式

$$
\tau_n=t_{end}+\Delta t_\ell\gamma_\ell,
\qquad
\gamma_\ell=
\begin{cases}
1,&\ell\le3,\\
2^{\ell-3},&\ell>3,
\end{cases}
$$

$$
\alpha=\operatorname{estimate\mbox{-}relevance}(n),
\qquad
\tau_n\leftarrow\tau_n+\alpha\Delta t_\ell.
$$

更新后仍有 $\tau_n<t_{now}$ 则遗忘；父节点寿命不得早于任何子节点。

---

## 3. 与现有双记忆方案的映射

| 当前方案 | 当前职责 | H²-EMV 的正确接入点 | 边界 |
|----------|----------|---------------------|------|
| M1 情景记忆 | 保存 pose/time/eid、视觉向量/图像和检测证据 | 在 M1 上增加 History Tree 时间目录、到期策略、相关性延寿、压缩/冷存和遗忘审计 | History Tree 不替代原始视觉/时空索引 |
| M2 习惯记忆 | 保存 object-relation-landmark-time 的稳定规律与置信度 | M2 的习惯置信度可作为 relevance signal；例如高频找物目标的支撑证据保留更久 | M2 不存 M1 原始事实；rule store 也不等同于 M2 习惯 schema |
| 在线查询 | M1 提供情景证据，M2 提供习惯先验 | 先用 History Tree 摘要缩小时间段，再回到 M1 图像/位姿取精确证据 | 不能仅依赖摘要回答精确导航位置 |
| 用户纠错 | 当前主要修正习惯或位置 | 增加“你不该忘 X”反馈通路，学习个性化 retention rules | 单次反馈不应无作用域地覆盖所有用户/任务 |

### 推荐架构角色

```text
M1 视觉-时空事实源
  ├─ evidence index：图像 / embedding / pose / detection
  ├─ History Tree：scene → event → goal → long summary
  └─ retention governor：TTL + relevance + placeholder + cold/delete policy

M2 习惯先验
  └─ object / relation / landmark / time_context / conf / support_refs

M-meta 治理规则（建议逻辑独立，不强行塞入 M2）
  └─ relevance rules / owner / scope / provenance / version / priority
```

---

## 4. 建议补充的数据字段

### M1 / History Tree 节点

```json
{
  "node_id": "evt_20260809_0012",
  "hierarchy_level": 2,
  "time_span": ["2026-08-09T08:10:00", "2026-08-09T08:13:12"],
  "summary": "The robot found the user's keys near the sofa.",
  "children": ["scene_..."],
  "evidence_refs": ["m1:e27:f084"],
  "expires_at": "2026-08-16T08:13:12",
  "retention_score": 3,
  "retain_rule_ids": ["rule_valuable_item_location"],
  "storage_tier": "hot",
  "forgetting_state": "active",
  "forgetting_audit_ref": "audit_..."
}
```

### Relevance Rule

```json
{
  "rule_id": "rule_valuable_item_location",
  "text": "Retain the exact time and location when a valuable item is seen.",
  "owner": "user_001",
  "scope": {"household": "home_A", "task": "object_search"},
  "priority": 80,
  "source_feedback": "dialog_20260809_turn_42",
  "created_at": "2026-08-09T20:30:00",
  "version": 3,
  "status": "active"
}
```

### Forgetting Audit

```json
{
  "audit_id": "audit_...",
  "node_id": "evt_...",
  "decision": "placeholder",
  "reason": "expired and no matching retention rule",
  "rule_snapshot": ["rule_...@v3"],
  "decided_at": "2026-08-16T02:00:00",
  "recoverable_until": "2026-09-15T02:00:00"
}
```

---

## 5. 适合当前方案的落地版本

原论文直接删除 children，家庭机器人产品中风险较高。建议分级实现：

| 状态 | 内容 | 查询可见性 | 触发 |
|------|------|------------|------|
| Hot | 完整图像、向量、检测、位姿、树节点 | 在线可见 | 新近或高相关性 |
| Warm | 关键帧 + 检测 + 摘要 + evidence refs | 在线可见 | 普通稳定经历 |
| Placeholder | 时间范围 + 摘要首行 + audit ref | 默认仅显示“细节已归档” | 到期且低相关性 |
| Cold | 加密压缩原证据 | 需权限/延迟恢复 | 删除前宽限期 |
| Deleted | 仅保留不可逆审计哈希或按隐私要求全部清除 | 不可恢复 | 用户删除或保留期结束 |

这样既吸收 H²-EMV 的查询效率，又避免 LLM 一次误判造成不可恢复的信息损失。

---

## 6. 实验结论与应谨慎解读之处

### 论文报告

- 摘要报告：记忆规模减少 45%，查询时计算减少 35%。
- TEACh 5-episode：完整系统正确率从 20% 升到 34%，这是相对 +70%，不是 +70 个百分点。
- ARMAR-7：35 段、20.5 小时真实多模态记录；完整系统是唯一在第二轮出现清晰提升的遗忘系统。
- 多周 live deployment：在线建树平均滞后 $47\pm41$ 秒，最大 7.2 分钟；遗忘在空闲/夜间执行，不阻塞问答快照读取。

### 风险

1. **检索是主要错误源**：规则已学会、信息仍在，也可能取错 episode。
2. **级联误差**：对象误识别会向上层摘要传播，甚至因命中规则而被保留更久。
3. **规则治理不足**：自然语言规则可能幻觉、误泛化、误删；原论文没有作用域、版本回滚和冲突解决。
4. **总计算不一定更低**：在线建树和遗忘使用大量 LLM token，只是把成本移出用户查询关键路径。
5. **非实时世界状态**：分钟级极端滞后意味着它适合“回顾过去”，不能替代导航的实时 world state。
6. **多用户冲突未解决**：家庭成员可能对“什么值得记”持相反意见。

---

## 7. 对当前双记忆方案的最终结论

H²-EMV 最值得吸收的不是它的文本 History Tree 本身，而是以下四个治理机制：

1. **层级相关的默认寿命**：细节短、高层摘要长；
2. **用户可教的保留规则**：从固定衰减升级为个性化 retention policy；
3. **摘要与遗忘共享相关性定义**：让“记重点”和“留细节”一致；
4. **显式 forgotten placeholder 与审计**：承认时间段存在，但细节已不可在线访问。

建议在 RAVEN/UcON 方案中将它实现为 **M1 retention governor + 独立 M-meta rule store**。M2 继续只负责习惯先验，二者通过 `support_refs` 和 relevance signals 交互，避免事实、习惯与治理规则三种语义混装。

---

## 参考

- [arXiv abstract](https://arxiv.org/abs/2604.11306)
- [Official PDF](https://arxiv.org/pdf/2604.11306v2)
- [[RAVEN_UcON_双记忆架构_方案一_工作流程]]
- [[论文笔记/视觉语言导航/RAVEN|RAVEN]]
- [[论文笔记/VLA与具身智能/Vesta|Vesta]]

