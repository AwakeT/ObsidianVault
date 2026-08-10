---
title: "Learning to Forget -- Hierarchical Episodic Memory for Lifelong Robot Deployment"
method_name: "H²-EMV"
authors: [Leonard Bärmann, Joana Plewnia, Alex Waibel, Tamim Asfour]
year: 2026
venue: arXiv
tags: [episodic-memory, selective-forgetting, hierarchical-memory, lifelong-robot, human-robot-interaction, memory-verbalization]
zotero_collection: "未入库"
image_source: local
arxiv_html: https://arxiv.org/html/2604.11306v2
created: 2026-08-09
---

# 论文笔记：Learning to Forget -- Hierarchical Episodic Memory for Lifelong Robot Deployment

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Karlsruhe Institute of Technology (KIT), Institute for Anthropomatics and Robotics |
| 版本 | arXiv v2，2026-05-05（v1：2026-04-13） |
| 任务 | [[Episodic Memory Verbalization|情景记忆言语化（EMV）]] |
| 前身 | H-EMV：离线构建的层级历史树 |
| 数据 | TEACh 模拟家庭任务；ARMAR-7 的 35 段、共 20.5 小时真实多模态记录 |
| 链接 | [arXiv](https://arxiv.org/abs/2604.11306) / [PDF](../论文原件/2604.11306v2.pdf) / [DOI](https://doi.org/10.48550/arXiv.2604.11306) |
| 代码 | 论文未提供公开代码仓库 |

---

## 一句话总结

> H²-EMV 用在线层级历史树组织机器人经历，以用户反馈学习自然语言保留规则，在到期时选择性遗忘，从而把“记什么”变成可个性化、可检查的生命周期策略。

---

## 核心贡献

1. **在线层级情景记忆**：把 H-EMV 的离线回顾式建树改为随观测到达而增量更新的 [[History Tree|History Tree]]。
2. **相关性感知的选择性遗忘**：每个节点有层级相关的默认寿命；过期后由 [[LLM]] 结合节点、父上下文和用户规则估计相关性，再决定延寿或替换成忘记占位符。
3. **用户反馈学习保留规则**：若问答暴露了“不该忘却已忘”的细节，规则学习 LLM 可增、改、删自然语言规则，并同时影响后续摘要和遗忘。
4. **模拟与真实部署验证**：在 TEACh 和 20.5 小时 ARMAR-7 数据上评估，并进行多周在线部署；摘要报告记忆规模下降 45%、查询时计算下降 35%，第二轮正确率相对提升 70%。

---

## 问题背景

### 要解决的问题

长期部署机器人持续接收图像、本体感知、动作、符号状态与对话。若把所有经历永久保留，[[Episodic Memory|情景记忆]]的存储和查询计算会无界增长；但如果只按固定时间删除，又会忘掉用户真正关心的细节，例如贵重物品的位置、失败原因或与某人的相遇时间。

### 现有方法的局限

- 扁平事件日志或 RAG 能检索，但记忆仍持续增长，长历史下查询 token 成本爆炸。
- H-EMV 的 [[Hierarchical Memory|层级记忆]]能通过递归摘要降低查询复杂度，但依赖查询时离线建树，不适合持续部署。
- 固定时间衰减无法表达“对不同用户、任务而言，相关性不同”。
- 只保留不遗忘的系统在第一轮可能更准，却会随历史增长发生错误检索和响应延迟。

### 本文动机

把机器人记忆拆成三个相互协同的问题：

1. **结构化**：新经历如何在线并入多粒度历史；
2. **治理**：哪些到期节点值得延寿；
3. **适配**：如何从用户对“你不该忘”的反馈中更新相关性定义。

本文所说的 forgetting 是对外部显式记忆条目的主动保留/删除策略，不是 [[Continual Learning]] 中模型参数知识的灾难性遗忘。

---

## 方法详解

### 总体架构

H²-EMV 是一套 **在线建树 + 生命周期治理 + agentic 问答 + 反馈学习** 的闭环：

1. 机器人技能执行与环境观测产生时间戳多模态快照 $r_t$。
2. 快照被转为最低层的 [[Scene Graph|场景图]]瞬时节点，并自底向上增量分组、摘要到历史树。
3. 每次建树后，系统自顶向下检查到期节点；[[Relevance Estimation|相关性估计]]决定延寿或遗忘。
4. 用户问题由 Dialog Manager 路由给 EMV；[[Agentic Memory|agentic LLM]]按需逐层展开历史树。
5. 若答案暴露了忘失细节，用户反馈进入规则学习器，更新自然语言 relevance rules；这些规则同时条件化建树摘要与遗忘判断。

对应可编辑中文架构图：[[../../研究资料/13_dual_memory_navigation/H2-EMV_层级情景记忆与自适应遗忘_一页总览.drawio]]。

### History Tree 的层级

| 层级 | 节点 | 内容与构建方式 |
|------|------|----------------|
| 原始观测 $ell=0$ | raw observation $r_t$ | 图像、本体感知、动作、符号状态、对话等时间戳快照 |
| L1 / $ell=1$ | scene graph instant $s_t$ | 一条原始观测及其推断信息，如检测对象 |
| L2 / $ell=2$ | event summary $e_j$ | 场景图/动作/对话变化触发的事件摘要 |
| L3 / $ell=3$ | goal summary $g_k$ | 目标或任务段；ARMAR-7 中允许嵌套子目标 |
| L4+ / $ell>3$ | higher-level summary $h_n$ | LLM 递归分组得到的更长时段活动摘要 |

每个摘要节点 $n$ 保存自然语言摘要 $S_n$ 与下层子节点集合 $operatorname{children}(n)$。

### 模块 1：时间感知的在线增量建树

给定已有树 $H_{t-1}$ 与新 scene node $s_t$，算法只重算时间上可见、确实受影响的尾部，而不是重建整棵树：

- 在每层 $ell$，向 LLM 提供现有父分组、尚无父节点的新项以及自然语言 relevance rules；
- LLM 可把新项并入现有分组、创建新分组，或重组最近节点；
- visibility cutoff 限制可重算的历史范围；time clustering 阻止跨夜、跨大时间间隔的错误合并；
- 变化从低层向高层传播，直到某层不再变化；
- relevance rules 不只控制删除，也让摘要优先保留用户关心的属性。

#### Algorithm 1：Time-aware incremental grouping and summarization

```text
Input: layer ℓ+1 的旧父节点 P_old；layer ℓ 的新项 N_new
N_all ← 所有 P_old 的 children
C ← time-based-cluster(N_all)
若每个节点都被分成单项簇，则退化为一个整体簇，避免单项摘要
P_merged ← []
for c in C:
    P_c ← children 完全落入 c 的旧父节点
    N_new,c ← N_new ∩ c
    if N_new,c 为空:
        原样复用 P_c，避免无意义重摘要
    else:
        P_merged ← P_merged ◦ group-and-summarize_LLM(P_c, N_new,c, ℓ)
Output: P_merged
```

#### Algorithm 2：Online history tree construction

```text
Input: 新 scene s_t；旧树 H_{t-1}
T ← copy(H_{t-1}); ℓ ← 0; N_new ← {s_t}
while ℓ < L-1:
    P_old ← T 在 layer ℓ+1 的边缘父节点
    若时间跨度不应继续上推:
        把 N_new 插入最近父节点，并用 LLM 更新摘要；停止
    t_cutoff ← visibility-cutoff(T, ℓ, N_new)
    P_visible ← 结束时间不早于 t_cutoff 的旧父节点
    P_merged ← Algorithm 1(P_visible, N_new, ℓ)
    P_new ← 不可见旧父节点 ◦ P_merged
    i_changed ← 第一个变化位置
    若无变化：停止
    N_new ← P_new[i_changed:]; ℓ ← ℓ+1
    若到最高层：T ← simple-summarize_LLM(P_new)
    否则：移除受影响旧父节点及其 siblings，作为下一层 N_new 重算
Output: H_t ← T
```

### 模块 2：相关性感知的选择性遗忘

遗忘流程是自顶向下的：

1. 节点创建时依据结束时间、层级默认跨度和层级倍率设置 $	au_n$；越高层的摘要寿命越长。
2. 若 $	au_n<t_{	ext{now}}$，LLM 读取当前节点、父节点上下文和规则集，输出整数相关性因子 $alpha$，也可输出 `inf` 表示永久保留。
3. 用 $alpha$ 延长寿命；若延长后仍过期，则遗忘。
4. 父节点寿命至少与最长寿的子节点一致，避免父节点先于仍需保留的子节点到期。
5. 被忘节点不直接消失，而是替换为 **forgotten placeholder**：只保留时间范围与摘要首行，删除正文和 children；相邻占位符合并。占位符对 EMV 隐藏，但对后续建树摘要 LLM 可见，既保留时间空洞，又避免上层摘要写成“此处已遗忘”。

### 模块 3：Dialog Manager 与 EMV

Dialog Manager 把用户话语分成三类：

- 关于过去的问题：调用 `answer_question_about_my_past`；
- 对忘失细节的反馈：调用 `handle_forgetting_feedback`；
- 其他话语：由一般对话系统直接处理。

EMV 使用 agentic LLM 对历史树进行语义搜索并逐层展开；证据充分时作答，证据不足时继续检索，或明确说明信息已被遗忘。

### 模块 4：反馈驱动的 relevance rule learning

规则学习 LLM 读取现有规则和新反馈，可执行：

- 新增规则；
- 修改或细化已有规则；
- 合并相似规则；
- 删除被用户纠正的规则。

论文示例把“洗碗机卸载时记住使用哪只手”和“装载时记住装了什么物体”合并成 dishwasher task 的结构化自然语言规则。规则可读、可人工修订，但也可能被 LLM 误泛化或意外删除。

### 真实部署调度

- scene construction 与 EMV 在 ARMAR-7 内部 PC 运行；在线建树和遗忘卸载到校内 GPU 服务器。
- 技能开始/结束、机器人或用户发言、人脸识别会触发新 scene batch。
- 在线建树异步处理；若队列已有新更新，遗忘让位；遗忘可安全中断，并在夜间空闲期集中执行。
- 查询总是读取最近完成的树快照，因此不被正在进行的建树阻塞。
- 多周部署中，建树平均滞后 $47pm41$ 秒，最大 7.2 分钟；低活动时第 10 百分位约 3 秒。

---

## 关键公式

### 公式 1：[[Selective Forgetting|节点默认到期时间]]

$$
\tau_n = t_{\text{end}} + \Delta t_\ell \cdot \gamma_\ell
$$

**含义**：节点从其覆盖事件的结束时刻起获得一个与层级相关的默认寿命。

**符号说明**：
- $\tau_n$：节点 $n$ 的到期时间；
- $t_{\text{end}}$：节点所描述时间段的结束时间；
- $\Delta t_\ell$：层级 $ell$ 的基础保留跨度，例如原始观测 15 分钟、goal 层 1 天；
- $\gamma_\ell$：高层摘要的寿命倍率。

### 公式 2：[[Hierarchical Memory|层级寿命倍率]]

$$
\gamma_\ell =
\begin{cases}
1, & \ell \le 3, \\
2^{\ell-3}, & \ell > 3.
\end{cases}
$$

**含义**：L1-L3 不额外放大寿命；L4 起每升一层寿命倍率翻倍，使高层摘要比细节保留更久。

**符号说明**：
- $\ell$：从原始观测层 $ell=0$ 起计数的树层级；
- $\gamma_\ell$：该层的寿命倍率。

### 公式 3：[[Selective Forgetting|到期检查]]

$$
\tau_n < t_{\text{now}}
$$

**含义**：只有当前时间已经超过到期时间的节点才进入相关性估计。

**符号说明**：
- $t_{\text{now}}$：执行遗忘检查时的当前时间。

### 公式 4：[[Relevance Estimation|相关性因子估计]]

$$
\alpha = \operatorname{estimate\mbox{-}relevance}(n)
$$

**含义**：LLM 根据节点、父上下文和自然语言规则输出延寿强度；无规则命中时默认 $\alpha=0$，永久保留时输出 `inf`。

**符号说明**：
- $n$：待评估的过期节点；
- $\alpha$：非负整数相关性因子或无穷大。

### 公式 5：[[Selective Forgetting|相关性加权延寿]]

$$
\tau_n \gets \tau_n + \alpha \cdot \Delta t_\ell
$$

**含义**：相关性越高，节点延长的基础寿命周期越多；更新后仍过期的节点被遗忘。

**符号说明**：
- $\alpha$：相关性因子；
- $\Delta t_\ell$：当前层的基础保留跨度。

### 公式 6：[[Hierarchical Memory|父子寿命一致性]]

$$
\tau_n \gets \max\left(\max_{c\in C}\tau_c,\;\tau_n\right)
$$

**含义**：父节点不得早于任何仍存活子节点过期。

**符号说明**：
- $C=\operatorname{children}(n)$：节点 $n$ 的子节点集合；
- $\tau_c$：子节点 $c$ 的到期时间。

---

## 关键图表

### Figure 1：系统架构、增量建树与遗忘流程

![[H2-EMV_fig1_system_architecture.png]]

**说明**：上半部分是完整闭环；左下是时间感知的增量分组；右下是自顶向下遗忘。最关键的双向作用是：自然语言规则同时影响 `Group & Summarize` 和 `Relevance Estimation`。

### Figure 2：两轮评估流程与成功/失败样例

![[H2-EMV_fig2_evaluation_procedure.png]]

**说明**：第一轮问题发生在默认寿命已过之后，随后用反馈强化相关性；第二轮用相似事件测试规则是否泛化。图中也显示规则学习成功但检索时间略偏、以及规则未泛化的失败。

### Figure 3：模拟与真实记录的总体结果

![[H2-EMV_fig3_experiment_results.png]]

**说明**：层级 + 在线 + 学习式遗忘在问答、树规模和查询 token 间形成折中。无遗忘扁平基线性能高，但长历史下 token 成本和树规模明显失控。

### Figure 4：ARMAR-7 多周在线部署

![[H2-EMV_fig4_live_deployment.png]]

**说明**：上图显示接收 scene 与处理完成之间的异步滞后；下图显示树白天增长、夜间遗忘大幅压缩，队列在活动密集期短暂升高。

### Figure A1：全部消融结果

![[H2-EMV_figA1_all_results.png]]

**说明**：补全正文 Figure 3 未展示的消融，包括“只对摘要学习”“不让摘要使用规则”等组合。

### Figure B2：成功学习“记住洗衣区访问时间”

![[H2-EMV_figB2_successful_example.png]]

**说明**：第一轮无记录；反馈后，第二轮能给出最近访问洗衣区的时间。

### Figure B3：规则成功泛化但时间误差 8 分钟

![[H2-EMV_figB3_partial_example.png]]

**说明**：从“第一次遇到 Leonard 的准确时间”泛化到“最近见到 Joana 的时间”，说明规则能跨人物迁移，但检索/时间 grounding 仍有偏差。

### Figure B4：正确但信息过多（TMI）的柜台物体样例

![[H2-EMV_figB4_correct_tmi_counter.png]]

**说明**：目标信息保留成功，但回答混入了相邻时间段的其他物体，显示层级摘要和检索边界仍不精确。

### Figure B5：药品位置记忆

![[H2-EMV_figB5_correct_tmi_medicine.png]]

**说明**：规则要求记住药品的精确位置，第二轮成功定位到移动厨房洗碗机附近，但时间仍有偏差。

### Figure B6：贵重物品规则从钥匙泛化到钱包

![[H2-EMV_figB6_correct_tmi_keys.png]]

**说明**：用户把 keys 抽象为 valuable item，规则能迁移到 wallet；答案位置正确但时间区间略早。

### Figure B7：规则已学习但检索到错误 episode

![[H2-EMV_figB7_retrieval_failure.png]]

**说明**：记忆内容仍在、规则也生效，但 QA LLM 取错“最后一次”episode，证明主要瓶颈仍可能是检索而非遗忘。

### Figure B8：规则泛化失败

![[H2-EMV_figB8_generalization_failure.png]]

**说明**：从 dishwasher load 的对象规则未泛化到 unload 的对象，暴露自然语言规则的方向性与覆盖范围问题。

### Figure B9：相关性估计缺上下文 + QA 误解问题

![[H2-EMV_figB9_context_failure.png]]

**说明**：规则本身正确，但 prompt 上下文不足导致相关性估计漏掉目标动作，随后 QA LLM 又误解问题，形成级联错误。

### Table A1：全部方法与消融配置

| Method | Tree construction | Forget by time | Relevance estimate | Learn for forgetting | Learn for summarization |
|--------|-------------------|----------------|--------------------|----------------------|-------------------------|
| ours | incremental | ✓ | ✓ | ✓ | ✓ |
| no summ. learn. | incremental | ✓ | ✓ | ✓ | ✗ |
| no learning | incremental | ✓ | ✓ | ✗ | ✗ |
| learn only summ. | incremental | ✓ | ✗ | ✗ | ✓ |
| time-only forg. | incremental | ✓ | ✗ | ✗ | ✗ |
| no f. learn summ. | incremental | ✗ | — | — | ✓ |
| no forg. | incremental | ✗ | — | — | ✗ |
| offl. full | offline | ✓ | ✓ | ✓ | ✓ |
| offl. no learning | offline | ✓ | ✓ | ✗ | ✗ |
| offl. time-only forg. | offline | ✓ | ✗ | ✗ | ✗ |
| offl. no forg. | offline | ✗ | — | — | ✗ |
| flat full | flat (L3) | ✓ | ✓ | ✓ | ✓ |
| flat no forg. | flat (L3) | ✗ | — | — | ✗ |

**说明**：实验将建树方式、时间衰减、LLM 相关性估计，以及规则是否作用于遗忘/摘要拆开，验证它们的交互而非只看单模块增益。

### Table A2：TEACh，$|h|=5$

| Method | $S_c$ | $S_p$ | $S_\uparrow$ | $S_\equiv$ | $S_c^1$ | $S_p^1$ | $S_c^2$ | $S_p^2$ | $N_f$ | $N_\varnothing$ | $C_{qa}$ K | $C_f$ K |
|--------|------:|------:|--------------:|-------------:|--------:|--------:|--------:|--------:|--------:|-----------------:|-----------:|--------:|
| ours | 27 | 41 | 34 | 50 | 20 | 34 | 34 | 48 | 52.2 | 38.8 | 9.4 | 152.9 |
| no summ. learn. | 20 | 27 | 16 | 64 | 22 | 28 | 18 | 26 | 45.7 | 38.2 | 9.1 | 150.3 |
| no learning | 17 | 27 | 16 | 56 | 20 | 34 | 14 | 20 | 43.6 | 38.6 | 7.9 | 127.2 |
| learn only summ. | 22 | 37 | 30 | 58 | 18 | 28 | 26 | 46 | 43.9 | 37.7 | 9.1 | — |
| time-only forg. | 17 | 26 | 18 | 58 | 18 | 30 | 16 | 22 | 42.7 | 37.7 | 8.3 | — |
| no f. learn summ. | 43 | 60 | 34 | 24 | 48 | 64 | 38 | 56 | 135.0 | 63.9 | 10.2 | — |
| no forg. | 38 | 56 | 20 | 38 | 48 | 66 | 28 | 46 | 131.8 | 63.1 | 9.1 | — |
| offl. full | 9 | 17 | 20 | 70 | 2 | 12 | 16 | 22 | 19.0 | 18.7 | 8.0 + 3.5 | 69.4 |
| offl. no learning | 7 | 15 | 18 | 70 | 2 | 12 | 12 | 18 | 17.4 | 18.3 | 6.2 + 3.5 | 58.5 |
| offl. time-only forg. | 4 | 9 | 6 | 86 | 2 | 10 | 6 | 8 | 13.4 | 14.9 | 6.2 + 3.5 | — |
| offl. no forg. | 51 | 69 | 18 | 52 | 58 | 74 | 44 | 64 | 94.9 | 71.8 | 11.1 + 3.5 | — |
| flat full | 9 | 9 | 2 | 94 | 10 | 10 | 8 | 8 | — | — | 15.0 | 147.7 |
| flat no forg. | 56 | 79 | 22 | 56 | 54 | 80 | 58 | 78 | — | — | 9.8 | — |

**关键发现**：完整系统正确率从第一轮 20% 升至第二轮 34%，即相对 +70%；与不学习规则的版本相比，第二轮恢复明显，同时保持较小树规模。

### Table A3：TEACh，$|h|=25$

| Method | $S_c$ | $S_p$ | $S_\uparrow$ | $S_\equiv$ | $S_c^1$ | $S_p^1$ | $S_c^2$ | $S_p^2$ | $N_f$ | $N_\varnothing$ | $C_{qa}$ K | $C_f$ K |
|--------|------:|------:|--------------:|-------------:|--------:|--------:|--------:|--------:|--------:|-----------------:|-----------:|--------:|
| ours | 17 | 22 | 20 | 64 | 14 | 20 | 20 | 24 | 265.1 | 148.3 | 10.2 | 1369.8 |
| no summ. learn. | 15 | 17 | 8 | 76 | 18 | 20 | 12 | 14 | 223.9 | 133.1 | 10.4 | 1354.7 |
| no learning | 23 | 27 | 14 | 66 | 24 | 32 | 22 | 22 | 178.5 | 114.4 | 8.8 | 1165.8 |
| learn only summ. | 17 | 23 | 16 | 72 | 14 | 20 | 20 | 26 | 187.3 | 117.1 | 10.4 | — |
| time-only forg. | 19 | 22 | 10 | 70 | 22 | 28 | 16 | 16 | 175.3 | 112.6 | 9.0 | — |
| no f. learn summ. | 34 | 43 | 20 | 50 | 40 | 48 | 28 | 38 | 944.3 | 441.5 | 12.8 | — |
| no forg. | 24 | 31 | 22 | 48 | 26 | 34 | 22 | 28 | 917.8 | 431.3 | 11.1 | — |
| offl. full | 10 | 16 | 20 | 74 | 4 | 10 | 16 | 22 | 26.5 | 26.9 | 9.9 + 5.3 | 363.5 |
| offl. no learning | 7 | 13 | 8 | 78 | 8 | 16 | 6 | 10 | 29.1 | 28.9 | 8.3 + 5.3 | 324.7 |
| offl. time-only forg. | 3 | 8 | 8 | 88 | 2 | 6 | 4 | 10 | 22.7 | 21.8 | 8.1 + 5.3 | — |
| offl. no forg. | 32 | 38 | 10 | 58 | 42 | 48 | 22 | 28 | 621.6 | 353.6 | 15.7 + 5.3 | 0.0 |
| flat full | 7 | 7 | 4 | 94 | 6 | 6 | 8 | 8 | — | — | 16.5 | 1330.4 |
| flat no forg. | 71 | 80 | 18 | 64 | 70 | 82 | 72 | 78 | — | — | 37.0 | — |

**关键发现**：历史长度增至 25 个 episode 后，所有层级遗忘方法的 QA 都更难；但无遗忘树从 131.8/135.0 级增长到约 918/944 个 L3+ 节点，而完整系统最终树为 265.1，效率差距显著扩大。

### Table A4：TEACh 低层信息完全遗忘比例（%）

| Method | $|h|=5$ Round 1 | $|h|=5$ Round 2 | $|h|=25$ Round 1 | $|h|=25$ Round 2 |
|--------|------------------:|------------------:|-------------------:|-------------------:|
| ours | 100 | 64 | 100 | 80 |
| no summ. learn. | 100 | 70 | 100 | 80 |
| no learning | 100 | 100 | 100 | 100 |
| learn only summ. | 100 | 100 | 100 | 100 |
| time-only forg. | 100 | 100 | 100 | 100 |
| no f. learn summ. | 0 | 0 | 0 | 0 |
| no forg. | 0 | 0 | 0 | 0 |
| offl. full | 100 | 74 | 100 | 82 |
| offl. no learning | 100 | 100 | 100 | 100 |
| offl. time-only forg. | 100 | 100 | 100 | 100 |
| offl. no forg. | 0 | 0 | 0 | 0 |
| flat full | 100 | 98 | 100 | 96 |
| flat no forg. | 0 | 0 | 0 | 0 |

**关键发现**：学习式相关性估计能在第二轮把完全遗忘率从 100% 降到 64%/80%；只让规则影响摘要而不影响遗忘时仍是 100%，说明“摘要记重点”和“细节是否延寿”是两种不同能力。

### Table A5：ARMAR-7 真实记录实验

| Method | $S_c$ | $S_p$ | $S_\uparrow$ | $S_\equiv$ | $S_c^1$ | $S_p^1$ | $S_c^2$ | $S_p^2$ | $N_f$ | $N_\varnothing$ | $C_{qa}$ K | $C_f$ K |
|--------|------:|------:|--------------:|-------------:|--------:|--------:|--------:|--------:|--------:|-----------------:|-----------:|--------:|
| ours | 16 | 36 | 34 | 54 | 12 | 24 | 20 | 48 | 231.8 | 185.5 | 13.9 | 3350.4 |
| no learning | 11 | 24 | 20 | 58 | 8 | 26 | 14 | 22 | 205.6 | 173.6 | 15.0 | 2801.4 |
| learn only summ. | 16 | 23 | 14 | 64 | 18 | 26 | 14 | 20 | 207.0 | 176.8 | 18.0 | — |
| time-only forg. | 11 | 22 | 14 | 68 | 10 | 24 | 12 | 20 | 193.4 | 171.6 | 16.3 | — |
| no forg. | 29 | 46 | 6 | 62 | 36 | 56 | 22 | 36 | 459.6 | 226.0 | 9.7 | — |
| offl. full | 9 | 20 | 18 | 70 | 8 | 18 | 10 | 22 | 50.8 | 50.4 | 23.4 + 4.9 | 2255.3 |
| offl. no forg. | 38 | 53 | 28 | 50 | 32 | 54 | 44 | 52 | 259.2 | 162.1 | 13.7 + 4.9 | — |

**关键发现**：真实数据中只有完整系统从第一轮到第二轮出现清晰提升（正确率 12→20，部分正确 24→48）；无遗忘在线系统反而从 36→22，说明更长历史带来的错误检索可能抵消“什么都保存”的优势。

---

## 实验

### 数据集与样本构造

| 数据 | 规模 | 特点 | 用途 |
|------|------|------|------|
| TEACh | 每条长历史随机拼接 5 或 25 个 episode | 使用模拟器 GT 对象状态，排除感知误差；动作/对象需至少出现两次且间隔超过低层寿命 | 大规模两轮 QA、消融与长度扩展 |
| ARMAR-7 recordings | 35 段，共 20.5 小时 | 图像、本体感知、符号 scene/action、用户对话；家庭任务和演示 | 真实域迁移与人工筛选 QA |
| ARMAR-7 QA benchmark | 5 条历史 × 10 episodes；每条 10 对问题，即 20 个 QA 样本 | 先自动生成，再人工筛选、自然化与复核 | 真实记录定量评估 |
| Live deployment | 多周 | 异步建树、夜间遗忘、真实对话 API | 工程可行性与延迟观察 |

### 两轮评估设计

1. 选取在同一历史中至少出现两次、且间隔较大的动作或对象。
2. 第一次出现后等待超过低层默认寿命，再问 $Q_1$；此时目标细节大概率已忘。
3. 不论答案是否正确，都把问题改写为“以后应始终记住 X”的反馈，等价于从查询行为推断相关性。
4. 最后一次相似事件后再次等待，再问 $Q_2$，测试规则学习能否改善保留与回答。
5. 每条历史独立评估，规则不跨样本迁移。

### 指标

- $S_c, S_p$：正确/部分正确答案比例；判定器在 200 个 TEACh 人工标注样本上达到 92% 分类准确率，论文使用 GPT-5 作为 judge，并在真实域对每条判定人工修正。
- $S_\uparrow$：从第一轮到第二轮评分等级改善的 QA 对比例。
- $S_\equiv$：两轮评分等级相同的 QA 对比例。
- $N_f, N_\varnothing$：最终/平均 L3+ 节点数。
- $C_{qa}$：EMV 查询历史树的 LLM token；离线建树基线还包含查询时建树成本。
- $C_f$：遗忘机制自身的离线 token 成本。

### 实现细节

- 增量分组/摘要和相关性估计的附录示例使用 `Llama-3.3-70B-Instruct`。
- relevance rule learning 的附录示例使用 `gpt-5-2025-08-07`。
- 本文是 LLM 提示与外部记忆系统，不涉及常规 optimizer、batch size 或 epoch 训练。
- TEACh 中为加速实验，L2 event 在 scene graph/动作变化或收到对话时创建；L3 goal 收集到非导航 interaction action 为止；L4+ 才用通用增量摘要算法。
- ARMAR-7 中 L2 由 scene graph、语音或 skill status 变化触发；L3 使用 skill invocation stack 恢复嵌套 subgoal。

### 如何解读摘要中的 45% / 35% / 70%

- **70%**：TEACh $|h|=5$ 中，完整系统正确率 $S_c^1=20$、$S_c^2=34$，相对提升 $(34-20)/20=70\%$，不是提升 70 个百分点。
- **45% 与 35%**：是作者摘要给出的代表性效率结论，应理解为特定实验/基线下的比较，不应外推为所有长度与真实场景都固定节省该比例。
- 这些收益并非“免费”：完整系统把大量计算转移到在线建树和离线遗忘，$C_f$ 很高，只是把用户问答时的关键路径变轻。

---

## 批判性思考

### 优点

1. **把遗忘变成可学习的治理问题**：不是固定 TTL，也不是只靠 LLM 常识，而是显式从用户反馈更新规则。
2. **规则可检查、可人工修订**：相比隐式向量权重，自然语言 relevance rules 更便于审计和用户解释。
3. **摘要与遗忘共用相关性定义**：既控制“删什么”，也控制“高层摘要强调什么”，形成协同。
4. **保留时间空洞**：forgotten placeholder 既节省内容，又避免时间轴被悄悄抹平。
5. **工程调度现实**：异步建树、可中断遗忘、快照查询把非实时 LLM 推理移出问答关键路径。

### 局限性

1. **主要错误仍来自检索**：附录多例显示信息还在但取错 episode，遗忘改进不能替代更强的时间/空间 grounding。
2. **级联误差**：底层对象误识别会被摘要层放大，且“被规则认为重要”的错误对象可能保留更久。
3. **规则不精确且会漂移**：LLM 增删自然语言规则可能幻觉、误泛化或意外删除；论文没有形式化一致性、版本回滚和冲突检测。
4. **遗忘成本很高**：$C_f$ 在长历史和真实数据中达到百万级 token；总计算成本可能高于无遗忘系统，只是查询路径更快。
5. **并非严格实时**：平均滞后 47 秒，极端 7.2 分钟；适合回顾过去，不适合作为实时安全状态。
6. **模拟评估偏理想**：TEACh 使用 GT 对象状态，低估真实感知错误；第一轮被设计成大概率遗忘，也强化了第二轮提升的可见度。
7. **多用户未解决**：不同家庭成员的保留规则可能冲突，论文未给出作用域、权限、优先级或隐私隔离。

### 潜在改进方向

1. 为规则增加 `rule_id / scope / owner / priority / provenance / version / rollback`，并在执行前做冲突检测。
2. 把时间、空间、人物和对象身份做成结构化约束，降低 QA LLM 取错 episode 的概率。
3. 用 novelty、任务成功/失败、社会情绪和视觉满意度补充纯语言相关性。
4. 在删除前保留可逆冷存储或哈希审计记录，实现“逻辑遗忘”与“物理删除”分级。
5. 让规则同时影响感知采样和对象检测优先级，避免重要细节在进入历史树前就被漏掉。

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型/完整运行环境公开
- [x] 两个核心算法给出伪代码
- [x] 三类核心 prompt 和示例模型输出公开
- [x] 全部消融表格与主要调度细节给出
- [ ] ARMAR-7 原始 20.5 小时数据公开可获取

---

## 调研归类与现有系统映射

### 主归类

**长期记忆驱动的具身智能体 → 外部情景/事件记忆 → 记忆生命周期治理 → 用户自适应 [[Selective Forgetting|选择性遗忘]]**。

### 不应归入

- **不是灾难性遗忘缓解**：没有通过回放、正则或参数隔离保护模型权重。
- **不是新的双记忆结构**：论文只有一棵多层级 episodic history tree；自然语言 rule store 是治理元记忆，不是与情景记忆并列的语义/习惯事实库。
- **不是纯 VLN 方法**：评估指标是过去经验问答、树规模和 token 成本，没有导航 SR/SPL 或闭环路径规划验证。

### 与当前 RAVEN + UcON 双记忆方案的关系

| 现有组件 | H²-EMV 可补强的部分 | 不应直接照搬的部分 |
|----------|-------------------|--------------------|
| M1 情景事实 | 增加 `retention governor`：层级摘要、到期时间、相关性延寿、forgotten placeholder、遗忘审计 | 不应把所有 RAVEN 向量条目强行改写为纯文本树；视觉证据仍需独立索引 |
| M2 习惯先验 | 可把稳定习惯/用户偏好作为 relevance signal，影响哪些 M1 证据保留更久 | M2 不应存被保留的原始事实，也不应与 rule store 混为同一 schema |
| 对话/纠错 | “你不该忘 X”可更新规则，补齐现有固定衰减策略 | 用户规则需要作用域、权限与回滚，不能让单次反馈全局生效 |
| 在线查询 | H²-EMV 的层级摘要先粗定位，再回到 RAVEN 图像/时空检索取证 | 仅靠摘要式 EMV 难以满足导航所需的精确位姿与视觉 grounding |

**建议落地形态**：保留 RAVEN 风格 M1 作为视觉-时空事实源，在其上建立轻量 History Tree 作为时间层级目录；H²-EMV retention governor 决定哪些 M1 细节保持热存、压缩或进入冷存。自然语言规则作为单独的治理/偏好元记忆，并可接收 M2 习惯置信度作为辅助信号。

---

## 关联笔记

### 最邻近

- [[RAVEN]]：同为长时程机器人经验问答；RAVEN 解决高保真视觉-时空检索，H²-EMV 解决生命周期和“该忘什么”，两者互补。
- [[Vesta]]：Memory Harness 使用手工图像/文本采样；H²-EMV 把保留策略改成可学习、个性化规则。
- [[OVAL]]：跨 episode 终身对象记忆与实例去重，但没有用户自适应遗忘。
- [[EvoMemNav]]：任务反馈用于更新导航记忆；H²-EMV 的反馈用于更新遗忘规则。
- [[TrajRAG]]：分层压缩跨 episode 轨迹经验，但主要关注导航检索而非长期删除治理。

### 方法相关

- [[Episodic Memory]]：本文操作的外部经历记忆类型。
- [[Hierarchical Memory]]：scene → event → goal → higher-level summary 的组织方式。
- [[History Tree]]：H²-EMV 的核心数据结构。
- [[Selective Forgetting]]：基于时间与相关性的主动保留策略。
- [[Relevance Estimation]]：决定过期节点延寿强度。
- [[Episodic Memory Verbalization]]：对过去经历进行自然语言问答的任务。
- [[Agentic Memory]]：查询时主动展开外部记忆的使用范式。

---

## 速查卡片

> [!summary] H²-EMV
> - **核心**：在线 History Tree + 到期衰减 + LLM 规则相关性 + 用户反馈学习
> - **层级**：raw observation → L1 scene → L2 event → L3 goal → L4+ summary
> - **遗忘**：过期节点估计 $\alpha$；仍过期则变 placeholder，而非完全抹去时间轴
> - **结果**：摘要报告 memory −45%、query compute −35%；TEACh 5-episode 第二轮正确率相对 +70%
> - **边界**：外部情景记忆治理，不是参数灾难性遗忘，也不是新的双记忆架构
> - **主要风险**：检索错误、级联感知误差、规则漂移、高离线 token 成本、多用户冲突
> - **代码**：未公开

---

*笔记创建时间：2026-08-09 ；基于 arXiv:2604.11306v2 全文、附录与官方 TeX 源核对。*
