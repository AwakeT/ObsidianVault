---
title: "EvoMemNav: Efficient Self-Evolving Fine-Grained Memory for Zero-Shot Embodied Navigation"
method_name: "EvoMemNav"
authors: [Zuhao Ge, Xiaosong Jia, Chao Wu, Yuchen Zhou, Zuxuan Wu, Yu-Gang Jiang]
year: 2026
venue: arXiv
tags: [zero-shot-navigation, embodied-navigation, visual-memory, scene-graph, coarse-to-fine, self-evolving-memory, vision-language-model]
zotero_collection: ""
image_source: local
arxiv_html: ""
created: 2026-06-05
---

# 论文笔记：EvoMemNav: Efficient Self-Evolving Fine-Grained Memory for Zero-Shot Embodied Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Institute of Trustworthy Embodied AI (TEAI), Fudan University；Shanghai Key Laboratory of Multimodal Embodied AI |
| 日期 | June 2026 |
| 项目主页 | N/A |
| 对比基线 | [[3D-Mem]]、[[MTU3D]]、[[MSGNav]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.03509) / [Code](https://github.com/caicaiya123/EvoMemNav) |

---

## 一句话总结

> EvoMemNav 构建以原始视图为一等记忆的视觉语义记忆图（VSMGraph），配合预算化粗到精导航策略与反思驱动的记忆回写（RDCMA），在零样本具身导航中无需训练即实现 SR/SPL 一致提升。

---

## 核心贡献

1. **VSMGraph（视觉语义记忆图）**: 将原始 RGB 视图作为"一等记忆"保存，用轻量语义线索（CLIP 房间标签、物体可见性标签）和拓扑关系组织成 room-view-object 层级结构，保留细粒度视觉证据用于消歧与 Stop 验证，避免检测器中心场景图的信息压缩与噪声累积。
2. **预算化粗到精导航策略（Budgeted Coarse-to-Fine）**: 粗阶段（Explore）通过房间感知分桶 + Top-K 压缩候选空间，仅在精阶段（Search + Verify）对小候选集调用 [[VLM]] 进行选择与多视图 Stop 验证，大幅降低昂贵 VLM 调用。
3. **反思驱动持续记忆适应（RDCMA）**: 在每个子任务后将交互结果回写为目标条件、图附着的先验（Episode-STM / Scene-LTM），无需重训练即实现自进化导航。

---

## 问题背景

### 要解决的问题
在家庭助手、服务机器人等场景中，移动智能体需要理解指令、感知环境并导航到目标物体。其中**记忆系统**是关键模块，使智能体能进行长时程推理而非仅依赖逐帧短期响应。如何为零样本（无需训练）具身导航构建高效、细粒度且能自我进化的记忆，是本文核心问题。

### 现有方法的局限
论文将现有记忆方案分为三类并指出各自缺陷：

- **检测器中心的[[Scene Graph|场景图]]记忆**：(i) 将观测压缩为稀疏符号节点，丢弃纹理、空间布局等细粒度线索；(ii) 对检测噪声敏感，噪声会沿下游推理累积；(iii) 阻止强大 VLM 直接对多视图图像推理。
- **基于 3D 重建的记忆**（如 [[3D-Mem]]）：运行时计算开销高，且与 VLM 缺乏原生兼容性。
- **基于图像的拓扑记忆**：保留丰富视觉证据，但 (i) 图像松散地组织在缓存中，缺乏对房间/前沿/可达性的结构化表示，导致同类物体上频繁过早终止（premature stop）；(ii) 记忆缺乏进化，成功与失败不会反馈为可复用状态，导致重复犯错。

### 本文的动机
作者认为应当**保留原始多视图图像作为一等记忆**（不压缩到物体节点），同时用轻量语义标签做索引/分桶；高层决策始终基于多视图图像证据（grounded in image evidence）而非检测器输出或稠密 3D 重建。在此之上引入预算化决策降低 VLM 成本，并通过反思回写实现免训练的持续自我改进。

---

## 方法详解

### 模型架构

EvoMemNav 采用 **图像锚定的记忆图 + 预算化粗到精控制器** 架构：

- **输入**: 多模态目标 $g$（物体标签 / 文本描述 / 参考图像）+ 带位姿的 RGB-D 观测流 $I_t = (I_t^{rgb}, I_t^{depth}, p_t)$，其中位姿 $p_t = (x_t, y_t, \theta_t)$
- **记忆**: [[Visual-Semantic Memory Graph]]（VSMGraph）$\mathcal{G}_t$，room-view-object 层级；视图节点存储原始观测与位姿作为一等图像证据
- **感知模块**: [[YOLOv8]]-World + [[SAM|Segment Anything]] 提供实例假设与可见性标签（仅作软线索）；[[CLIP]] argmax 给视图打离散房间标签
- **核心模块**: [[Coarse-to-Fine|粗到精]]策略（Explore → Search + Verify）+ [[RDCMA|反思驱动持续记忆适应]]
- **VLM**: [[Qwen3-VL]]-8B 同时用于选择与验证阶段
- **输出**: 下一目标视图节点 + Stop / Reselect 决策

> 决策始终 grounded 在 VSMGraph 上的多视图图像证据，物体节点仅用于索引和分桶，不压缩记忆。

### 核心模块

#### 模块1: VSMGraph 视觉语义记忆图

**设计动机**: 用结构化的图组织原始视图，既保留细粒度证据，又支持高效检索与候选压缩。

**具体实现** —— 图定义 $\mathcal{G}_t = (\mathcal{R}_t, \mathcal{V}_t, \mathcal{O}_t, \mathcal{E}_t)$：

- **房间节点 $\mathcal{R}_t$**: 索引同一区域内的视图与物体，支持区域级检索
- **视图节点 $\mathcal{V}_t$**: 存储图像证据与位姿；划分为两类——
  - **[[Anchor View|锚点视图]] $\mathcal{V}_{A,t}$**: 物体丰富的已探索观测区域，用于验证
  - **[[Frontier View|前沿视图]] $\mathcal{V}_{F,t}$**: 可达边界，用于探索
- **物体节点 $\mathcal{O}_t$**: 存储实例假设（类别 + 3D 定位）
- **边 $\mathcal{E}_t$**: (i) 连续无碰撞视点间的**可导航边（navigability）**；(ii) 连接视图与其观测到的物体节点的**可见性边（visibility）**
- **在线更新**: 每访问一个位姿即添加视图节点并与前一视图连可导航边；[[Occupancy Grid|占用栅格]]作为度量骨架，最短路径在栅格上计算并沿视图边执行，高层推理与检索在 $\mathcal{G}_t$ 上进行
- **房间标签**: 每个视图用 CLIP argmax 与常见房间类型匹配，得到离散房间标签 $\nu \in T_{room}$，作为免阈值的上下文标签用于房间感知分桶

> 物体缓存 $O_{map}$ 仅为视图节点标注轻量线索（visibility tags $O_t^{vis}$），不定义 VSMGraph；抑制这些标签只会降低候选排序/探索效率，不会破坏闭环控制器，因为 Stop 验证始终基于原始多视图图像。

#### 模块2: 粗到精导航策略（Budgeted Coarse-to-Fine）

**设计动机**: 常见零样本流水线对庞大且增长的候选池查询 VLM，且一旦选中锚点视图就立即停止——这导致 (i) 随记忆规模扩展性差，(ii) 缺乏比较探索前沿的结构，(iii) 易出现同类错误实例与过早停止。本模块将候选压缩与细粒度 VLM 推理分离。

**具体实现**:

- **粗阶段（Explore）——候选压缩与路由**:
  - 构造紧凑候选集 $C_t^A \subseteq \mathcal{V}_{A,t}$（锚点）与 $C_t^F \subseteq \mathcal{V}_{F,t}$（前沿），预算约束 $|C_t^A| \le K_A$、$|C_t^F| \le K_F$
  - 通过软约束下的**房间感知分桶**压缩候选，使用离散房间标签 $\nu$ 与来自 $O^{vis}$ 的物体命中标签
  - 可选地对同一 VLM 做单次纯文本查询，获得粗粒度聚焦提示（仅作软线索）
  - 若压缩后锚点集为空（$C_t^A = \varnothing$），直接路由到前沿，**无需查询 VLM**
- **精阶段（Search + Verify）——小集选择与 Stop 验证**:
  - **Search**: 仅对紧凑集 $C_t = C_t^A \cup C_t^F$ 查询 VLM（见公式1）
  - 若选中锚点视图但置信度低（$\tau_t \in \{uncertain, unknown\}$），则**门控回退到探索**，路由到前沿，减少同类错误实例
  - **Verify**: 到达选中锚点视图后**不立即停止**，而是调用结构化 Stop 验证，返回 STOP 或 RESELECT；RESELECT 触发锚点视图冷却并重新进入粗阶段，形成处理过早停止/错误实例的鲁棒闭环
- **Recover 兜底**: 仅当重复 reselect 或 cooldown 指示死锁时触发，临时强制仅前沿探索

#### 模块3: RDCMA 反思驱动持续记忆适应

**设计动机**: 终身多子任务设定允许智能体复用同一场景早期累积的证据。除在线增长的 VSMGraph 结构外，需要一种免训练机制将交互结果写回为可复用先验。

**具体实现**:

- **两个时间尺度的先验**:
  - **[[Episodic Memory|Episode-STM]]**: 捕捉当前 episode 内的短期抑制与防回环线索，episode 结束时重置
  - **Scene-LTM**: 每场景持久化，支持跨子任务/episode 的持续复用，保守更新
  - 两者均由目标签名 $s_g$（如模态和/或目标类别）索引，并支持粗到精的回退以避免键过于稀疏
- **图附着先验**: 关联到房间节点、锚点视图、前沿视图，存储为紧凑统计量（支持计数、可靠性/风险估计）；**不更新任何模型参数**，只持续更新这些轻量统计量
- **经验记忆来源（反思）**: 反思在 Stop 验证时被调用（决定 STOP vs RESELECT），每个子任务后聚合并回写——综合 (i) VSMGraph 上的轨迹事件（访问房间、选中的前沿/锚点视图、重访、死端/回环指标）与 (ii) 结构化 Stop 验证结果
- **经验检索与使用（两个耦合通道）**:
  - **探索排序先验（Exploration Ordering Priors）**: 在 Explore 中作为保守的 tie-breaker，使前沿选择偏向历史效用高、死端风险低的区域，锚点选择偏向相似目标下提供过可靠证据的视图；不覆盖粗分桶约束，只在有界候选集内细化排序
  - **停止校准先验（Stop Calibration Priors）**: 在 Verify 中校准结构化 Stop 决策，为每个候选锚点视图维护 $s_g$ 下的 Stop 可靠性统计，并将紧凑的 stop-memory 提示注入验证 prompt，直接针对过早停止与错误实例失败
- **维护机制**: 指数衰减（陈旧先验）、每签名小容量上限（保留 Top-K 最有用条目）、对反复被反驳先验的保守抑制；这些操作仅影响附着统计量，不引入新决策阈值

**补充解读：两层记忆按什么形式保存**

> 核心：STM / LTM 存的不是图像或经历，也不动任何模型参数，而是**以目标签名 $s_g$ 为索引、附着在 VSMGraph 节点（房间 / 锚点视图 / 前沿视图）上的紧凑统计量**（计数、可靠性 / 风险估计）。可类比为贴在图节点上的"小便签"：STM 是铅笔写的（用完即擦），LTM 是钢笔写的（长期保留、改动保守）。

- **形态**：图附着（graph-attached）的轻量统计量，不另开独立存储；只持续刷新数字，**不更新 YOLO/CLIP/VLM 任何权重**——这是"免训练"的根本。
- **索引键**：目标签名 $s_g$（模态 / 目标类别）分桶；键过稀疏时支持**粗到精回退**（从具体类别退回模态层级）取经验。
- **两层差异（数据形式相同，仅生命周期 + 更新策略不同）**：

  | | Episode-STM | Scene-LTM |
  |---|---|---|
  | 活多久 | 当前 episode 内 | 跟随场景持久 |
  | 何时清 | episode 结束即重置 | 不重置，跨子任务复用 |
  | 更新风格 | 即时，短期抑制 / 防回环 | 保守更新 |
  | 解决什么 | 本轮防原地打转 | 同场景下次找别的物体仍可复用 |

- **读出使用**：作为先验喂回找物流程——**探索排序先验**在 Explore 当 tie-breaker（偏向高效用 / 低死端风险候选，不覆盖粗分桶硬约束）；**停止校准先验**在 Verify 把紧凑 stop-memory 提示注入验证 prompt，治过早停止与错误实例。

---

## 关键公式

### 公式1: [[Coarse-to-Fine|Search 阶段 VLM 查询]]

$$
(a_t, \tau_t) = \mathrm{VLM}(g, C_t), \quad \tau_t \in \{certain,\ uncertain,\ unknown\}
$$

**含义**: 精阶段仅对压缩后的紧凑候选集 $C_t = C_t^A \cup C_t^F$ 查询 VLM，输出下一动作/选择 $a_t$ 及置信度 $\tau_t$。这是全流程中唯一对多候选调用 VLM 的环节，实现"预算化"。

**符号说明**:
- $g$: 多模态目标（物体标签/文本/图像）
- $C_t = C_t^A \cup C_t^F$: 粗阶段压缩后的紧凑候选集（锚点 + 前沿）
- $a_t$: VLM 选择的下一目标视图/动作
- $\tau_t$: 置信度等级；当为 $uncertain$ 或 $unknown$ 时门控回退到前沿探索

### 公式2: [[Anchor View|锚点视图定义]]

$$
v_t^A = (I_t^{rgb},\ p_t,\ O_t^{vis},\ r_t^{room})
$$

**含义**: 锚点视图捕捉物体丰富的已探索观测，存储原始 RGB、位姿、可见物体标签与房间标签，作为 Stop 验证的一等图像证据。

**符号说明**:
- $I_t^{rgb}$: 该视图的原始 RGB 图像
- $p_t = (x_t, y_t, \theta_t)$: 视图位姿
- $O_t^{vis}$: 该视图关联的可见物体假设/标签（来自物体缓存 $O_{map}$，仅作软线索）
- $r_t^{room}$: 该视图的 CLIP 房间标签

### 公式3: [[Frontier View|前沿视图定义]]

$$
F_f = (r_f,\ p_f,\ I_f^{front},\ f^{room})
$$

**含义**: 前沿视图表示可达探索边界。在自顶向下占用栅格 $M_t$ 上，自由格若邻接至少一个未知格即为前沿格，前沿格聚类成前沿区域；对每个区域选可达入口位姿并拍摄前沿朝向图像。

**符号说明**:
- $r_f$: 栅格上的前沿区域
- $p_f$: 该前沿区域的可达入口位姿
- $I_f^{front}$: 前沿朝向图像
- $f^{room}$: 该前沿的 CLIP 房间标签
- 每个前沿视图连接到最近的已探索视图（及其房间节点），使未探索区域作为边界节点出现在 $\mathcal{G}_t$ 的外围

### 公式4: [[Visual-Semantic Memory Graph|VSMGraph 图定义]]

$$
\mathcal{G}_t = (\mathcal{R}_t,\ \mathcal{V}_t,\ \mathcal{O}_t,\ \mathcal{E}_t), \quad \mathcal{V}_t = \mathcal{V}_{A,t} \cup \mathcal{V}_{F,t}
$$

**含义**: VSMGraph 是 room-view-object 层级图；视图节点划分为锚点视图与前沿视图两个不相交子集。

**符号说明**:
- $\mathcal{R}_t$: 房间节点集（区域级检索）
- $\mathcal{V}_t$: 视图节点集（图像证据 + 位姿）
- $\mathcal{O}_t$: 物体节点集（实例假设：类别 + 3D 定位）
- $\mathcal{E}_t$: 边集（可导航边 + 可见性边）

---

## 关键图表

### Figure 1: Overview of EvoMemNav / 系统概览

![[EvoMemNav_fig1_overview.png]]

**说明**: EvoMemNav 总览。(A) [[Visual-Semantic Memory Graph|VSMGraph]] 将记忆组织成 room-view-object 图，视图节点为一等公民，物体/房间线索用于检索；(B) 预算化[[Coarse-to-Fine|粗到精]]策略通过分桶 + Top-K 压缩候选，仅对 shortlist 调用 [[VLM]] 做多视图 Stop 验证（图中"find cabinet → fridge is likely there → STOP"）；(C) 反思驱动回写更新图附着先验（Episode-STM / Scene-LTM）实现持续自进化。

### Figure 2: Framework of EvoMemNav / 整体框架

![[EvoMemNav_fig2_framework.png]]

**说明**: EvoMemNav 维护图像锚定的 VSMGraph（room-view-object 记忆图，视图节点存储原始图像证据）。预算化粗到精控制器 shortlist 候选并仅对 shortlist 查询 VLM 做目标决策与多视图图像 Stop 验证。[[RDCMA]] 在每个子任务后更新图附着先验，实现免重训练的自进化导航。

### Figure 3: VSMGraph Construction and Attributes / VSMGraph 构建与属性

![[EvoMemNav_fig3_vsmgraph.png]]

**说明**: 左：智能体在[[Occupancy Grid|占用栅格]]上探索时，沿轨迹添加视图节点、在可达边界添加前沿视图节点；物体实例维护在物体中心地图 $O_{map}$ 并通过可见性关系链接到视图。右：得到的 VSMGraph 组织成 room-view-object 层级，每个视图节点存原始图像证据与位姿，并标注离散 CLIP 房间类型标签用于房间感知分桶。实线边表示可导航性，虚线边表示可见性。

### Figure 4: Reflection-Driven Continual Memory Adaptation (RDCMA) / 反思驱动持续记忆适应

![[EvoMemNav_fig4_rdcma.png]]

**说明**: VSMGraph 的免训练、图附着先验机制。轨迹事件与 Stop 结果（STOP / RESELECT）在目标签名 $s_g$ 下更新紧凑先验，存为 Episode-STM（重置）与 Scene-LTM（持久），带衰减 / Top-K / 冲突抑制。目标条件查找将先验用作 Explore 阶段的 tie-breaker 与 Verify 阶段的 stop 提示，无任何参数更新。

### Figure 5: Impact of Reflection-Driven Experience Memory / 反思经验记忆的子任务级影响

![[EvoMemNav_fig5_subtask_curves.png]]

**说明**: GOAT-Bench VAL-UNSEEN 上的子任务级 SR (a) / SPL (b) 曲线。阴影区（子任务 3-5）标出 episode 内经验累积带来最大增益的区间。对比 baseline、VSMGraph+C2F（无 RDCMA）、EvoMemNav（含 RDCMA）：VSMGraph+C2F 一致优于 baseline；启用 RDCMA 后增益随累积经验递进，在中段子任务峰值（此时 Stop 验证与探索反馈已充分更新图附着先验），早期子任务的改进则说明场景级先验的有效复用。

### Figure 6: Qualitative Comparison / 定性对比

![[EvoMemNav_fig6_qualitative.png]]

**说明**: GOAT-Bench 上与 baseline 的定性对比，从左到右为文本、图像、物体目标子任务（如"refrigerator in the kitchen, located next to the kitchen cabinet and the worktop"）。EvoMemNav 从 VSMGraph 检索紧凑候选、在自顶向下地图上规划、并验证 Stop 决策，避免了非结构化记忆 baseline 中常见的错误实例选择与目标遗漏（target missing）。

### Table 1: GOAT-Bench Val Unseen 对比

| Method | Training-free | SR | SPL |
|--------|:---:|:---:|:---:|
| SenseAct-NN Monolithic [20] | | 12.3 | 6.8 |
| Modular CLIP on Wheels [20] | ✓ | 16.1 | 10.4 |
| Modular GOAT [20] | | 24.9 | 17.2 |
| SenseAct-NN Skill Chain [20] | | 29.5 | 11.3 |
| VLMnav [12] | ✓ | 20.1 | 9.6 |
| DyNaVLM [18] | ✓ | 25.5 | 10.2 |
| TANGO [28] | ✓ | 32.1 | 16.5 |
| 3D-Mem [46] | ✓ | 42.6 | 22.8 |
| MTU3D [61] | | 47.2 | 27.7 |
| MSGNav [17] | ✓ | 52.0 | 29.6 |
| **EvoMemNav (Ours)** | ✓ | **59.6** | **38.9** |

**说明**: EvoMemNav 在 GOAT-Bench VAL-UNSEEN 达到 59.6 SR / 38.9 SPL，超越所有先前 training-free 基线，且使用开源 VLM 主干、无任务特定训练，说明增益来自记忆组织与预算化决策策略。（注：原文表格部分行的 training-free 标记与数值排版较乱，此处按数值列对齐整理，EvoMemNav 为最优。）

### Table 2: HM3D ObjectGoal 对比

| Method | Training-free | HM3Dv1 SR | HM3Dv1 SPL | HM3Dv2 SR | HM3Dv2 SPL |
|--------|:---:|:---:|:---:|:---:|:---:|
| ZSON [26] | ✓ | 25.5 | 12.6 | – | – |
| PixNav [2] | ✓ | 37.9 | 20.5 | – | – |
| VLFM [50] | ✓ | – | – | 63.6 | 32.5 |
| ESC [58] | ✓ | 39.2 | 22.3 | – | – |
| L3MVN [51] | ✓ | 50.4 | 23.1 | 36.3 | 15.7 |
| InstructNav [24] | ✓ | 58.0 | 20.9 | – | – |
| GAMap [16] | ✓ | 53.1 | 26.0 | – | – |
| SG-Nav [48] | ✓ | 54.0 | 24.9 | 49.6 | 25.5 |
| UniGoal [49] | ✓ | 54.5 | 25.1 | – | – |
| WMNav [27] | ✓ | 58.1 | 31.2 | – | – |
| TriHelper [52] | ✓ | 56.5 | 25.3 | – | – |
| **EvoMemNav (Ours)** | ✓ | **59.2** | **33.6** | **63.8** | **39.4** |

**说明**: EvoMemNav 在 HM3Dv1 达 59.2/33.6，HM3Dv2 达 63.8/39.4，确认方法可迁移到大规模物体目标基准。（注：原文 Table 2 数值与方法行排版严重错位，此处依据正文 "59.2/33.6 on HM3Dv1 and 63.8/39.4 on HM3Dv2" 锚定 EvoMemNav 的最终数值并尽量对齐其余行，部分基线数值的精确归属以原文为准。）

### Table 3: 组件消融（GOAT-Bench VAL-UNSEEN，每场景 episode index=1，36 场景）

| VSMGraph | Coarse | Fine | RDCMA | Object SR/SPL | Language SR/SPL | Image SR/SPL | Overall SR/SPL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| – | – | – | – | 49.5 / 27.5 | 34.1 / 18.4 | 35.2 / 19.5 | 39.9 / 22.0 |
| ✓ | – | – | – | 53.5 / 31.3 | 42.9 / 26.7 | 44.3 / 32.2 | 47.1 / 30.1 |
| ✓ | ✓ | – | – | 70.7 / 46.0 | 54.2 / 34.3 | 49.7 / 38.3 | 58.7 / 39.7 |
| ✓ | – | ✓ | – | 66.7 / 34.2 | 53.9 / 31.8 | 50.0 / 38.4 | 57.2 / 34.7 |
| ✓ | ✓ | ✓ | – | 72.7 / 48.3 | 48.4 / 30.4 | 60.2 / 43.3 | 60.8 / 40.8 |
| ✓ | ✓ | ✓ | ✓ | **75.8 / 49.5** | 53.9 / 33.2 | **63.6 / 46.2** | **64.8 / 43.1** |

**关键发现**:
- 从 baseline 39.9/22.0 起步；加入 VSMGraph 提升至 47.1/30.1，验证结构化视觉语义记忆有效。
- 启用 Coarse（Explore）带来最大单项提升（58.7/39.7），**预算化候选压缩是首要性能驱动**。
- 加入 Fine（Search+Verify）细化至 60.8/40.8，显著改善图像目标（缓解过早停止）。
- 最后 RDCMA 达峰值 64.8/43.1（Object 75.8 SR，Image 63.6 SR），证明免训练自适应有效互补粗到精策略。
- 禁用 Coarse 时回退到同预算的朴素 Top-K 启发式；禁用 Fine 时智能体到达选中视图即立即停止。

### Table 4: 性能-效率分析（GOAT-Bench VAL-UNSEEN，36 场景，每子任务均值(std)）

| Method | SR | VLM Calls | VLM Tokens | Steps | VLM Time(s) | Graph Time(s) | Total(s) |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3D-Mem [46] | 49.6 | 10.7 (15.3) | 14.5k (18.3k) | 7.0 (7.7) | 14.0 (18.5) | 88.2 (81.1) | 102.2 |
| VSMGraph+C2F | 60.8 | 6.3 (1.9) | 18.9k (6.3k) | 12.4 (6.4) | 25.5 (7.4) | 37.3 (21.6) | 62.8 |
| **EvoMemNav (Ours)** | **64.8** | 6.5 (2.3) | 19.4k (7.1k) | 13.3 (7.6) | 21.1 (6.3) | 37.6 (19.9) | **58.7** |

**关键发现**: 相比 3D-Mem，VSMGraph+C2F 减少 VLM 调用 41%（10.7→6.3）、图维护 58%（88.2→37.3s）、总代理运行时 39%（102.2→62.8s），同时 SR +11.2（49.6→60.8）。EvoMemNav 进一步将 SR 提升到 64.8、运行时降到 58.7s，仅边际开销——说明 RDCMA 以可忽略代价带来增益。结构化验证略增 token，但因 VLM 调用更少、图操作更轻而整体效率提升；步数适度增加反映验证驱动的 reselection 有效避免过早终止。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[GOAT-Bench]] | 36 场景 × 10 episode，每 episode 5-10 子任务 | 多模态终身导航，目标为类别名/语言描述/图像，开放词汇 | 主测试 |
| [[HM3D]] v1 | 20 场景，2000 episode，6 物体类别 | 大规模 3D 室内物体目标导航 | 泛化测试 |
| [[HM3D]] v2 | 36 场景，1000 episode，6 物体类别 | 同上 | 泛化测试 |

### 实现细节

- **仿真器**: [[Habitat]]-Lab，所有组件用 PyTorch 实现
- **感知**: [[YOLOv8]]-World + [[SAM|Segment Anything]] 提供实例假设与可见性标签（仅作分桶/验证软线索，核心记忆仍为图像锚定视图证据）
- **VLM**: [[Qwen3-VL]]-8B（选择 + 验证两阶段共用）
- **房间标签**: [[CLIP]] argmax 匹配常见房间类型
- **评估指标**: [[Success Rate|SR]]（停在目标 1m 内即成功）+ [[SPL]]（按探索距离加权的成功分数）
- **效率实验设置**: ablation 与 runtime profiling 用固定子集（36 场景各取 episode index=1）；主结果遵循完整 GOAT-Bench 协议

### 可视化结果
定性实验（Figure 6）显示，借助 VSMGraph 检索与拓扑规划，EvoMemNav 在文本/图像/物体三类目标上均避免了非结构化记忆 baseline 普遍存在的错误实例选择与目标遗漏。

---

## 批判性思考

### 优点
1. **记忆设计的核心洞察清晰**: "保留原始视图为一等记忆、物体节点仅作索引"直击场景图压缩信息与图像缓存缺乏结构两类方法的痛点。
2. **预算化设计兼顾性能与效率**: Table 4 显示在 SR 大幅提升的同时 VLM 调用与运行时显著下降，对实际部署友好；消融明确指出 Coarse 压缩是首要驱动。
3. **免训练自进化**: RDCMA 不更新任何参数即实现跨子任务的持续改进，且通过衰减/Top-K/冲突抑制控制记忆增长，工程上务实。
4. **全开源栈**: 使用开源 Qwen3-VL-8B 而非闭源 VLM，可复现性好；代码已开源。

### 局限性
1. **依赖感知模块质量**: 物体命中标签来自 YOLOv8-World + SAM，虽仅作软线索，但分桶/排序质量仍受其影响，极端开放词汇场景下可能退化。
2. **RDCMA 的可迁移性受限**: Scene-LTM 仅按场景持久化，跨场景的知识迁移未涉及；终身收益主要体现在同场景多子任务（Figure 5 中段）。
3. **大量实现细节缺失**: 正文未给出 $K_A, K_F$ 的具体取值、分桶/排序的精确打分函数、Stop 验证 prompt 模板，这些"deferred to Supplementary Material"，正文可读性偏向 high-level。
4. **表格排版混乱**: Table 1/2 在 PDF 中数值与方法行严重错位，给精确核对带来困难。

### 潜在改进方向
1. 引入跨场景的元先验（meta-prior）以加速新场景的早期子任务。
2. 端到端学习候选分桶/排序打分，替代手工软约束。
3. 将 VSMGraph 与轻量 3D 几何融合，进一步提升空间推理（当前刻意避免稠密重建）。

### 可复现性评估
- [x] 代码开源（github.com/caicaiya123/EvoMemNav）
- [ ] 预训练模型（使用现成 Qwen3-VL-8B / YOLOv8-World / SAM，无需自训）
- [ ] 训练细节完整（核心超参与 prompt 在补充材料）
- [x] 数据集可获取（GOAT-Bench / HM3D 公开）

---

## 关联笔记

### 基于
- [[3D-Mem]]: 多视图快照 3D 场景记忆（主要对比基线，EvoMemNav 在其基础上去除 3D 重建开销）
- [[Frontier Exploration]]: 前沿探索是粗阶段路由的基础
- [[Topological Map]]: 图像拓扑图记忆的思想来源

### 对比
- [[MTU3D]]: 桥接视觉 grounding 与探索的具身导航
- [[MSGNav]]: 多模态 3D 场景图零样本导航（GOAT-Bench 上的强基线）
- [[VLFM]]: 视觉语言前沿地图（HM3D 对比）
- [[ESC]]: 软常识约束的零样本物体导航
- [[TANGO]]: 可通行性感知的拓扑目标导航
- [[Instruct-Nav]]: 通用指令导航零样本系统
- [[MapGPT]]: 地图引导提示的统一 VLN

### 方法相关
- [[Visual-Semantic Memory Graph]]: 本文核心记忆结构
- [[Coarse-to-Fine]]: 预算化两阶段决策策略
- [[RDCMA]]: 反思驱动持续记忆适应
- [[Anchor View]]: 物体丰富的验证视图节点
- [[Frontier View]]: 可达探索边界视图节点
- [[Scene Graph]]: 对比的检测器中心记忆范式
- [[Occupancy Grid]]: 度量骨架
- [[Reflexion]]: LLM 反思机制的相关思想来源
- [[Qwen3-VL]]: 选择与验证阶段的 VLM
- [[CLIP]]: 房间标签匹配
- [[YOLOv8]]: 实例假设
- [[SAM|Segment Anything]]: 实例分割

### 硬件/数据相关
- [[GOAT-Bench]]: 多模态终身导航基准
- [[HM3D]]: Habitat-Matterport 3D 数据集
- [[Habitat]]: 仿真平台
- [[Object Navigation]]: 任务范式
- [[Success Rate]] / [[SPL]]: 评估指标

---

## 速查卡片

> [!summary] EvoMemNav: Efficient Self-Evolving Fine-Grained Memory for Zero-Shot Embodied Navigation
> - **核心**: 以原始视图为一等记忆的 VSMGraph + 预算化粗到精策略 + 免训练反思回写（RDCMA），实现高效自进化零样本导航
> - **方法**: room-view-object 记忆图（锚点/前沿视图）→ Explore 房间感知分桶压缩候选 → Search+Verify 仅对 shortlist 调 VLM 做多视图 Stop 验证 → RDCMA 回写 Episode-STM/Scene-LTM 先验
> - **结果**: GOAT-Bench 59.6 SR / 38.9 SPL，HM3Dv2 63.8/39.4；相比 3D-Mem VLM 调用 -41%、运行时 -39% 且 SR +11.2
> - **代码**: https://github.com/caicaiya123/EvoMemNav

---

*笔记创建时间: 2026-06-05*
