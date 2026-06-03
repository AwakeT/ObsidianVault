---
title: "TrajRAG: Retrieving Geometric-Semantic Experience for Zero-Shot Object Navigation"
method_name: "TrajRAG"
authors: [Yiyao Wang, Sixian Zhang, Keming Zhang, Xinhang Song, Songjie Du, Shuqiang Jiang]
year: 2026
venue: CVPR 2026
tags: [object-navigation, zero-shot-navigation, retrieval-augmented-generation, lifelong-memory, semantic-map, topological-map]
image_source: online
arxiv_html: https://arxiv.org/html/2605.01700v1
created: 2026-05-20
---

# 论文笔记：TrajRAG: Retrieving Geometric-Semantic Experience for Zero-Shot Object Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | State Key Laboratory of AI Safety, Institute of Computing Technology, CAS; University of Chinese Academy of Sciences |
| 日期 | May 2026 |
| 项目主页 | N/A |
| 对比基线 | [[VLFM]], [[SG-Nav]], [[UniGoal]], [[ApexNAV]], [[BeliefMapNav]], [[ESC]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.01700) / [HTML](https://arxiv.org/html/2605.01700v1) |

---

## 一句话总结

> 提出面向 zero-shot ObjectNav 的轨迹级 RAG 记忆库，用拓扑-极坐标轨迹压缩历史导航经验，并通过粗到细检索把相似几何-语义经验注入 LLM 规划器，提升开放词汇目标导航效率。

---

## 核心贡献

1. **Trajectory RAG 框架**: 将过往导航 episode 转化为可检索的几何-语义经验，弥补 LLM/VLM 仅依赖互联网文本常识、缺少 embodied 3D experience 的问题
2. **Topo-polar trajectory 表示**: 通过可通行区域骨架化提取拓扑节点，并在每个节点周围用极坐标扇区记录语义布局，实现紧凑、可匹配的轨迹记忆
3. **层次化 chunking 与粗到细检索**: 将相似 topo-polar 轨迹合并成 summary 作为 coarse index，再用轨迹编码器做 fine-grained retrieval，提高实时检索效率和相关性
4. **持续经验积累**: episode 结束后把新轨迹整合进 TrajRAG，支持 lifelong navigation memory，而不是每个 episode 后丢弃观测
5. **跨数据集泛化**: 在 MP3D、HM3Dv1、HM3Dv2 上取得 zero-shot ObjectNav SOTA，并显示 HM3D/MP3D 构建的经验库可互相迁移

---

## 问题背景

### 要解决的问题

[[Object Navigation|ObjectNav]] 要求智能体在未知环境中根据目标类别（如 chair、bed）找到目标。Zero-shot 方法通常借助 [[LLM]] / [[VLM]] 的常识推理，但这些常识主要来自网页文本，不包含真实 embodied 3D 导航经验。

### 现有方法的局限

- 单步 LLM/VLM 方法只看当前观测，容易重复探索、反复访问已知区域
- episodic memory 方法虽然把当前 episode 的语义地图、scene graph 或 similarity map 组织进 prompt，但 episode 结束后通常丢弃，无法积累长期经验
- 现有 embodied RAG 多数检索当前场景内部信息，缺少跨 episode、跨场景的历史导航经验迁移
- 纯文本轨迹描述难以表达连续路径和空间布局；scene graph 又缺少轨迹连续性和可操作的路径信息

### 本文的动机

人类导航依赖短期记忆和长期经验：当前环境细节用于即时决策，过去相似场景和路径经验用于判断哪个方向更可能接近目标。TrajRAG 试图把这种长期导航经验做成可检索、可更新、可供 LLM 规划器使用的外部记忆。

---

## 方法详解

### 模型架构

TrajRAG 是一个 **RAG-style navigation memory system**：

- **输入**: 历史 RGB-D 轨迹、agent pose、目标类别、当前 episode 的语义地图
- **在线感知**: [[Grounding-DINO]] 检测开放词汇类别，[[MobileSAM]] 精化 mask，构建语义地图
- **记忆表示**: topo-polar trajectory chunk
- **检索结构**: coarse topo-polar summary + fine trajectory embedding
- **规划器**: Qwen3-32B 作为 LLM planner
- **低层控制**: frontier waypoint + A* / local policy
- **输出**: 下一个探索 waypoint 或局部动作

整体流程：

```text
RGB-D observations + pose
  -> open-vocabulary semantic map
  -> skeletonized topology
  -> topo-polar trajectory
  -> candidate frontier trajectories
  -> TrajRAG coarse-to-fine retrieval
  -> retrieved geometric-semantic experience
  -> LLM planner selects waypoint
  -> local policy executes action
```

### 核心模块

#### 模块1: Topo-polar Trajectory

**设计动机**: 原始 RGB-D 轨迹存在大量冗余，包括局部重复、小幅运动变化、跨 episode 空间重叠。需要一种既紧凑又保留几何-语义布局的表示。

**具体实现**:

- 从 RGB-D 观测和 pose 构建 open-vocabulary semantic map
- 对 explored/navigable region 做 morphological thinning，得到一像素宽 skeleton
- 选择 8-neighborhood 中连通分量数不少于 3 的 skeleton 像素作为候选拓扑节点
- 用非极大距离抑制合并相邻冗余节点
- 每个节点周围按极坐标划分扇区，论文使用 30°/sector，即 12 个方向扇区
- 每个扇区沿 ray 搜索第一个非 free pixel，记录 object / obstacle / unknown 等语义
- 对同一节点的连续访问进行合并；对长程 loop，只保留节点最后一次出现，删除中间环路

得到的 topo-polar trajectory 同时包含：

- 离散拓扑节点序列
- 节点间有向边
- 每个节点周围的相对几何-语义布局
- 节点在 episode world frame 中的位置，用于后续几何对齐

#### 模块2: TrajRAG Hierarchical Chunking

**设计动机**: 历史轨迹数量很大，直接全库检索会慢且容易返回不相关经验。

TrajRAG 采用三层结构：

1. **Chunk**  
   每个 chunk 对应一条 topo-polar trajectory，包含轨迹结构、语言描述和轨迹 embedding。

2. **Coarse index**  
   几何-语义相似的 chunks 被分组并合并成 topo-polar summary。summary 作为组级 descriptor，表示一类相似空间布局。

3. **Fine index**  
   组内每条 trajectory 用轨迹编码器转成 embedding，支持快速 top-k 检索。

#### 模块3: Trajectory Encoder

**设计动机**: 通用 text embedding 不擅长表达轨迹顺序、路径连续性和拓扑结构。

**具体实现**:

- 每个 topo-polar node 由 sector vector 表示
- 使用 encoder-only transformer（如 DistilBERT）得到节点 embedding；该部分使用预训练权重并冻结
- 使用 decoder-only transformer（如 DistilGPT2）编码节点序列的顺序依赖
- 将最终 token 表示与 goal semantic embedding 拼接，得到 trajectory embedding
- 用 contrastive learning 训练：同组轨迹或相同 navigation goal 为正样本，不同组随机采样为负样本

#### 模块4: Incremental Construction

**设计动机**: 新 episode 结束后，需要判断它是否是已有经验的重复，还是能补充新结构。

**语义匹配**:

- 计算新轨迹节点与 summary 节点的 sector vector 相似度
- 对 12 维 sector vector 做 cyclic rotation，补偿 agent 朝向差异
- 使用 mutual KNN 过滤伪匹配

**几何匹配**:

- 使用匹配节点的 world coordinates
- 通过 [[RANSAC]] 估计新轨迹到 summary 的几何变换
- 以 inlier ratio 作为几何相似度

**冗余检查**:

- 如果两条轨迹目标相同，且节点序列存在严格包含关系，较短轨迹视为冗余并丢弃
- 否则把新轨迹节点和边融合进 summary graph
- 若无法匹配任何已有组，则新建 group

#### 模块5: Navigation with TrajRAG

在线导航时：

1. 根据实时 RGB-D 观测维护 semantic map
2. 从 explored / obstacle map 中提取 frontiers
3. 对每个 frontier，用 BFS 在当前 topo-polar trajectory 上生成候选路径
4. 每条候选路径作为 query，在 TrajRAG 中先检索相似 summary，再检索组内相似 trajectory chunks
5. 把 candidate paths 和 retrieved experiences 一起提供给 LLM planner
6. LLM 选择下一 waypoint，local policy 用 A* 到达该 waypoint

---

## 关键公式

### 公式1: Topo-polar 扇区表示

```text
node -> 12 directional sectors -> first non-free semantic label per sector
```

**含义**: 用节点周围的相对空间-语义分布作为局部布局 fingerprint，避免直接依赖绝对坐标。

### 公式2: 粗到细检索

```text
candidate trajectory
  -> match topo-polar summaries by semantic + geometric consistency
  -> retrieve top-k trajectory chunks by learned sequence embedding
  -> return language descriptions as navigation experience
```

**含义**: 先用结构过滤大范围不相关经验，再在局部候选组中做 embedding 检索。

### 公式3: 轨迹对比学习

```text
positive: same topological group or same navigation goal
negative: different groups
similarity: cosine similarity with temperature
```

**含义**: 训练轨迹编码器，使相似布局/相似目标轨迹在 embedding 空间中靠近。

---

## 实验

### 实验设置

| 项目 | 内容 |
|------|------|
| 任务 | zero-shot ObjectNav |
| 数据集 | [[HM3D]]v1, HM3Dv2, [[MP3D]] |
| 仿真器 | [[Habitat]] |
| 指标 | SR, SPL |
| 感知模块 | GroundingDINO + MobileSAM |
| 规划器 | Qwen3-32B |
| 经验库构建 | HM3Dv1 train 采样 200k+ trajectories；MP3D train 采样 150k+ trajectories |
| 测试约束 | TrajRAG 仅用训练数据预构建，测试时冻结，避免 test-set leakage |

### 消融实验

#### Table 1: 节点表示消融 (HM3Dv1)

| TNA | TPS-G | TPS-S | SR | SPL |
|-----|-------|-------|----|-----|
| ✓ | - | - | 53.9 | 25.7 |
| - | ✓ | - | 48.1 | 22.3 |
| - | - | ✓ | 57.3 | 30.6 |
| - | ✓ | ✓ | **61.7** | **33.2** |

**说明**: 仅几何 TPS-G 最差，说明 ObjectNav 需要语义；仅语义 TPS-S 明显优于文本邻居聚合 TNA；几何+语义组合最佳。

#### Table 2: 检索策略消融 (HM3Dv1)

| Coarse | Fine | SR | SPL |
|--------|------|----|-----|
| - | SE | 54.3 | 25.6 |
| ✓ | TE | 57.8 | 29.7 |
| ✓ | SE | **61.7** | **33.2** |

**说明**: coarse layout filtering 和专门训练的 sequence embedding 都有贡献。通用 text embedding 不足以表达轨迹结构。

#### Table 3: 与其他 RAG 形式对比 (HM3Dv1)

| 方法 | 检索方式 | 检索内容 | SR | SPL |
|------|----------|----------|----|-----|
| TrajTextRAG | text embedding | text description | 53.3 | 25.6 |
| GraphRAG | graph embedding | textual scene graph | 55.2 | 30.7 |
| TrajRAG | hierarchical retrieval | topo-polar trajectory | **61.7** | **33.2** |

**说明**: 文本 RAG 缺少路径连续性，GraphRAG 信息过于泛化；topo-polar trajectory 更适合表示可操作的导航经验。

### 跨数据集泛化

| Test | Retrieval Corpus | SR | SPL |
|------|------------------|----|-----|
| HM3Dv1 val | HM3Dv1 train | 61.7 | 33.2 |
| HM3Dv1 val | MP3D train | 59.8 | 31.6 |
| HM3Dv1 val | HM3Dv1 + MP3D train | **62.5** | **33.9** |
| MP3D val | HM3Dv1 train | 39.4 | 16.2 |
| MP3D val | MP3D train | 41.7 | 17.6 |
| MP3D val | HM3Dv1 + MP3D train | **42.6** | **18.0** |

**关键发现**: 单数据集构建的经验库可以跨数据集迁移，混合经验库效果最好，说明 topo-polar 表示捕捉到一定的通用导航结构。

### SOTA 对比

| 方法 | MP3D SR/SPL | HM3Dv1 SR/SPL | HM3Dv2 SR/SPL |
|------|-------------|---------------|---------------|
| VLFM | 36.4 / 17.5 | 52.5 / 30.4 | 63.6 / 32.5 |
| SG-Nav | 40.2 / 16.0 | 54.0 / 24.9 | 49.6 / 25.5 |
| UniGoal | 41.0 / 16.4 | 54.5 / 25.1 | - |
| ApexNAV | 39.2 / 17.8 | 59.6 / 33.0 | 76.2 / 38.0 |
| BeliefMapNav | 37.3 / 17.6 | 61.4 / 30.6 | - |
| TrajRAG | **42.6 / 18.0** | **62.5 / 33.9** | **78.1 / 40.2** |

**说明**: TrajRAG 在三个 benchmark 上均取得最高 SR 和 SPL。

---

## 优点与局限

### 优点

1. **把 RAG 从文本知识扩展到 embodied trajectory experience**: 记忆内容不是文档，而是可操作的几何-语义路径经验
2. **表示设计贴合导航**: topo-polar trajectory 同时保留拓扑、方向、语义和路径连续性
3. **可持续积累**: 新 episode 可以并入 TrajRAG，形成 lifelong navigation memory
4. **和 LLM planner 解耦**: 检索模块可作为外部记忆增强不同 LLM/VLM 规划器
5. **实验充分验证检索结构有效**: 节点表示、检索策略、RAG 类型、跨数据集迁移均有消融

### 局限性

1. **依赖 RGB-D 和准确 pose**: 方法默认有深度、agent pose 和语义地图构建能力，真实机器人部署中这些输入可能有噪声
2. **依赖开放词汇检测/分割质量**: GroundingDINO/MobileSAM 错误会直接影响 semantic map 和 topo-polar node 描述
3. **经验库构建成本较高**: 需要从训练环境采样数十万 trajectories，并进行结构化归并
4. **仍以 frontier exploration 为基础**: TrajRAG 主要辅助 waypoint 选择，不直接解决底层控制、动态障碍、定位漂移等问题
5. **只验证 ObjectNav**: 论文没有扩展到更复杂的指令导航、交互式任务或真实机器人长期部署

---

## 相关工作

### 导航方法

- [[VLFM]]: 使用 VLM 相似度为 frontier 打分
- [[SG-Nav]]: scene graph 辅助开放词汇导航
- [[UniGoal]]: 统一 ObjectNav / InstNav 的目标导航方法
- [[ApexNAV]]: 训练自由开放词汇 ObjectNav 方法
- [[BeliefMapNav]]: 维护目标 belief map 辅助探索

### 记忆与 RAG

- [[Retrieval-Augmented Generation]]: 文档检索增强语言模型
- [[GraphRAG]]: 图结构 RAG，强调实体关系
- [[EmbodiedRAG]]: 面向具身任务的场景记忆检索
- [[Topological Map]]: 用节点和边表示可通行空间结构

---

## 代码与资源

| 类型 | 链接 |
|------|------|
| arXiv | https://arxiv.org/abs/2605.01700 |
| HTML | https://arxiv.org/html/2605.01700v1 |
| DOI | https://doi.org/10.48550/arXiv.2605.01700 |
| Code | N/A |

---

## 个人思考

TrajRAG 对家庭机器人导航有直接参考价值：它没有把长期记忆做成纯文本日志，而是把轨迹压缩成可匹配的拓扑-语义结构。相比简单保存“哪个房间有什么物体”，这种表示更接近“从类似布局走到目标的经验”，因此更适合长程探索决策。

但如果要落地到真实机器人，需要重点替换或增强三点：

1. 用在线 SLAM / VIO 提供稳定 pose，并处理 drift 对 topo-polar 匹配的影响
2. 用实例级开放词汇地图替代单纯类别 semantic map，避免同类物体混淆
3. 把 TrajRAG 的经验库和家庭长期记忆合并，让同一住宅中的多次任务形成个性化经验，而不是只依赖跨场景通用经验

---

## 速查卡片

> [!summary] TrajRAG
> - **任务**: zero-shot ObjectNav
> - **核心**: 检索历史几何-语义轨迹经验增强 LLM planner
> - **方法**: topo-polar trajectory + hierarchical chunking + coarse-to-fine retrieval
> - **感知**: GroundingDINO + MobileSAM 构建 open-vocabulary semantic map
> - **规划**: Qwen3-32B 选择 frontier waypoint
> - **结果**: MP3D 42.6/18.0, HM3Dv1 62.5/33.9, HM3Dv2 78.1/40.2
> - **局限**: 依赖 RGB-D、pose、语义分割质量和大规模预构建经验库
