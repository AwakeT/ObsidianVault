---
title: "ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation"
method_name: ABot-N0
authors:
  - AMAP CV Lab
year: 2026
venue: Technical Report / arXiv
tags:
  - embodied-navigation
  - vla
  - vision-language-action
  - vln
  - object-navigation
  - point-goal-navigation
  - poi-navigation
  - person-following
  - flow-matching
  - qwen3
  - topo-memory
  - map-as-memory
  - dataset
image_source: online
arxiv: https://arxiv.org/abs/2602.11598
arxiv_pdf: https://arxiv.org/pdf/2602.11598
project: https://amap-cvlab.github.io/ABot-Navigation/ABot-N0/
code: https://github.com/amap-cvlab/ABot-Navigation/tree/ABot-N0
created: 2026-03-26
---

# 论文笔记：ABot-N0

## 元信息

| 项目   | 内容                                                                                                                                                                                                                          |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 论文   | ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation                                                                                                                                     |
| 机构   | AMAP CV Lab, Alibaba Group                                                                                                                                                                                                  |
| 作者说明 | 正文首页署名为 AMAP CV Lab；贡献页列出 Zedong Chu、Shichao Xie、Xiaolong Wu、Yanfen Shen、Minghua Luo、Zhengbo Wang、Fei Liu 等研究与工程贡献者                                                                                                         |
| 日期   | 2026-02-12                                                                                                                                                                                                                  |
| 任务   | Point-Goal, Object-Goal, Instruction-Following, POI-Goal, Person-Following                                                                                                                                                  |
| 核心模型 | Universal Multi-Modal Encoder + Qwen3-4B Cognitive Brain + Flow Matching Action Expert                                                                                                                                      |
| 数据规模 | 7,802 个高保真 3D 场景；10.7 km²；16.9M 专家轨迹；5.0M reasoning samples                                                                                                                                                                 |
| 真实部署 | Unitree Go2 X；三目 RGB；4D LiDAR；RTK-GNSS；Jetson Orin NX；VLA 2Hz；控制 10Hz+                                                                                                                                                      |
| 链接   | [arXiv](https://arxiv.org/abs/2602.11598) / [PDF](https://arxiv.org/pdf/2602.11598) / [Project](https://amap-cvlab.github.io/ABot-Navigation/ABot-N0/) / [Code](https://github.com/amap-cvlab/ABot-Navigation/tree/ABot-N0) |

---

## 一句话总结

> ABot-N0 是一个面向通用具身导航的 VLA foundation model：用统一的 Brain-Action 架构覆盖 Point-Goal、Object-Goal、Instruction-Following、POI-Goal 和 Person-Following 五类任务，并通过大规模场景、轨迹、推理数据和 Topo-Memory agentic system 将模型推进到真实机器人长时程导航部署。

---

## 核心贡献

1. **五类导航任务统一建模**：将几何点目标、开放词表物体目标、自然语言路径指令、POI 入口目标和动态人跟随都转成统一 goal-conditioned navigation 问题，避免为每个任务单独训练一个策略。
2. **Brain-Action 分层 VLA 架构**：使用 Qwen3-4B 作为 Cognitive Brain 处理语义理解、空间推理和任务上下文，再用 Flow Matching Action Expert 生成连续局部轨迹，而不是让 LLM 直接输出离散动作。
3. **大规模导航数据引擎**：构建三层数据体系：高保真 3D 场景生态、16.9M 多任务专家轨迹、5.0M 显式导航推理数据，覆盖室内、室外、公共空间、动态城市和真实机器人日志。
4. **社会合规与部署导向训练**：三阶段训练从 cognitive warm-up 到 unified sensorimotor SFT，再到 SAFE-GRPO，对 Action Expert 做社会规范、专家相似性、平滑性和效率对齐。
5. **Agentic Navigation System**：把 ABot-N0 放入 Planner + Actor + Episodic Memory + Topo-Memory + Self-Reflector + Neural Controller 的系统中，支持长时程、多阶段、可重规划的真实机器人任务。
6. **多 benchmark 强结果**：在 CityWalker、SocNav、VLN-CE R2R/RxR、HM3D-OVON、BridgeNav、EVT-Bench 上展示统一模型相对专用模型的竞争力。

---

## 问题背景

### 要解决的问题

传统 embodied navigation 通常按任务切分：

- PointNav 关注给定局部坐标如何到达；
- ObjectNav 关注在未知场景里找到目标类别；
- VLN 关注按自然语言路径描述行动；
- POI navigation 关注从室外粗粒度场景进入具体商户或地点入口；
- Person Following 关注动态人体目标跟踪。

这些任务在 benchmark 上各自发展，但真实机器人任务往往混合发生。例如“带我去最近的奶茶店，进门后找个空沙发”等指令，需要先用地图做远程 approaching，再用 POI-Goal 找入口，用 Instruction-Following 进店，最后用 Object-Goal 找座位。ABot-N0 的核心目标是把这些任务统一到一个可被上层 planner 调用的 VLA 导航底座中。

### 现有方法的局限

- **任务割裂**：不同任务使用不同输入格式、动作空间和模型结构，难以共享空间先验。
- **动作表达离散或过短视**：许多 VLN / ObjectNav 方法输出离散动作或单步动作，难以直接满足真实机器人连续控制。
- **高层语义与低层控制脱节**：LLM/VLM 擅长理解意图，但不擅长直接生成稳定、可执行、避障的连续轨迹。
- **长期记忆缺失**：单次 episode 内策略很难利用跨任务积累的空间经验。
- **社会合规不足**：到达目标不等于可接受路径，真实环境还要求不进车道、草坪、禁行区域，并与行人保持合适关系。

### 本文动机

作者的判断是：导航基础模型不能只靠单任务模仿学习，而需要同时具备三类能力：

1. **统一目标接口**：把文本、坐标、POI、人等不同目标统一进 token 空间；
2. **连续动作生成**：输出短时局部轨迹，而不是只输出离散动作；
3. **外部记忆与系统规划**：用 Topo-Memory 和 Planner 承担跨尺度、长时程任务分解。

---

## 方法详解

### 总体架构

ABot-N0 采用 hierarchical Brain-Action design，核心是三段式：

```text
RGB / Pano RGB + Episodic Visual Memory + Goal + Reasoning Task
        ↓
Universal Multi-Modal Encoder
        ↓
Qwen3-4B Cognitive Brain
        ↓
Reasoning Head 或 Action Head
        ↓
Flow Matching Action Expert
        ↓
5 个局部 BEV waypoint: (x, y, theta)
```

模型的输入输出可以按任务模式理解：

| 模式 | 输入 | 输出 |
|------|------|------|
| Reasoning 模式 | 当前图像、历史图像、任务描述、问题文本 | 文本化 reasoning / VQA / grounding 结果 |
| Navigation 模式 | 当前图像、历史图像、导航目标、任务 token | 条件轨迹分布，采样得到未来 5 个 waypoint |

关键点是 Reasoning Head 和 Action Head 不是串行的“先生成 CoT 再行动”，而是受 task token 控制的双分支结构。Reasoning 主要在训练中塑形表征，让 Brain 学到可通行区域、社会规范、目标 grounding 和空间关系；导航时 Action Expert 直接消费 Brain 的 latent context。

### 模块1：Universal Multi-Modal Encoder

Universal Multi-Modal Encoder 负责把不同机器人传感器配置和不同任务目标统一成 token 序列。

#### Flexible Vision Interface

视觉输入包括：

- **Current Observation**：当前 RGB 观察，用 ViT 编码；
- **Front-view mode**：只使用前向单目图像；
- **Panoramic mode**：左、前、右三路图像分别送入视觉编码器，并用 view-specific special tokens 标识视角；
- **Episodic Visual Memory**：显式维护短期历史视觉缓存，把相关历史帧编码后追加进上下文窗口。

三视角没有直接拼接成一张全景图，这一点很重要。拼接会引入几何畸变，并把不同相机视角的空间语义混在一起；分视角编码则保留了“左/前/右”相对方位。

#### Heterogeneous Navigation Target Encoder

ABot-N0 把导航目标分为两类：

| 目标类型 | 覆盖任务 | 编码方式 |
|----------|----------|----------|
| 语义目标 | Instruction-Following、Object-Goal、POI-Goal、Person-Following | 直接用 LLM tokenizer 编码自然语言目标、类别名、POI 名或目标人物描述 |
| 几何目标 | Point-Goal | 局部 BEV 坐标 `(x, y)` 经过 MLP 投影成 shared embedding，作为 pseudo-token 输入 LLM |

这使得 Point-Goal 不再是单独的低层控制任务，而是与文本目标共用 Brain 的上下文空间。

#### Reasoning Task Encoder

Reasoning Task Encoder 注入具体认知任务描述，例如：

- 识别哪片区域可通行；
- 判断红绿灯、斑马线、行人让行等社会导航规则；
- 定位 POI 的实际入口；
- 判断目标物体是否可见以及相对方位。

这些任务作为辅助训练目标，让模型不仅学习“专家怎么走”，也学习“为什么应该这么走”。

### 模块2：Cognitive Brain

Cognitive Brain 使用预训练 LLM **Qwen3-4B**。输入包括：

- reasoning instructions；
- navigation goals；
- episodic visual memory；
- current observations；
- task tokens。

其作用不是替代传统控制器，而是形成 physically grounded latent context：

- 对自然语言指令做语义解析；
- 对目标、地标、对象、POI 做 grounding；
- 理解可通行区域与社会规范；
- 为 Action Expert 提供条件上下文。

可以把 Cognitive Brain 理解成一个“任务条件化的空间语义中枢”，而不是直接控制机器人底盘的模块。

### 模块3：Flow Matching Action Expert

Action Expert 输出局部 BEV 坐标系中的短时轨迹：

$$
W=\{(x_1,y_1,\theta_1),(x_2,y_2,\theta_2),...,(x_5,y_5,\theta_5)\}
$$

其中：

- $(x_i, y_i)$ 是第 $i$ 个未来 waypoint 的局部位置；
- $\theta_i$ 是对应朝向；
- 共 5 个 waypoint，用作下游控制器的短时目标。

作者选择 Flow Matching，而不是简单回归，有两个直接动机：

1. **连续精度**：真实机器人需要平滑、可执行、带朝向的连续控制目标；
2. **多峰动作分布**：同一场景中从左绕障或右绕障都可能合理，MSE 回归容易把多个模式平均成一条撞障路径。

因此 Action Expert 不是预测单条确定轨迹，而是建模条件轨迹分布，再从中生成可执行 waypoint。

---

## 数据集详解

ABot-N0 的数据部分是论文最重的部分。它不是把已有 benchmark 简单拼接，而是试图建立一个统一导航数据工厂：

```text
High-Fidelity 3D Scene Ecosystem
        ↓
Universal Trajectories Dataset
        ↓
Cognitive Reasoning Dataset
```

### 第一层：High-Fidelity 3D Scene Ecosystem

场景生态包含：

| 指标 | 数值 |
|------|------|
| 3D 场景数 | 7,802 |
| 总面积 | 10.7 km² |
| 室内面积 | 6.25 km² |
| 室外面积 | 4.42 km² |
| 导航图总长度 | 384,754 m |

所有场景都人工标注可通行导航图，用于生成 collision-free 和 socially compliant 轨迹。

#### 室内环境

室内场景覆盖从家庭到大型公共空间：

| 子类 | 数据来源 / 特点 | 作用 |
|------|----------------|------|
| Residential Home | HM3D + InteriorGS，共 2,318 个住宅单元 | Object-Goal、Instruction-Following、家庭服务导航 |
| Office | 私有 3DGS 公共室内扫描，包含窄走廊、工位、会议室 | 细粒度避障、走廊转向、目标搜索 |
| Shopping Mall | 大规模商场，约 3.02 km²，多层结构、反光玻璃店面 | POI、开放词表地标、复杂拓扑 |
| Transit Station | 车站、闸机、等待大厅、窄通道、行人流瓶颈 | 社会导航、长距离室内公共空间导航 |

室内数据的价值在于把传统家庭场景扩展到更难的公共空间。公共空间有更复杂拓扑、更宽阔但约束更强的可通行区域，也更接近真实服务机器人需求。

#### 室外环境

室外场景包括真实扫描和动态虚拟城市：

| 子类 | 规模 / 特点 | 作用 |
|------|-------------|------|
| Intersections | 37 个真实室外扫描的一部分，LiDAR + photogrammetry 重建 | 学习人行道、斑马线、车道边界，禁止进入机动车道 |
| Parks | 8 个公园场景，窄步道和开放广场 | 非结构化道路、开放空间路径选择 |
| SocCity | 3.37 km² 动态虚拟城市 | 动态车辆/行人、层级 occupancy map、SAFE-GRPO 社会合规训练 |

SocCity 是社会导航对齐的关键来源，因为它能提供车辆、行人、道路类别和动态障碍的可控标注。

### 第二层：Universal Trajectories Dataset

轨迹数据总量约 **16.9M**，覆盖五类导航任务，并统一成连续 waypoint 动作监督。

| 任务 | 规模 | 核心来源 |
|------|------|----------|
| Point-Goal | 4.0M | 互联网第一人称视频伪轨迹、3D 场景合成、真实机器人日志 |
| Instruction-Following | 2.8M | VLN-CE R2R/RxR、Door-Traversal、Person Search、Short-Horizon |
| Object-Goal | 3.6M | HM3D ObjectNav、OVON、OVON-sub、InteriorGS ObjectNav |
| POI-Goal | 2.5M | BridgeNav + occupancy map planning + video generation |
| Person-Following | 4.0M | humanoid avatar trajectories、STT/DT/AT、target-absent cases |

#### Point-Goal Trajectories：4.0M

Point-Goal 是 ABot-N0 的低层移动基础能力，数据来自三条流：

| 来源 | 规模 | 构造方式 | 价值 |
|------|------|----------|------|
| 互联网第一人称视频伪轨迹 | 2.0M | 用 $\pi^3$ 从单目视频恢复稠密 3D 结构，再用 MoGe 做尺度对齐，沿相机路径采样 point-goal pairs | 捕获长尾建筑风格、天气、街景外观 |
| 高保真 3D 场景合成 | 1.7M | 在导航图上采样可达起终点，用路径规划生成最优路径，并加入 off-path / near-collision recovery trajectories | 提供可控、无碰撞、社会合规轨迹 |
| 真实机器人示范 | 340K | SCAND、HuRoN、Recon、CityWalker teleoperation logs | 引入真实动力学、传感噪声、非完整约束、延迟 |

这一设计把 passive video、synthetic simulation 和 real robot demos 统一到 point-goal 表达中，是数据工程上的关键点。

#### Instruction-Following Trajectories：2.8M

Instruction-Following 数据分四块：

| 子集 | 规模 | 构造细节 | 训练作用 |
|------|------|----------|----------|
| VLN-CE R2R | 10K clips；600K 初始样本过滤到 200K | Matterport3D 连续环境，teacher forcing 展开每个时间步，做动作分布平衡 | 短路径自然语言导航 |
| VLN-CE RxR | 20K clips；1.8M 初始样本保留 1.3M | 更长路径、更复杂拓扑、更高语言密度 | 长时程语言 grounding |
| Door-Traversal | 6K clips；0.3M samples | InteriorGS 中门口附近采样初始姿态，控制穿门后的朝向和距离，多指令标注 | 窄通道、门洞、局部几何精度 |
| Language-Guided Person Search | 4K clips；0.2M samples | InteriorGS 200 个 3DGS avatars + Habitat 2,000 个 mesh characters，2-5 人/场景，LLM 生成自然描述 | 用语言描述定位人物 |
| Short-Horizon | 12K clips；0.8M samples | 左转、掉头、左转 60 度、前进 1m、后退 3m 及组合动作 | 稳定短指令与细粒度执行 |

这里的重点是：论文没有只依赖 R2R/RxR，而是额外补了门洞、人物搜索和短程原子动作，从而缓解 VLN 数据对低层控制覆盖不足的问题。

#### Object-Goal Trajectories：3.6M

Object-Goal 数据分三类：

| 子集 | 规模 | 构造细节 | 作用 |
|------|------|----------|------|
| HM3D ObjectNav | 10K expert clips；0.4M trajectories | 6 个目标类别，80 个场景 | 标准封闭类别 ObjectNav |
| OVON | 35K episodes；1.4M trajectories | 145 个场景，每个 object instance 采样 10 个起点，开放词表查询 | unseen categories、synonyms、语义泛化 |
| OVON-sub | 6K clips；0.2M samples | 从目标第一次进入视野到 episode 结束的短程片段 | 强化可见目标的视觉-语义对齐 |
| InteriorGS ObjectNav | 40K clips；1.6M samples | 1,000 个 photorealistic scenes，700+ object categories bounding boxes，自定义 object visibility estimation | 高保真室内开放类别目标搜索 |

Object-Goal 的设计思路是同时覆盖“长程搜索”和“目标可见后的精确靠近”。OVON-sub 这类短程数据非常关键，因为标准 ObjectNav 的长程路径中有大量时间看不到目标，容易让模型学到场景先验而不是目标 grounding。

#### POI-Goal Trajectories：2.5M

POI-Goal 目标是解决“最后一米入口定位”：

1. 输入 street-view / outdoor 图像；
2. 用图像分割和深度估计生成 local occupancy grid；
3. 用 A* 从起点规划到 POI 入口；
4. 将输入图像和轨迹送入 Wan2.1-I2V Diffusion Transformer；
5. 合成第一视角导航视频；
6. 从生成视频中采样 2.5M 训练轨迹。

这类数据不是简单地“找到店铺招牌”，而是要定位可进入的物理入口。对真实场景来说，POI 的语义位置和可通行入口经常不完全一致，例如店名在高处招牌上，实际入口在侧面或玻璃门处。

#### Person-Following Trajectories：4.0M

Person-Following 数据参考 TrackVLA 的数据构造，但由于其训练数据未完全开源，作者基于公开 humanoid avatar motion trajectories 自建数据。

| 维度 | 设置 |
|------|------|
| 跟随距离 | 2.0m、1.5m、1.2m |
| 场景类别 | STT、DT、AT |
| 每个距离-类别组合 | 400K samples |
| 基础样本数 | 3.6M = 3 距离 × 3 类别 × 400K |
| 目标缺失样本 | 400K |
| 总量 | 4.0M |

三类挑战分别是：

- **STT, Single-Target Tracking**：单目标正常跟随；
- **DT, Distracted Tracking**：存在相似行人干扰，考验身份保持；
- **AT, Ambiguity Tracking**：频繁遮挡，需要重识别和目标预测。

每个样本包含当前图像、历史帧、未来轨迹和跟随命令。这与模型中的 Episodic Visual Memory 直接对应。

### 第三层：Cognitive Reasoning Dataset

Reasoning 数据总量 **5.0M**，用于让 Brain 显式学习空间、语义、社会规则和 grounding。

| 子集 | 规模 | 构造方式 | 作用 |
|------|------|----------|------|
| Navigable Areas Analysis | 1.2M | 户外第一视角图像，人工标注 socially compliant polygons，例如人行道、斑马线，排除车道、草坪 | 学习“可走”和“应该走”的区别 |
| Social Navigation CoT | 0.8M | SocialNav pipeline + Qwen-VL-Max teacher 生成结构化决策理由 | 红灯等待、避让行人、社会规范 |
| Instruction-Following Reasoning | 1.3M | Gemini-3 Pro + Qwen3-VL 处理 30K R2R/RxR clips，拆成 atomic sub-instructions 和 milestone nodes | 语言进度跟踪、Visual CoT |
| Object-Goal Reasoning | 0.1M | OVON 上由 MLLM 生成目标可见性、空间关系、路径规划、原子动作链，并用 GT trajectory 做一致性过滤 | 开放词表目标搜索的中间推理 |
| POI Grounding | 0.5M | BridgeNav street-view 图像，Qwen3-VL-Plus 生成 `<POI Name, Entrance Coordinates (u,v)>` | POI 入口级 grounding |
| General VQA | 1.1M | Blip3、COCO2014/2017、MAmmoTH-VL、RefCOCO、Objects365、R2R-EnvDrop、ScanQA | 保持通用视觉语言能力和 referring grounding |

Reasoning 数据的价值在于，它把“显式认知任务”作为训练主料之一，而不是只做事后解释。这使得 Brain 的 latent context 更容易包含社会合规、空间关系和目标 grounding 这些对行动有用的信息。

---

## 训练 Recipe

ABot-N0 使用三阶段 curriculum learning。

### Phase 1：Cognitive Warm-up

目标是先让模型学会看懂和推理，再学习行动。

- 使用 5.0M Reasoning Dataset；
- 冻结 Vision Encoder 和 tokenizer；
- fine-tune LLM core；
- 使用 Next Token Prediction loss；
- Action Expert 冻结。

这一阶段优化的是视觉-语言-空间推理表征，而不是控制能力。

### Phase 2：Unified Sensorimotor SFT

目标是把五类导航任务统一到 sensorimotor policy 中。

- 引入 16.9M Trajectory Dataset；
- 五类任务混合训练；
- reasoning replay buffer 占约 20%，防止 Phase 1 的认知能力遗忘；
- 联合优化 AR reasoning head 和 Flow Matching Action Expert。

训练目标：

$$
\mathcal{L}_{Phase2}
=\lambda_{txt}\mathcal{L}_{NTP}(\theta_{brain})
+\lambda_{flow}\mathcal{L}_{CFM}(\theta_{action}\mid\theta_{brain})
$$

其中：

- $\mathcal{L}_{NTP}$ 是文本生成的 next token prediction loss；
- $\mathcal{L}_{CFM}$ 是 conditional flow matching trajectory loss；
- $\theta_{brain}$ 是 Cognitive Brain 参数；
- $\theta_{action}$ 是 Action Expert 参数。

论文没有展开 CFM 的完整推导。按标准 conditional flow matching，可理解为在条件上下文 $c$ 下学习从噪声轨迹到专家轨迹的速度场：

$$
\mathcal{L}_{CFM}
=
\mathbb{E}_{t,x_0,x_1,c}
\left[
\left\|
v_\theta(x_t,t,c)-(x_1-x_0)
\right\|^2
\right]
$$

这里 $x_1$ 可视为专家 waypoint trajectory，$x_0$ 是噪声轨迹，$x_t$ 是中间插值状态。

### Phase 3：SAFE-GRPO

目标是让 Action Expert 不仅模仿专家，还显式对齐社会规范。

- 冻结 Brain；
- 只 fine-tune Action Expert；
- 使用 SocCity 专家轨迹和语义 occupancy map；
- 通过 SAFE-GRPO 做 post-training value alignment。

奖励函数：

$$
R=w_{soc}R_{social}+w_{exp}R_{expert}+w_{sm}R_{smooth}+w_{eff}R_{eff}
$$

各项含义：

| 奖励项 | 作用 |
|--------|------|
| $R_{social}$ | 基于语义 occupancy map，惩罚进入草坪、车道、禁行区域、行人区域等不合规路径 |
| $R_{expert}$ | 约束策略不要偏离专家轨迹分布，防止 reward hacking |
| $R_{smooth}$ | 惩罚急促、抖动、不平滑动作 |
| $R_{eff}$ | 鼓励朝目标有效推进，避免低效绕行 |

这一步体现了论文的一个重要判断：真实导航评价不能只看是否到达，还要看路径是否符合社会规则和机器人动力学可执行性。

---

## 关键公式

### 公式1：Action Expert 输出

$$
W=\{(x_1,y_1,\theta_1),(x_2,y_2,\theta_2),...,(x_5,y_5,\theta_5)\}
$$

**含义**：模型不输出单步离散动作，而是输出未来 5 个局部 BEV waypoint，每个 waypoint 包括位置和朝向。

### 公式2：Phase 2 联合训练目标

$$
\mathcal{L}_{Phase2}
=\lambda_{txt}\mathcal{L}_{NTP}(\theta_{brain})
+\lambda_{flow}\mathcal{L}_{CFM}(\theta_{action}\mid\theta_{brain})
$$

**含义**：同时保持语言/推理能力和轨迹生成能力。

### 公式3：SAFE-GRPO 奖励

$$
R=w_{soc}R_{social}+w_{exp}R_{expert}+w_{sm}R_{smooth}+w_{eff}R_{eff}
$$

**含义**：对 Action Expert 进行社会合规、专家相似性、平滑性和效率对齐。

### 公式4：Planner 任务分解

$$
G=P(I,M_L,M_S,O_t)
$$

其中：

- $I$：用户高层指令；
- $M_L$：长期 Topo-Memory；
- $M_S$：短期 Episodic Visual Memory；
- $O_t$：当前观测；
- $G=\{g_1,g_2,...,g_n\}$：可由 ABot-N0 执行的子任务序列。

### 公式5：Coarse-to-Fine 系统分解

$$
P(W\mid I,M_L,M_S)
=
P(G\mid I,M_L,M_S)
\cdot
\prod_{j=1}^{N}P(W_j\mid g_j,M_L,M_S)
$$

**含义**：上层 Planner 先把高层意图拆成子任务，ABot-N0 再为每个子任务生成局部轨迹。

### 公式6：Self-Reflection 与重规划

$$
(r,f)=S(M_L,M_S,g_i)
$$

$$
G'=P(I,M_L,M_S,O_t,f)
$$

**含义**：Self-Reflector 判断子任务是否完成；失败时把反馈重新输入 Planner 生成新计划。

---

## 实验结果

论文覆盖 5 类任务、7 个 benchmark：

| 任务 | Benchmark | 指标 |
|------|-----------|------|
| Point-Goal | CityWalker | MAOE |
| Point-Goal | SocNav | SR, RC, SPL, DCR, TCR |
| Instruction-Following | VLN-CE R2R / RxR | NE, OS, SR, SPL |
| Object-Goal | HM3D-OVON | SR, SPL |
| POI-Goal | BridgeNav | entrance SR, trajectory deviation |
| Person-Following | EVT-Bench | SR, TR, CR |

### Point-Goal：CityWalker Open-Loop

MAOE 越低越好。

| Method | Mean | Turn | Crossing | Detour | Proximity | Crowd | Other | All |
|--------|------|------|----------|--------|-----------|-------|-------|-----|
| GNM | 16.2 | 31.1 | 14.8 | 12.5 | 14.7 | 12.8 | 11.0 | 12.1 |
| ViNT | 16.5 | 31.1 | 15.4 | 12.9 | 14.8 | 13.3 | 11.6 | 12.6 |
| NoMaD | 19.1 | 35.1 | 18.5 | 15.6 | 18.1 | 14.3 | 12.8 | 12.1 |
| CityWalker | 15.2 | 26.6 | 14.1 | 13.9 | 14.3 | 12.0 | 10.4 | 11.5 |
| ABot-N0 | 11.2 | 21.3 | 9.8 | 12.8 | 8.1 | 8.8 | 6.3 | 7.6 |

ABot-N0 在所有场景类别上 MAOE 都更低，说明它在开环轨迹预测上更接近专家行为。

### Point-Goal：SocNav Closed-Loop

| Method | SR ↑ | RC ↑ | SPL ↑ | DCR ↑ | TCR ↑ |
|--------|------|------|-------|-------|-------|
| GNM | 43.3 | 62.4 | 37.0 | 26.5 | 28.7 |
| ViNT | 45.6 | 66.2 | 39.5 | 31.4 | 33.8 |
| NoMaD | 41.1 | 60.5 | 35.4 | 29.5 | 31.6 |
| CityWalker | 47.8 | 64.7 | 44.7 | 36.1 | 36.6 |
| ABot-N0 | 88.3 | 92.1 | 79.2 | 85.1 | 85.4 |

最值得注意的不是 SR 从 47.8 提升到 88.3，而是 DCR/TCR 从约 36 提升到约 85。这说明 SAFE-GRPO 和 social reasoning 数据对“走在合规区域内”有直接意义。

### Instruction-Following：VLN-CE

R2R-CE Val-Unseen：

| Method | NE ↓ | OS ↑ | SR ↑ | SPL ↑ |
|--------|------|------|------|-------|
| NaVILA | 5.22 | 62.5 | 54.0 | 49.0 |
| StreamVLN | 4.98 | 64.2 | 56.9 | 51.9 |
| InternVLA-N1 (S1+S2) | 4.83 | 63.3 | 58.2 | 54.0 |
| NavFoM (Four views) | 4.61 | 72.1 | 61.7 | 55.3 |
| ABot-N0 | 3.78 | 70.8 | 66.4 | 63.9 |

RxR-CE Val-Unseen：

| Method | NE ↓ | SR ↑ | SPL ↑ |
|--------|------|------|-------|
| NaVILA | 6.77 | 49.3 | 44.0 |
| StreamVLN | 6.22 | 52.9 | 46.0 |
| InternVLA-N1 (S1+S2) | 5.91 | 53.5 | 46.1 |
| NavFoM (Four views) | 4.74 | 64.4 | 56.2 |
| ABot-N0 | 3.83 | 69.3 | 60.0 |

ABot-N0 在 SR 和 SPL 上都明显强，说明五任务统一训练没有削弱 VLN，反而可能通过 Point-Goal、Object-Goal、Short-Horizon 等数据增强了空间执行能力。

### Object-Goal：HM3D-OVON

| Method | Val-Seen SR ↑ | Val-Seen SPL ↑ | Synonyms SR ↑ | Synonyms SPL ↑ | Val-Unseen SR ↑ | Val-Unseen SPL ↑ |
|--------|---------------|----------------|---------------|----------------|-----------------|------------------|
| DAgRL+OD | 38.5 | 21.1 | 39.0 | 21.4 | 37.1 | 19.8 |
| Uni-NaVid | 41.3 | 21.1 | 43.9 | 21.8 | 39.5 | 19.8 |
| MTU3D | 55.0 | 23.6 | 45.0 | 14.7 | 40.8 | 12.1 |
| NavFoM (Four views) | 40.1 | 27.1 | 45.4 | 32.6 | 45.2 | 31.9 |
| ABot-N0 | 55.3 | 32.1 | 55.4 | 33.2 | 54.0 | 30.5 |

ABot-N0 的亮点是开放词表泛化：Val-Seen 到 Val-Unseen 的 SR 只从 55.3 降到 54.0。需要注意的是，在 Val-Unseen SPL 上，NavFoM 的 31.9 略高于 ABot-N0 的 30.5，因此更准确的表述是 ABot-N0 在 SR 上最强，在 SPL 上大体持平或接近最强，而不是所有列都绝对第一。

### POI-Goal：BridgeNav

| Method | SR@0.1m ↑ | SR@0.2m ↑ | SR@0.3m ↑ | TR mean ↓ | TR best ↓ | TR worst ↓ |
|--------|-----------|-----------|-----------|-----------|-----------|------------|
| NoMaD | 4.13 | 15.07 | 29.20 | 31.35 | 5.45 | 85.91 |
| CityWalker | 13.79 | 41.02 | 65.96 | 15.58 | 0.76 | 56.47 |
| OmniNav | 18.78 | 46.99 | 72.39 | 14.16 | 0.99 | 53.79 |
| ABot-N0 | 32.14 | 71.50 | 88.68 | 9.84 | 0.44 | 51.38 |

POI-Goal 最能体现“最后一米”能力。ABot-N0 在最严格的 0.1m 入口中心阈值下仍有明显优势，说明它不是只找到大概店铺位置，而是更接近可执行的入口到达。

### Person-Following：EVT-Bench

| Method | STT SR ↑ | STT TR ↑ | STT CR ↓ | DT SR ↑ | DT TR ↑ | DT CR ↓ | AT SR ↑ | AT TR ↑ | AT CR ↓ |
|--------|----------|----------|----------|---------|---------|---------|---------|---------|---------|
| TrackVLA | 85.1 | 78.6 | 1.65 | 57.6 | 63.2 | 5.80 | 50.2 | 63.7 | 17.1 |
| NavFoM | 85.0 | 80.5 | - | 61.4 | 68.2 | - | - | - | - |
| TrackVLA++ | 86.0 | 81.0 | 2.10 | 66.5 | 68.8 | 4.71 | 51.2 | 63.4 | 15.9 |
| ABot-N0 | 86.9 | 87.6 | 8.54 | 66.7 | 75.4 | 11.6 | 67.3 | 79.5 | 7.05 |

ABot-N0 在 SR/TR 上尤其强，特别是 AT 遮挡重识别场景中提升明显。但表格也暴露一个重要问题：在 STT 和 DT 中，ABot-N0 的 Collision Rate 并不是最低，甚至高于 TrackVLA/TrackVLA++。因此它更像是“跟踪成功率和目标保持率更强”，但安全碰撞指标仍需要额外分析。

---

## Agentic Navigation System

ABot-N0 不是单独部署为端到端策略，而是嵌入一个 agentic navigation system。

### 系统模块

| 模块 | 作用 |
|------|------|
| Agentic Planner | 用 VLM/LLM 解析用户高层指令，结合 memory 做任务分解 |
| Actor | 包含 ABot-N0 和 Neural Controller，执行具体子任务 |
| Episodic Memory | 保存当前 episode 的短期视觉历史，用于上下文决策和错误恢复 |
| Topo-Memory | 长期层级拓扑空间记忆，支持跨尺度路径规划和经验沉积 |
| Self-Reflector | 子任务完成后判断是否成功，失败则生成反馈触发重规划 |
| Neural Controller | 将 ABot-N0 的低频 waypoint 转成高频速度控制 |

### Coarse-to-Fine 任务分解

系统将真实任务拆成三个层级：

| 阶段 | 对应能力 | 示例 |
|------|----------|------|
| Approaching | Point-Goal | 根据 Topo-Memory 走到目标区域或建筑附近 |
| Reaching | Object-Goal / POI-Goal | 找到具体物体、店铺入口、功能区 |
| Interaction | Instruction-Following / Person-Following | 进门、绕过人、跟随目标、执行局部指令 |

例如“去奶茶店买伯牙绝弦并帮我占座”可以拆成：

1. Point-Goal：到达奶茶店附近；
2. POI-Goal：精确到店铺入口；
3. Instruction-Following：进店并到柜台；
4. Object-Goal：寻找空沙发。

### Map-as-Memory / Topo-Memory

Topo-Memory 是论文对 VLN / VLA 系统最有启发的部分之一。它不是把地图当作静态背景，而是把地图作为持续更新的外部空间记忆。

层级结构：

| 层级 | 内容 | 作用 |
|------|------|------|
| Block Layer | 房间、街区、校园区域、商场片区 | 粗粒度定位和远程任务分解 |
| Road Layer | 门、走廊、路口、道路连通 | 物理可达性约束 |
| Function Layer | 厨房、休息区、电梯厅、卫生间等功能节点 | 将抽象语言意图映射到功能区域 |
| Object/POI Layer | 具体物体、店铺、入口、语义锚点 | 支持最后一米 Object/POI navigation |

维护机制：

- 每次任务执行的观察、轨迹和结果都可以回写到图中；
- 环境变化会更新边权或图结构，例如施工、封路、拥堵；
- 避免基于过时地图重复失败；
- 支持 indoor-outdoor 的统一空间表示。

这与传统 occupancy map 不同。Topo-Memory 更像 agent 的长期语义空间记忆，服务于任务分解、检索、重规划和经验复用。

### Neural Controller

ABot-N0 在边端以约 2Hz 输出 strategic waypoints，但 2Hz 对密集动态环境中的避障和底盘控制不够。因此系统加入 Neural Controller：

- 基于 CE-Nav 框架；
- 使用预训练 VelFlow expert 作为先验；
- 在 Isaac Sim 中用 RL fine-tune dynamics-aware refiner；
- 输入 ABot-N0 waypoint 和 LiDAR 构建的实时 occupancy map；
- 输出机器人速度命令 $(v_x,v_y,v_{yaw})$；
- 在边缘设备上 10Hz+ 闭环运行。

这一设计说明 ABot-N0 的定位不是最低层控制器，而是低频神经规划器；高频安全控制仍由专门控制层负责。

---

## 真实部署

### 硬件平台

| 组件 | 配置 |
|------|------|
| 机器人 | Unitree Go2 X 四足机器人，12 个驱动自由度 |
| RGB 感知 | 左/前/右三路 rolling-shutter 单目 RGB，每路 H120° × V90°，合计约 270° 水平视野 |
| 几何感知 | Unitree 4D LiDAR L2 |
| 全局定位 | RTK-GNSS |
| 计算平台 | NVIDIA Jetson Orin NX，157 TOPS，16GB RAM |
| 云端 | RTX 4090，用于 Agentic Planner |

### 部署策略

系统采用 cloud-edge hybrid：

- **云端**：Agentic Planner 运行在 RTX 4090 上，负责高层指令拆解和 Self-Reflection；
- **边端**：ABot-N0 和 Neural Controller 运行在 Jetson Orin NX 上，保证弱网或断网时仍能低延迟避障与控制；
- **视觉压缩**：部署版使用 93M SigLIP-B/16 视觉 backbone；
- **Token 压缩**：使用 token merging，merge size = 4；
- **性能**：边端 VLA 2Hz，性能下降约 3%。

### 局部建图

真实部署中还使用了 LiDAR + legged odometry 的 BEV occupancy pipeline：

- 过滤瞬时障碍噪声；
- 融合时间一致性；
- 替代 Unitree SDK 默认 height-map；
- 为 Neural Controller 提供空间 grounded 的实时状态反馈。

这一点很关键：虽然论文是 VLA foundation model，但真实部署并没有抛弃经典几何感知和 occupancy map。系统是 VLA、地图、LiDAR、控制器的组合。

---

## 批判性思考

### 1. 最强贡献更像“数据 + 系统 + 模型”的组合

ABot-N0 的结构本身并不复杂：多模态编码、LLM Brain、Flow Matching action head。真正的壁垒来自：

- 7,802 个高保真 3D 场景；
- 私有 3DGS 公共空间；
- 16.9M 轨迹；
- 5.0M reasoning samples；
- SocCity 社会导航对齐；
- Agentic system 和真实机器人部署。

因此这篇论文更像“导航基础模型工程报告”，不是单一 architecture trick 论文。

### 2. 消融证据仍不够完整

Technical report 的主结果很强，但至少还缺少这些关键消融：

- 没有充分拆分 Qwen3-4B Cognitive Brain 的贡献；
- Flow Matching 相比 MSE trajectory regression / diffusion / discrete action head 的收益不够清楚；
- Reasoning Dataset 对下游五类任务分别贡献多少没有系统量化；
- SAFE-GRPO 对 DCR/TCR、碰撞率和任务成功率的 trade-off 没有完整拆解；
- 五任务联合训练相比单任务或子集训练是否真的存在正迁移，证据还不够；
- Topo-Memory、Self-Reflector、Neural Controller 对真实长时任务的独立贡献没有严格 ablation。

### 3. 数据可复现性是主要限制

论文依赖大量私有或半私有数据：

- 私有 3DGS public interiors；
- 大规模互联网第一人称视频处理；
- Wan2.1 生成式视频合成；
- MLLM 自动 reasoning 数据；
- SocCity 和 BridgeNav 相关数据；
- 真实机器人 teleoperation logs。

项目页显示 data/code 会逐步开放，但实际可复现性仍取决于后续发布范围。对于研究复现者来说，复现模型结构容易，复现数据引擎很难。

### 4. Person-Following 安全性需要谨慎解读

EVT-Bench 表格显示 ABot-N0 在 SR/TR 上很强，但 STT 和 DT 的 Collision Rate 并非最优。这说明统一 VLA 的目标保持能力强，但动态人群中的安全控制可能仍依赖 Neural Controller 或后处理安全层。不能简单把“跟得住”理解成“跟得安全”。

### 5. 真实部署仍是分层系统，不是纯端到端

ABot-N0 的真实价值在于它作为可调用的导航 foundation skill，而不是取代地图、Planner 和控制器。真实系统仍依赖：

- Topo-Memory；
- RTK-GNSS；
- LiDAR occupancy map；
- legged odometry；
- high-frequency Neural Controller；
- cloud planner。

这对后续研究很重要：VLA 模型应该和经典机器人系统协同，而不是把全部几何、控制、安全都塞进一个 LLM。

---

## 对当前 VLN / 语义地图方向的启发

### 对 VLN 的启发

ABot-N0 表明 VLN 可能会从单一 instruction-following benchmark 走向更大的 navigation VLA 框架。未来 VLN 模型不应只理解语言路线，还要与 Point-Goal、Object-Goal、POI、Person Following 等技能共享空间表征。

### 对语义地图的启发

Topo-Memory 的核心思想值得借鉴：

- 地图不是单一几何结构，而是多层语义拓扑记忆；
- 地图要服务于任务分解和语言 grounding；
- 语义节点应该包含功能区、对象、POI、入口和可达关系；
- 执行结果需要回写地图，形成 memory maintenance loop；
- 对 VLN 来说，地图不仅用于定位，也用于“把高层语言意图变成可执行子目标”。

### 对 DINO-GeoSemMap / LingBot-Map 类方案的启发

如果要设计基于 DINOv2/DINOv3 backbone 的语义几何地图模型，ABot-N0 提供了几个系统级方向：

1. **模型输出不应只停留在检测/分割/深度/位姿**，还要能被上层导航任务检索成 function nodes、object nodes、POI nodes 和 road/door connectivity。
2. **地图需要分层表达**：局部 BEV occupancy、对象级语义锚点、房间/功能区拓扑、长程导航图应该分层维护。
3. **流式 RGB 输入需要短期 memory**：单帧语义地图很容易受遮挡和视角限制，Episodic Visual Memory 是必要模块。
4. **VLN 需要 coarse-to-fine 接口**：长程用拓扑点或 Point-Goal，局部用 Object/POI/Instruction-Following。
5. **地图更新要记录执行结果**：目标找不到、入口封闭、道路拥堵等都应该回写图结构或边权，而不是每次从零推理。

---

## 关联笔记

- [[VGGT]]
- [[VGGT-Omega]]
- [[AgentVLN]]
- [[论文笔记/Anticipation-VLA]]
- [[DeCoNav]]
- [[DyGeoVLN]]
- [[GLMap]]
- [[MetaNav]]
- [[RAGNav]]
- [[TrajRAG]]
- [[论文笔记/Semantic-Autonomy-Stack]]

---

## 速查卡片

| 问题 | 答案 |
|------|------|
| ABot-N0 是什么？ | 面向具身导航的统一 VLA foundation model |
| 统一了哪些任务？ | Point-Goal、Object-Goal、Instruction-Following、POI-Goal、Person-Following |
| 核心架构是什么？ | Universal Multi-Modal Encoder + Qwen3-4B Cognitive Brain + Flow Matching Action Expert |
| 输出什么动作？ | 局部 BEV 下未来 5 个 waypoint，每个包含 `(x, y, theta)` |
| 为什么用 Flow Matching？ | 连续精度更好，并能表达多峰轨迹分布 |
| 数据规模多大？ | 7,802 场景、10.7 km²、16.9M 轨迹、5.0M reasoning samples |
| Reasoning 数据有什么用？ | 显式学习可通行区域、社会规则、目标 grounding、语言进度和空间关系 |
| 三阶段训练是什么？ | Cognitive Warm-up、Unified Sensorimotor SFT、SAFE-GRPO |
| Agentic System 的关键是什么？ | Planner + Episodic Memory + Topo-Memory + Self-Reflector + Neural Controller |
| Topo-Memory 有哪些层？ | Block、Road、Function、Object/POI |
| 真实部署平台？ | Unitree Go2 X + 三路 RGB + 4D LiDAR + RTK-GNSS + Jetson Orin NX |
| VLA 推理频率？ | 约 2Hz |
| 控制频率？ | Neural Controller 10Hz+ |
| 最大局限？ | 数据和系统依赖重，消融不足，复现难度高，部分安全指标仍需谨慎解读 |
