---
title: "Vesta: A Generalist Embodied Reasoning Model"
method_name: "Vesta"
authors: [Johan Bjorck, Zhiqi Li, Yunze Man, Jing Wang, An-Chieh Cheng, Sifei Liu, Shihao Wang, Zhiding Yu, Abhishek Badki, Stan Birchfield, Valts Blukis, Yevgen Chebotar, Siyi Chen, Sicong Leng, Yu-Cheng Chou, Tianli Ding, Boyi Li, Zhengyi Luo, Hang Su, Jonathan Tremblay, Tingwu Wang, Bowen Wen, Jimmy Wu, Xianghui Xie, Hanrong Ye, Hongxu Yin, K. R. Zentner, Liangyan Gui, Yu-Xiong Wang, Yuke Zhu, Linxi Jim Fan, Jan Kautz]
year: 2026
venue: arXiv
tags: [embodied-ai, generalist-planner, robot-planning, memory, navigation, vla, spatial-reasoning]
zotero_collection: _待整理
image_source: online
arxiv_html: https://arxiv.org/html/2606.20905
created: 2026-06-30
---

# 论文笔记：Vesta: A Generalist Embodied Reasoning Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA; UIUC; UC San Diego; Hong Kong Polytechnic University; University of Michigan; NTU; Johns Hopkins University; University of Tübingen |
| 日期 | June 2026 |
| 项目主页 | - |
| 对比基线 | [[Qwen3-VL]], RynnBrain, RoboBrain-2.5, [[InternVLA-N1]], UniNaVid, [[GR00T-N1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.20905) / [HTML](https://arxiv.org/html/2606.20905) |

---

## 一句话总结

> Vesta 把定位、导航、具身问答和长程动作规划统一到一个 8B [[VLM]] planner 中，用偏空间能力的 SFT 数据配比和显式 [[Memory Harness]]，在多个具身 benchmark 与真实双臂机器人任务上超过同尺寸 specialist 和简单专家集成。

---

## 核心贡献

1. **统一具身 planner**: 以 [[Qwen3-VL]]-8B 为基座，把 grounding/localization、[[Vision-Language Navigation|VLN]]、[[Embodied Reasoning]]、real-robot [[Action Planning]] 合并到单个模型，而不是为每类能力部署独立 specialist。
2. **空间偏置 SFT 配方**: 构造 6 类 SFT 混合数据，空间智能、导航、grounding 合计约 69.7%，保留通用 VLM 数据以避免过拟合到机器人小域。
3. **轻量多模态记忆机制**: 用历史图像采样 + 文本子任务缓存构成 [[Memory Harness]]，在不塞入完整视频历史的情况下支持长程非 Markov 任务。
4. **真实机器人验证**: 在三类记忆/推理密集双臂任务上，用 Vesta planner + [[GR00T-N1]] actor 的层级控制显著提升成功率，说明 benchmark 提升能迁移到物理执行。

---

## 问题背景

### 要解决的问题

具身智能系统通常同时需要四类能力：看懂空间位置、按语言导航、理解可供性和任务状态、把长程目标拆成低层可执行子任务。传统做法常把它们拆成多个模型或模块，部署成 specialist ensemble。

### 现有方法的局限

- 多模型 stack 计算开销高，每个模块都要单独维护 prompt、接口和推理服务。
- specialist 之间容易级联错误：上游 grounding 或 navigation 输出错了，下游 planner 很难自我修复。
- 单一 specialist 的跨域能力弱，例如导航模型在具身问答任务上会灾难性遗忘。
- 许多 [[Vision-Language-Action]] 系统缺显式历史，面对“找过哪里”“盒子里是什么”等长程任务时容易短视。

### 本文的动机

作者认为高层 planner 本质上需要一个统一的空间-语言-记忆接口：只要 SFT 数据覆盖足够广、历史信息组织得足够简洁，一个 generalist planner 可以替代 specialist ensemble，并且从跨任务正迁移中受益。

---

## 方法详解

### 模型架构

Vesta 采用 **planner VLM + low-level actor** 的层级架构：

- **输入**: 当前 egocentric 图像 $o_t$、高层语言目标 $g$、历史图像采样、历史子任务文本缓存。
- **Backbone**: [[Qwen3-VL]]-8B-Instruct，全参数 SFT，视觉塔、多模态 projector 和语言 backbone 全部解冻。
- **核心模块**: [[Memory Harness]] 将历史步骤压缩成 $\mathcal{M}_t$，prompt 中同时注入图像和文本历史。
- **输出**: 文本形式的下一步高层子任务 $a_t$ 或导航动作，例如 waypoint、turn sequence、stop。
- **执行**: 真实机器人中，Vesta 只做 planner；低层动作由 fine-tuned [[GR00T-N1]] actor 根据当前观测和子任务执行。

### 核心模块

#### 模块1: 多能力 SFT 语料

**设计动机**: 让模型在同一语言接口下学习“看哪里、往哪走、怎么理解任务状态、下一步做什么”。

**具体实现**:
- **Localization / Grounding**: 使用 [[Objects365]]、[[COCO]]、[[LVIS]] 等基础检测/grounding 数据，再加入具身视角、操作相关点位和时序交互数据；boxes/points 都作为文本 token 输出。
- **Navigation**: 使用 [[R2R]]、RxR、ScaleVLN，在 [[Habitat]] 中渲染轨迹，训练模型输出 pixel goal、左右转序列或 stop。
- **Embodied reasoning**: 覆盖 VQA、检测、pointing、affordance、placement prediction、轨迹 waypoint、任务进度估计。
- **Real robot planning**: 用真实/人手视频轨迹标注 step-level 子任务，训练模型在历史上下文下选择下一步动作。

#### 模块2: Memory Harness

**设计动机**: 长程任务是非 Markov 的，下一步动作依赖“之前看过什么、做过什么、是否已经完成某一步”。直接塞完整视频历史会造成上下文爆炸。

**具体实现**:
- 每个历史 step 记录为 $m_i=\langle i,\tau_i,o_i,a_i,g\rangle$。
- 历史集合为 $\mathcal{H}_t=\{m_1,\ldots,m_{t-1}\}$。
- 图像历史用采样集合 $\mathcal{S}_t$ 截断到最多 $K$ 帧，保留第一帧以记录初始状态。
- 采样策略包括 uniform sampling 和 recency-biased sampling。
- prompt 中加入图像历史 + 文本子任务日志；但只有最终 action $a_t$ 写入 memory，不把完整 [[Chain-of-Thought]] 写回。

#### 模块3: CoT 格式的 planner 输出

Vesta 在预测每个子任务前生成四段结构化推理：

- **Observation**: 当前看到了什么。
- **Progress**: 任务推进到哪里。
- **Reasoning**: 为什么下一步应该这样做。
- **Action**: 输出下一条子任务。

这种格式主要帮助模型显式组织中间状态，但部署时 actor 只接收 Action 子任务。

#### 模块4: Planner-Actor 真实机器人闭环

真实机器人部署中，Vesta 是高层 planner $\pi_{\text{plan}}$，[[GR00T-N1]] 是低层 actor $\pi_{\text{act}}$：

- actor 发布最新 observation 给 planner。
- planner 根据目标、最新图像和 $\mathcal{M}$ 发布子任务 $z$。
- actor 根据 $(o_t,z)$ 输出连续低层动作 $a_t$。
- memory 只在 planner 侧维护，不传给 actor。

论文实现同步和异步两种耦合方式：主文报告同步结果，异步方式更适合生产式部署，因为低层动作不会因为 planner 推理阻塞。

---

## 关键公式

### 公式1: [[Vision-Language Navigation|VLN 成功目标]]

$$
\max_{\theta}\ \mathbb{E}_{e\sim\mathcal{D}}
\left[
\mathbf{1}\{d(s_T,g)\le d_{\text{succ}}\}
\right]
$$

**含义**: 在 R2R-style VLN 中，训练/评估目标是让最终状态 $s_T$ 落在目标 $g$ 的成功半径 $d_{\text{succ}}$ 内。

**符号说明**:
- $e=(I,s_0,g)$: 一个导航 episode，包含路线指令、初始位姿和目标。
- $s_T$: episode 结束时 agent 的最终位姿。
- $d(s_T,g)$: 最终位置到目标的 geodesic 距离。
- $d_{\text{succ}}$: 成功阈值，R2R-CE 评估中为 3.0 m。

### 公式2: [[Action Planning|长程动作规划策略]]

$$
a_t=\pi(o_{\le t},a_{<t},g)
$$

**含义**: 下一步高层子任务 $a_t$ 不是只由当前图像决定，而依赖完整历史观测、历史动作和目标。

**符号说明**:
- $o_{\le t}$: 当前及历史观测。
- $a_{<t}$: 已执行或已预测的历史子任务。
- $g$: 语言目标。
- $\pi$: 高层 planner 策略。

### 公式3: [[Memory Harness|记忆压缩状态]]

$$
s_t=\Phi(o_t,\mathcal{M}_t,g)
$$

**含义**: 为避免完整历史上下文过长，Vesta 用 memory harness 得到压缩上下文 $\mathcal{M}_t$，再与当前观测和目标组成 planner 状态。

**符号说明**:
- $\Phi$: 状态构造/压缩函数。
- $o_t$: 当前观测。
- $\mathcal{M}_t$: 压缩后的多模态历史。
- $g$: 任务目标。

### 公式4: [[Memory Harness|历史步骤元组]]

$$
m_i=\langle i,\tau_i,o_i,a_i,g\rangle,\quad
\mathcal{H}_t=\{m_1,\ldots,m_{t-1}\},\quad
\mathcal{S}_t\subseteq\{1,\ldots,t-1\},\ |\mathcal{S}_t|\le K
$$

**含义**: 每个历史步骤记录索引、时间戳、观测、子任务和目标；图像历史通过采样集合 $\mathcal{S}_t$ 控制在 $K$ 帧以内。

**符号说明**:
- $m_i$: 第 $i$ 个历史记忆单元。
- $\tau_i$: 时间戳。
- $\mathcal{H}_t$: 第 $t$ 步之前的完整历史。
- $\mathcal{S}_t$: 被保留并送入 prompt 的历史图像索引集合。

### 公式5: [[Temporal Intersection-over-Union|离线规划评分]]

$$
\operatorname{tIoU}(\hat{a},a)
=
\frac{\operatorname{duration}(\hat{a}\cap a)}
{\operatorname{duration}(\hat{a}\cup a)}
$$

**含义**: 论文的离线 action planning benchmark 用时间段重叠而不是逐帧准确率评估预测子任务，预测 action 必须和 ground truth action 标签一致才计入重叠。

**符号说明**:
- $\hat{a}$: 预测动作段。
- $a$: 标注动作段。
- $\operatorname{duration}(\cdot)$: 时间长度。

### 公式6: [[Vision-Language-Action|Planner-Actor 耦合]]

$$
z'=\pi_{\text{plan}}(i_t,g,\mathcal{M}),\qquad
a_t=\pi_{\text{act}}(o_t,z)
$$

**含义**: 高层 planner 根据最新图像/目标/记忆输出文本子任务，低层 actor 只根据当前观测和子任务产生机器人动作。

**符号说明**:
- $z,z'$: 当前/新子任务文本。
- $\pi_{\text{plan}}$: Vesta planner。
- $\pi_{\text{act}}$: GR00T-N1.6 actor。
- $i_t,o_t$: planner/actor 使用的视觉观测。

---

## 关键图表

### Figure 1: Unified Generalist / 统一具身能力

![Figure 1](https://arxiv.org/html/2606.20905/x1.png)

**说明**: Vesta 将 localization、navigation、embodied reasoning、action planning 放进单个 generalist 模型，平均分数比 prior baseline 高 20+ points，比每类最强 baseline 组成的 oracle ensemble 也高 10+ points；真实机器人 memory-heavy 任务成功率提升 38.3%。

### Figure 2: Generalist Embodied Model / 多模态层级控制

![Figure 2](https://arxiv.org/html/2606.20905/x2.png)

**说明**: 展示 Vesta 作为高层 embodied model 的输入输出形态：多模态输入、历史上下文、层级控制，以及 localization、navigation、EQA、real-world planning 四类能力。

### Figure 3: Action Planning Task / 动作规划示例

![Figure 3](https://arxiv.org/html/2606.20905/x3.png)

**说明**: 给定整体目标和 memory context，planner 在每个决策点预测下一条子任务；图中省略中间步骤，强调“继续当前动作”与“切换到下一动作”的 transition 判断。

### Figure 4: SFT Data Mixture / 监督微调数据配比

![Figure 4](https://arxiv.org/html/2606.20905/x4.png)

**说明**: SFT 数据包括 Spatial Intelligence 27.1%、Navigation 21.8%、Grounding 20.8%、General VLM 16.2%、Embodied Reasoning 9.8%、Real Robots 4.3%。配比明显偏向空间落地能力。

### Figure 5: Navigation Evaluation / 导航评估示例

![Figure 5](https://arxiv.org/html/2606.20905/x5.png)

**说明**: Vesta 在 R2R-CE 中输出 turn、goto waypoint 和 stop 等高层导航动作，由底层导航后端执行。

### Figure 6: Real Robot Tasks / 真实机器人任务

![Figure 6](https://arxiv.org/html/2606.20905/x6.png)

**说明**: 三个双臂机器人任务分别是 Find Object、Count Fruits、Memorize Candy，均要求模型结合观察历史、任务进度和隐藏状态记忆。

### Figure 7: Real Robot Evaluation / 真实机器人成功率

![Figure 7](https://arxiv.org/html/2606.20905/x7.png)

**说明**: Vesta planner 在三类任务上显著超过 actor-only 和 Qwen3-VL planner；平均相对 actor-only 提升 38.3%，统计显著性超过 $4\sigma$。

### Figure 8: Hierarchical Execution / 层级执行流程

![Figure 8](https://arxiv.org/html/2606.20905/x8.png)

**说明**: Planner VLM 接收自然语言任务和连续图像，把文本与图像写入 memory，在任意时刻向 actor VLA 发送下一条子任务；actor 接收图像、状态和子任务并输出动作。

### Table 1: Embodied Benchmarks

| Benchmark | Vesta | RynnBrain 8B | RoboBrain 2.5 8B | Qwen3-VL 8B |
|-----------|------:|-------------:|-----------------:|------------:|
| Open-X VQA | 89.3 | 74.0 | 52.9 | 59.8 |
| SAT | 81.3 | 70.0 | 67.3 | 65.3 |
| VSI-Bench | 64.5 | 71.0 | 42.9 | 60.3 |
| MMSI-Bench | 40.8 | 39.6 | 29.4 | 30.8 |
| ERQA | 44.9 | 46.8 | 44.0 | 44.8 |
| MindCube-Tiny | 80.9 | 56.6 | 29.2 | 36.0 |
| CV-Bench | 88.1 | 87.7 | 87.6 | 86.2 |
| PAI-U | 57.9 | 56.6 | 55.0 | 57.9 |
| EgoTaskQA | 81.9 | 72.5 | 85.0 | 57.8 |
| RoboSpatial | 57.8 | 73.1 | 73.0 | 58.2 |
| **Cognition Avg.** | **68.7** | **64.8** | **56.6** | **55.7** |
| CrossPoint | 76.0 | 44.3 | 75.4 | 28.7 |
| EmbSpatial | 81.9 | 79.3 | 75.8 | 78.5 |
| Where2Place | 68.3 | 66.9 | 66.0 | 64.7 |
| RefSpatial | 59.9 | 59.2 | 60.5 | 53.4 |
| PointBench | 63.2 | 59.7 | 69.1 | 61.4 |
| **Localization Avg.** | **69.9** | **61.9** | **69.4** | **57.3** |

**说明**: Vesta 在 cognition 平均分和 localization 平均分上领先；个别单项如 VSI-Bench、RoboSpatial、PointBench 并非第一，但整体更均衡。

### Table 2: Real-World Action Planning

| Model | AgiBot CD | AgiBot PF | AgiBot SP | AgiBot FS | AgiBot RS | Egocentric-Human Diverse Tasks | Avg. |
|-------|----------:|----------:|----------:|----------:|----------:|-------------------------------:|-----:|
| RoboBrain-2.5-8B | 35.3 | 81.6 | 15.9 | 38.3 | 33.0 | 27.0 | 38.5 |
| Qwen3-VL-8B | 36.7 | 67.8 | 18.1 | 22.1 | 30.2 | 26.7 | 33.6 |
| RynnBrain-8B | 38.7 | 69.5 | 16.0 | 18.4 | 32.4 | 26.0 | 33.5 |
| **Vesta** | **74.4** | **91.0** | **64.0** | **80.3** | **82.3** | **60.5** | **75.4** |

**说明**: 离线 action planning benchmark 覆盖 160 个 zero-shot episodes，Vesta 的 temporal IoU 平均分显著高于同尺寸基线。

### Table 3: Navigation in R2R-CE

| Model | SR ↑ | NE ↓ | OS ↑ | SPL ↑ |
|-------|-----:|-----:|-----:|------:|
| RynnBrain-8B | 0.0 | 8.86 | 0.0 | 0.0 |
| RoboBrain-2.5-8B | 0.0 | 9.03 | 0.0 | 0.0 |
| Qwen3-VL-8B | 0.0 | 8.83 | 0.0 | 0.0 |
| UniNaVid | 47.0 | 5.58 | 53.3 | 42.7 |
| InternVLA-N1-8B | 55.4 | **4.89** | 60.6 | **52.1** |
| **Vesta** | **55.5** | 5.16 | **61.4** | 50.8 |

**说明**: Vesta 与导航 specialist [[InternVLA-N1]] 基本持平，在 SR 和 OS 上略优，在 NE 和 SPL 上略低；通用 VLM 基线在该闭环导航设置中失败。

### Table 4: Generalist vs. Specialist Training

| Training Mix | R2R-CE SR ↑ | R2R-CE SPL ↑ | Embodied Cognition ↑ | Embodied Localization ↑ | Avg. |
|--------------|------------:|-------------:|---------------------:|-------------------------:|-----:|
| Nav-only specialist | 54.1 | 49.8 | 0 | 0 | 26.0 |
| Embodied-only specialist | 0 | 0 | 64.3 | 68.3 | 33.2 |
| **Vesta unified** | **55.5** | **50.8** | **70.5** | **69.9** | **61.7** |

**说明**: 统一训练不只是折中，而是在 navigation 和 embodied benchmark 上都超过对应 specialist，说明多任务数据有正迁移。

### Table 5: Ablation Study

| Setting | Trans. | Exec. | Overall |
|---------|-------:|------:|--------:|
| Transition Sampling 1x | 58.1 | **94.9** | 73.6 |
| Transition Sampling 2x | 68.1 | 90.0 | **75.9** |
| Transition Sampling 3x | **69.1** | 88.7 | 75.3 |
| Memory: Image | 61.0 | 70.4 | 63.1 |
| Memory: Text | 40.3 | 83.1 | 49.7 |
| Memory: Image-Text Uniform | 68.1 | **90.0** | **75.9** |
| Memory: Image-Text Recency-Bias | **68.9** | 87.5 | 75.3 |

**说明**: Transition step 过少会让模型不会切换动作，2x oversampling 最均衡。Memory 只用图像或只用文本都不理想，图像 + 文本历史最好。

### Table 6: Robot Task Details

| Task | # Train | # Eval | Success criterion |
|------|--------:|-------:|------------------|
| Count Fruits | 401 | 20 | basket closed, with the correct number of fruits inside |
| Find Object | 774 | 20 | object placed on the table, and no drawer opened twice in the trajectory |
| Memorize Candy | 500 | 20 | candy placed in the box, box closed, and box placed on the tray whose color matches the candy |

**说明**: 三类任务都显式考察历史依赖。尤其 Memorize Candy 在盒子关闭后必须记住糖果颜色。

### Table 7: SFT Training Details

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-VL-8B-Instruct |
| Trainable parameters | full, vision + projector + LM |
| Precision | pure bf16 |
| Learning rate | 1e-5 |
| LR schedule | cosine |
| Weight decay | 0 |
| Warmup ratio | 0.1 |
| Max grad norm | 1.0 |
| Sequence packing | enabled |
| Cutoff length | 8,192 |
| Max video frames | 32 |
| Per-device batch size | 2 |
| Gradient accumulation | 2 |
| Epochs | 1 |
| Infrastructure | H100 GPUs, FSDP full-shard, FlashAttention-2, gradient checkpointing |

**说明**: 主文第 3 节写到 batch size 256、128 H100、weight decay 0.01；Appendix Table 7 则列出 weight decay 0。这里保留 Table 7 原值，同时在“实现细节”里标注这个不一致点。

### Table 8: Actor Training Details

| Parameter | Value |
|-----------|-------|
| Base model | Gr00t-N1.6 |
| Batch size | 512 |
| Learning rate | 1e-5 |
| Training steps | 10,000 |
| Weight decay | 1e-5 |
| Action horizon | 50 |
| Color jitter | brightness 0.3, contrast 0.4, saturation 0.5, hue 0.08 |

**说明**: actor 只在真实机器人评估中与 planner 通过自然语言子任务交互，训练阶段两者分开。

---

## 实验

### 数据集

| 数据/Benchmark | 规模 | 特点 | 用途 |
|----------------|------|------|------|
| Spatial Intelligence SFT | 27.1% | 空间关系、位置理解、具身空间问题 | 训练 |
| Navigation SFT | 21.8% | R2R/RxR/ScaleVLN 渲染轨迹 | 训练 |
| Grounding SFT | 20.8% | boxes/points/text grounding | 训练 |
| General VLM SFT | 16.2% | 通用视觉语言能力 | 训练 |
| Embodied Reasoning SFT | 9.8% | affordance、placement、task progress | 训练 |
| Real Robots SFT | 4.3% | 真实机器人子任务规划 | 训练 |
| R2R-CE val-unseen | 1839 episodes | held-out [[Matterport3D]] scenes | 导航评估 |
| AgiBot + Egocentric Human-Hand | 160 episodes | zero-shot action planning | 离线规划评估 |
| YAM bimanual robot tasks | 3 tasks × 20 eval | 真实双臂机器人 | 真实执行评估 |

### 实现细节

- **Base planner**: [[Qwen3-VL]]-8B-Instruct。
- **训练方式**: full-parameter SFT，vision tower、projector、LM 全部训练。
- **精度/并行**: pure bf16，FSDP full-shard，无 parameter offload，FlashAttention-2，gradient checkpointing。
- **上下文**: cutoff length 8192，最多 32 个 video frames。
- **主文训练规模**: 1 epoch，learning rate 1e-5，128 H100，batch size 256；主文提 weight decay 0.01，但 Table 7 写 weight decay 0，存在记录不一致。
- **真实机器人 actor**: fine-tuned [[GR00T-N1]]-1.6，batch size 512，10k steps，action horizon 50。

### 评估设置

- **Embodied reasoning/localization**: 在 Open-X VQA、SAT、VSI-Bench、MMSI-Bench、ERQA、MindCube-Tiny、CV-Bench、PAI-U、EgoTaskQA、RoboSpatial、CrossPoint、EmbSpatial、Where2Place、RefSpatial、PointBench 上评估。
- **Navigation**: NavSuite 的 R2R val-unseen，闭环执行，500-step budget，报告 SR、SPL、OS、NE。
- **Offline planning**: 以 MCQ 形式评估每个 decision point 的下一子任务，使用 temporal IoU。
- **Real robot**: actor-only、actor + Qwen3-VL planner、actor + Vesta planner 三种配置，每个任务 20 个样本。

### 关键发现

1. **统一模型不是折中**: Vesta unified mix 在 navigation 和 embodied 两类任务上都超过对应 specialist mix。
2. **导航能力来自数据和接口，不只是 base VLM**: Qwen3-VL、RynnBrain、RoboBrain 在 R2R-CE 闭环导航中 SR 为 0，而 Vesta 接近 InternVLA-N1。
3. **记忆必须多模态**: image-only 不知道任务进度，text-only 容易依赖历史文本 shortcut；image-text 才最稳。
4. **动作切换是瓶颈**: transition 阶段远少于 execution 阶段，2x transition oversampling 能明显提高整体 planning。

---

## 批判性思考

### 优点

1. **问题设定很实用**: 论文没有只停在 benchmark，而是明确面向“planner VLM + actor VLA”的真实机器人架构。
2. **数据配方透明**: Figure 4 和 Table 7/8 给出了足够清楚的训练配比和超参，便于复现思路。
3. **消融结果有信息量**: generalist vs specialist、transition sampling、memory modality 三组消融直接回答了“为什么统一”和“为什么记忆这样做”。
4. **真实机器人闭环有说服力**: 虽然规模不大，但任务专门设计了隐藏状态和历史依赖，能测出 planner 的实际价值。

### 局限性

1. **真实机器人评估规模仍小**: 每个任务 20 个样本，且没有和专门为机器人 planning 调过的 specialist baseline 对比。
2. **actor 错误混入 planner 评估**: 作者也承认失败多数来自 actor motion/language-following，真实成功率无法完全隔离 planner 本身。
3. **Memory harness 仍是手工设计**: uniform/recency sampling 加文本 log 简洁有效，但不是可学习记忆，长期部署下如何巩固、遗忘和检索还未解决。
4. **Table 7 与正文 weight decay 不一致**: 主文第 3 节写 weight decay 0.01，Appendix 表格写 0；复现时需要查代码或作者更新。
5. **8B 单尺度结论有限**: 论文主要报告 Qwen3-VL-8B 规模，尚不能说明更大/更小模型的数据配比和 memory 规律是否相同。

### 潜在改进方向

1. 学习式 memory manager：让模型决定保留哪些图像、文本、事件和空间状态，而不是固定采样。
2. Planner-actor joint evaluation：把 planner 预测质量、actor 可执行性、语言粒度适配拆开评估。
3. 更强真实机器人基线：加入 OpenVLA/pi0/pi0.5 等 planner 或 VLA 系统的同平台对比。
4. 长期家庭/仓储任务：从 episodic memory 扩展到跨 episode 的 lifelong memory。

### 可复现性评价

- [ ] 代码开源
- [ ] 模型权重开源
- [x] 训练细节较完整
- [x] 数据来源大体清楚
- [x] 真实机器人任务定义清楚
- [ ] 训练数据清单与处理脚本完整公开

---

## 关联笔记

### 基于
- [[Qwen3-VL]]: Vesta 的 8B base VLM。
- [[Vision-Language-Action]]: 层级系统中的 actor 模型范式。
- [[GR00T-N1]]: 真实机器人实验中使用的低层 actor。
- [[R2R]]: R2R-CE 导航评估来源。

### 对比
- [[InternVLA-N1]]: 导航 specialist，Vesta 在 R2R-CE 上与其接近。
- [[Qwen-VLA]]: 具身 VLA 系统相关背景。
- [[Pi0]] / [[Pi05]]: 通用机器人控制和 open-world generalization 的代表工作。

### 方法相关
- [[Memory Harness]]: 本文长程 planning 的关键机制。
- [[Embodied Reasoning]]: 本文统一能力之一。
- [[Action Planning]]: 离线和真实机器人评估核心任务。
- [[Chain-of-Thought]]: planner 结构化推理输出格式。
- [[Temporal Intersection-over-Union]]: 离线 planning 评分方式。

### 数据/平台相关
- [[Objects365]], [[COCO]], [[LVIS]]: grounding 基础数据。
- [[Habitat]], [[Matterport3D]]: R2R-CE 导航评估环境。
- [[Egocentric Observation]]: planner 输入视角。

---

## 速查卡片

> [!summary] Vesta
> - **核心**: 单个 8B generalist planner 统一具身定位、导航、推理和动作规划。
> - **方法**: Qwen3-VL 全参数 SFT + 空间偏置数据混合 + 图像/文本 memory harness。
> - **结果**: 平均超过 prior specialist 20+ points；真实机器人 memory-heavy 任务相对 actor-only 提升 38.3%。
> - **风险点**: 真实机器人样本少、actor/planner 错误耦合、memory 仍是手工采样。
> - **链接**: https://arxiv.org/abs/2606.20905

---

*笔记创建时间: 2026-06-30*
