---
title: "M3E: Continual Vision-and-Language Navigation via Mixture of Macro and Micro Experts"
method_name: "M3E"
authors: [Yongliang Jiang, Huaidong Zhang, Xuandi Luo, Shengfeng He]
year: 2026
venue: ICLR
tags: [vision-language-navigation, continual-learning, mixture-of-experts, llm-agent, embodied-ai, navigation]
zotero_collection: _待整理
image_source: local
project_page: https://yongliangjiang.top/m3e
created: 2026-07-03
---

# 论文笔记：M3E: Continual Vision-and-Language Navigation via Mixture of Macro and Micro Experts

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[M3E]] |
| 机构 | South China University of Technology; Singapore Management University |
| 作者 | Yongliang Jiang, Huaidong Zhang, Xuandi Luo, Shengfeng He |
| 年份 | 2026 |
| 会议/来源 | ICLR 2026 |
| 项目主页 | [M3E](https://yongliangjiang.top/m3e) |
| 任务 | [[Vision-and-Language Navigation]] continual learning |
| 相关工作 | [[NaviLLM]]、[[R2R]]、[[REVERIE]]、[[Mixture-of-Experts]]、[[LoRA]] |

---

## 一句话总结

> M3E 是一个 replay-free continual VLN 框架，用 Macro Router 做全局场景/拓扑级专家路由，用 Micro Router 做 token 级指令-视觉对齐路由，再用动态 MoE 动量更新保护跨环境知识。

---

## 核心贡献

1. **定义 VLNCL 设置**：将 [[Vision-and-Language Navigation]] 放到 domain-incremental continual learning 中，要求 agent 在不回放旧轨迹的情况下持续适应新环境。
2. **Macro-Micro 双路由 MoE**：把导航推理拆成 scene-level planning 和 token-level grounding 两层，避免把全局场景策略与局部指令对齐混在同一个路由器里。
3. **Macro Router**：从在线 [[Cognitive Map]] 中取 visited/frontier nodes，构图后用 [[Graph Neural Network]] 做拓扑传播，再用指令 embedding 做 task-focused aggregation，输出全局专家权重。
4. **Micro Router**：从 [[LLM]] token hidden state 生成 token- expert weights，负责局部语义 grounding，例如动词、房间名、物体名对应不同专家。
5. **Dynamic MoE Momentum Update**：按当前 task 的专家贡献选择 top-K 重要专家，对重要专家更激进更新，对非重要专家保守更新，从而兼顾 plasticity 与 stability。
6. **实验结论**：在 R2R 和 REVERIE domain-incremental setting 中，M3E 在 AvgSR/AvgSPL、BWT/FWT 上优于 fine-tuning、regularization 和 rehearsal baselines。

---

## 问题背景

### 要解决的问题

VLN agent 需要在室内环境中根据自然语言指令导航到目标位置。已有方法在固定数据集上表现不错，但部署到连续变化的新场景时，会遇到两个问题：

- **适应新域**：新建筑、新布局、新房间类型带来不同的视觉/拓扑分布。
- **灾难性遗忘**：继续训练新场景会破坏旧场景中的导航能力。

现有 continual VLN 多依赖 rehearsal buffer 保存旧轨迹，但这会带来存储、计算、隐私和可扩展性问题。M3E 要解决的是：**不回放旧数据，如何让 LLM-based VLN agent 持续学习新环境，同时保留旧能力？**

### 现有方法局限

- 普通 fine-tuning 容易遗忘，R2R 上 BWT 为 -5.42。
- Regularization 方法如 L2/EWC 能缓解参数漂移，但对 VLN 的序列规划和空间 grounding 不够针对。
- Rehearsal 方法如 ER/PerR/ESR/Dual-SR 依赖旧轨迹存储，不满足 replay-free。
- 视觉问答里的 CL-MoE 思路不能直接迁移到 VLN，因为 VLN 还需要部分可观测下的连续路径规划与拓扑推理。

### 本文动机

作者认为 continual VLN 的遗忘来自两个层次被混在一起：

- **全局场景推理**：办公室、住宅、长走廊、房间连接关系等结构性策略。
- **局部感知对齐**：当前指令 token 与当前视觉观测之间的细粒度 grounding。

因此 M3E 设计了 Macro Router + Micro Router，让不同层次的信息控制不同的专家选择。

---

## 方法详解

### 总体架构

M3E 建在 end-to-end trainable [[LLM]] navigation agent 上，基础 agent 包含：

- **Scene Encoder**：对全景观测提取多视角视觉特征，更新 cognitive map。
- **LLM Backbone**：decoder-only Transformer，根据 instruction、navigation history、scene embeddings 输出 action query hidden state。
- **Action Head**：对 reachable candidate viewpoints 打分，生成下一步动作或 STOP。

M3E 的核心改造是：把 LLM 中标准 FFN 替换为 [[MoE-LoRA]] 层，并用 Macro/Micro 双路由产生专家权重。

```text
Panoramic image + instruction + navigation history
  -> Scene encoder / LLM token hidden states
  -> Macro Router: cognitive map -> topology-aware scene routing
  -> Micro Router: token hidden states -> token-level semantic routing
  -> Dual-router fusion: w = beta w_ma + (1-beta) w_mi
  -> MoE-LoRA expert selection
  -> Dynamic MoE Momentum Update across tasks
```

### VLNCL 问题设定

普通 VLN 给定室内导航图 $G=(V,E)$、指令 $I=(w_1,\dots,w_L)$ 和起点 $v_0$，agent 在每步观察全景并选择动作，直到 STOP 或达到预算。

Continual VLN 进一步给定任务流 $\{D_t\}_{t=1}^T$。每个 $D_t$ 来自新的 scene/domain。训练时只能访问当前任务数据，不能 replay 旧数据；训练后要评估所有已见任务，衡量适应与遗忘。

### Macro Router：全局场景级路由

Macro Router $G_{\text{ma}}$ 捕捉的是当前环境的结构性信息。它不是直接平均视觉特征，而是使用 Topology-Aware, Task-Focused Routing：

1. **Global Map Representation**：从 online cognitive map 取 visited nodes 与 frontier nodes，构建稀疏邻接矩阵 $\hat{A}_t$ 和节点特征 $X$。
2. **Topology-aware Propagation**：用 GNN 在拓扑图上传播节点信息，得到 $H_{\text{gnn}}$。
3. **Task-focused Aggregation**：用 instruction embedding 作为 query，对节点表示做 attention，得到 scene representation $s_t$。
4. **Scene-level Routing**：用 MLP + Softmax 输出 macro expert weights $w^{\text{ma}}$。

这个路由器对应“我处在什么类型的环境/拓扑结构中，应该用哪类导航策略专家”。

### Micro Router：局部 token 级路由

Micro Router $G_{\text{mi}}$ 从每个 token hidden state $h$ 生成 expert weights：

- 动词 token 如 “go” 更偏向 action reasoning 专家。
- 名词 token 如 “kitchen/door” 更偏向 object/scene understanding 专家。
- 它捕捉跨场景共享的局部语言-视觉 grounding primitives。

这个路由器对应“当前词/局部观测需要哪类语义专家”。

### Dual-Router Fusion

Macro 和 Micro 通过凸组合融合：

$$
w = \beta w^{\text{ma}} + (1-\beta) w^{\text{mi}}
$$

实现中 $\beta=0.3$，即路由决策更偏向 Micro Router 的局部即时观测，同时保留 Macro Router 的全局拓扑先验。这个设计也让模型对 cognitive map 噪声更鲁棒。

### Dynamic MoE Momentum Update

在 task $t$ 上训练后，M3E 不简单覆盖所有专家参数，而是先统计当前 task 中每个专家的 workload / contribution：

1. 对所有 token 的 fused routing weights 求和，得到专家使用量 $u$。
2. 归一化为贡献分布 $p$。
3. 选择 top-K 重要专家 $E^{\text{imp}}_t$。
4. 重要专家更多靠近当前任务参数 $\Phi_t$，非重要专家更多保留历史参数 $\Theta_{t-1}$。

直觉是：当前任务真正用到的专家应快速适应；没怎么被用到的专家应该少动，保留旧知识。

---

## 关键公式

### 公式1：[[Vision-and-Language Navigation|VLN 训练目标]]

$$
\min_\theta \mathbb{E}_{(G,I)}\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T}\mathcal{L}(s_t,a_t)
\right]
$$

**含义**：在导航图和指令分布上优化策略，使轨迹动作损失最小。

**符号说明**：

- $G=(V,E)$：室内导航图。
- $I$：自然语言指令。
- $\tau$：agent 执行策略产生的轨迹。
- $\mathcal{L}$：imitation learning、reinforcement learning 或混合损失。

### 公式2：[[Continual Learning|VLNCL 参数更新]]

$$
\Phi_t = \arg\min_\theta
\mathbb{E}_{(G,I)\sim D_t}
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_k \mathcal{L}(s_k,a_k)
\right],
\qquad
\Theta_t = C(\Theta_{t-1},\Phi_t)
$$

**含义**：当前任务先得到 task-specific 参数 $\Phi_t$，再通过 consolidation operator $C$ 融合成长期参数 $\Theta_t$。

### 公式3：[[Cognitive Map|节点特征矩阵]]

$$
X = [x_v^\top]_{v\in V_t}\in\mathbb{R}^{N\times d}
$$

**含义**：把 cognitive map 中 visited/frontier nodes 的节点特征组成矩阵。

### 公式4：[[Graph Neural Network|拓扑感知传播]]

$$
H_{\text{gnn}} = \operatorname{GNN}(\hat{A}_t,X)\in\mathbb{R}^{N\times d_h}
$$

**含义**：在稀疏邻接图 $\hat{A}_t$ 上做消息传递，得到 topology-aware node representations。

### 公式5：[[Attention Mechanism|任务聚焦聚合]]

$$
\alpha_v = \operatorname{softmax}_v(h_v^\top \operatorname{Emb}_{\text{Ins}}),
\qquad
s_t = \sum_{v\in V_t}\alpha_v h_v\in\mathbb{R}^{d_h}
$$

**含义**：用指令 embedding 关注与当前任务相关的地图节点，得到 scene representation。

### 公式6：[[Macro Router|宏观专家权重]]

$$
w^{\text{ma}} = G_{\text{ma}}(s_t)
= \operatorname{Softmax}(\operatorname{MLP}(s_t))\in\mathbb{R}^{n}
$$

**含义**：Macro Router 根据全局场景表示输出专家分布。

### 公式7：[[Micro Router|微观专家权重]]

$$
w^{\text{mi}} = G_{\text{mi}}(h)
= \operatorname{Softmax}(\operatorname{MLP}(h))\in\mathbb{R}^{n}
$$

**含义**：Micro Router 根据 token hidden state 输出 token-wise expert distribution。

### 公式8：[[Dual-Router Fusion|双路由融合]]

$$
w = \beta w^{\text{ma}} + (1-\beta)w^{\text{mi}}\in\mathbb{R}^{n}
$$

**含义**：将全局场景先验与局部 token 判断融合，得到最终 MoE expert routing weights。

### 公式9-10：[[Expert Contribution|专家 workload 与贡献]]

$$
u = \sum_{x\in D_t} w(x)\in\mathbb{R}^{n}
$$

$$
I_t(E_i) \triangleq p[i]
= \frac{u[i]}{\sum_{j=1}^{n}u[j]}
=
\frac{\sum_{x\in D_t} w_i(x)}
{\sum_{j=1}^{n}\sum_{x\in D_t} w_j(x)}
$$

**含义**：统计当前任务中每个专家被路由使用的总量，并归一化为专家贡献。

### 公式11：[[Top-K Routing|重要专家选择]]

$$
E^{\text{imp}}_t = \operatorname{TopK}(p,K),
\qquad
E^{\text{non}}_t=\{E_1,\ldots,E_n\}\setminus E^{\text{imp}}_t
$$

**含义**：贡献最高的 K 个专家被视为当前任务重要专家。

### 公式12-13：[[Dynamic MoE Momentum Update|动态动量整合]]

$$
\lambda_i =
\begin{cases}
\gamma, & E_i\in E^{\text{imp}}_t,\\
1-\gamma, & E_i\in E^{\text{non}}_t,
\end{cases}
\qquad \gamma\in[0,0.5)
$$

$$
\Theta_t = \Lambda\odot\Theta_{t-1} + (1-\Lambda)\odot\Phi_t
$$

**含义**：重要专家使用较小 $\lambda_i$，更靠近当前任务参数 $\Phi_t$；非重要专家使用较大 $\lambda_i$，更多保留历史参数 $\Theta_{t-1}$。

### 公式14：[[Continual Learning|BWT 与 FWT]]

$$
\operatorname{BWT}
= \frac{1}{T-1}\sum_{i=0}^{T-1}(R_{T,i}-R_{i,i}),
\qquad
\operatorname{FWT}
= \frac{1}{T-1}\sum_{i=2}^{T}(R_{i-1,i}-R_{i,i})
$$

**含义**：BWT 衡量学新任务后对旧任务的影响；FWT 衡量训练某任务之前对该任务的零样本迁移。

### 公式15：[[Continual Learning|Seen tasks 累计平均 SPL]]

$$
A_t = \frac{1}{t+1}\sum_{i=0}^{t}R_{t,i}
$$

**含义**：跟踪到 stage $t$ 时所有已见任务的平均 SPL，用于观察稳定性趋势。

---

## 关键图表

> 图表来源：本地 PDF 裁剪。论文共 Figure 4 张、Table 10 张。

### Figure 1: Overall architecture of M3E

![[M3E_fig1_architecture.png]]

**说明**：M3E 用 Macro Router 捕捉 cognitive map 上的全局拓扑/场景结构，用 Micro Router 处理 token hidden states 的局部 grounding，两者融合后路由 MoE-LoRA 专家，并在任务间通过 Dynamic MoE Momentum Update 巩固知识。

### Figure 2: Performance trend analysis

![[M3E_fig2_trend.png]]

**说明**：M3E 在 cumulative stability 上保持稳定并后期上升，而 Finetune 随任务增加持续下降；在当前任务 plasticity 上，M3E 也高于 Finetune。

### Figure 3: Lifelong transfer matrix

![[M3E_fig3_transfer_matrix.png]]

**说明**：M3E 的下三角矩阵颜色保持较深，表示旧任务表现能被保留；Finetune 的下三角迅速变浅，直观显示灾难性遗忘。

### Figure 4: Expert specialization

![[M3E_fig4_expert_specialization.png]]

**说明**：Macro Router 激活随 domain 变化，说明其捕捉 scene-level context；Micro Router 激活更稳定，说明其捕捉跨任务共享的语言/视觉局部语义。

### Table 1: R2R domain-incremental learning

| Method | Strategy | AvgSR | AvgSPL | AvgNE | BWT | FWT |
|--------|----------|-------|--------|-------|-----|-----|
| Finetune | RF | 63.28 | 59.08 | 3.72 | -5.42 | -2.41 |
| L2 | Reg | 58.78 | 56.20 | 4.23 | -5.10 | -3.43 |
| EWC | Reg | 64.15 | 60.21 | 3.60 | -3.50 | -2.80 |
| ER | Reh | 66.35 | 62.10 | 3.45 | -1.50 | 0.50 |
| PerR | Reh | 67.05 | 62.93 | 3.38 | -1.35 | 0.62 |
| ESR | Reh | 68.12 | 63.88 | 3.25 | -1.10 | 0.85 |
| Dual-SR | Reg+Reh | 70.25 | 65.40 | 3.05 | -0.45 | 1.85 |
| **M3E** | RF | **71.92** | **66.96** | **2.95** | **0.04** | **2.15** |

**表格说明**：M3E 在不 replay 旧轨迹的情况下优于 rehearsal baselines，并且 BWT 由负转接近零/正。

### Table 2: REVERIE domain-incremental learning

| Method | SR | SPL | BWT | FWT |
|--------|----|-----|-----|-----|
| Finetune | 50.12 | 39.86 | -16.91 | -10.26 |
| **M3E** | **51.23** | **48.30** | **-5.91** | **-8.09** |

**表格说明**：REVERIE 更强调 object-centric grounding，M3E 仍显著提升 SPL 并减轻遗忘。

### Table 3: Full val-unseen post training

| Dataset | Method | State | val-seen SR/SPL | test-unseen SR/SPL |
|---------|--------|-------|-----------------|--------------------|
| R2R | HAMT | Before | 70.13 / 66.30 | 62.30 / 57.57 |
| R2R | HAMT | After | 57.59 / 52.67 | 58.01 / 52.06 |
| R2R | NaviLLM | Before | 73.36 / 68.93 | 67.60 / 59.84 |
| R2R | NaviLLM | After | 71.11 / 67.97 | 63.52 / 58.27 |
| R2R | **M3E** | After | **75.51 / 71.94** | **67.28 / 61.33** |
| REVERIE | HAMT | Before | 43.29 / 40.19 | 33.41 / 26.67 |
| REVERIE | HAMT | After | 42.38 / 37.90 | 28.67 / 24.23 |
| REVERIE | NaviLLM | Before | 50.11 / 47.24 | 39.53 / 32.51 |
| REVERIE | NaviLLM | After | 38.93 / 36.05 | 34.98 / 28.93 |
| REVERIE | **M3E** | After | **46.24 / 42.93** | **37.79 / 31.64** |

**表格说明**：在 bulk post-training 设置中，M3E 对 val-seen 的遗忘更小，R2R 甚至有正向提升。

### Table 4: R2R component ablation

| Micro | Macro | Momentum | AvgSR | AvgSPL | AvgNE | BWT | FWT |
|-------|-------|----------|-------|--------|-------|-----|-----|
| No | No | No | 63.28 | 59.08 | 3.72 | -5.42 | -2.41 |
| No | No | Yes | 61.52 | 57.90 | 3.95 | -2.15 | -2.80 |
| No | Yes | No | 64.20 | 60.15 | 3.68 | -5.65 | 1.80 |
| No | Yes | Yes | 65.15 | 61.05 | 3.60 | -0.12 | 1.92 |
| Yes | No | No | 65.51 | 61.23 | 3.55 | -5.81 | -1.50 |
| Yes | No | Yes | 66.72 | 62.34 | 3.48 | -0.35 | -1.48 |
| Yes | Yes | No | 67.83 | 64.12 | 3.34 | -6.05 | 2.15 |
| **Yes** | **Yes** | **Yes** | **71.92** | **66.96** | **2.95** | **0.04** | **2.15** |

**表格说明**：双路由提升 plasticity，但必须加 momentum 才能保留旧知识；三者组合最优。

### Table 5: R2R splits

| Split | Scans | Paths | Instructions | Notes |
|-------|-------|-------|--------------|-------|
| Train | 61 | 4,675 | 15,060 | standard training set |
| Val-seen | 56 | 340 | 1,021 | overlapping scans, new paths/instructions |
| Val-unseen | 11 | 783 | 2,349 | disjoint scans from all other splits |
| Test-unseen | 18 | 1,391 | 4,173 | disjoint scans from all other splits |

**表格说明**：continual stream 从 val-unseen scene partition 构造。

### Table 6: R2R val-unseen incremental entries

| Scan ID | # Trajectories | Train entries | Val entries |
|---------|----------------|---------------|-------------|
| 2azQ1b91cZZ | 100 | 80 | 20 |
| oLBMNvg9in8 | 100 | 80 | 20 |
| TbHJrupSAjP | 100 | 80 | 20 |
| zsNo4HB9uLZ | 100 | 80 | 20 |
| X7HyMhZNoso | 98 | 78 | 20 |
| QUCTc6BB5sX | 93 | 74 | 19 |
| EU6Fwq7SyZv | 64 | 51 | 13 |
| Z6MFQCViBuw | 60 | 48 | 12 |
| x8F5xyUWy9e | 47 | 37 | 10 |
| 8194nk5LbLH | 15 | 12 | 3 |
| pLe4wQe7qrG | 6 | 4 | 2 |
| Total | 883 | 624 | 159 |

**表格说明**：小于等于 10 条验证轨迹的 scan 会从 incremental benchmark 中排除。

### Table 7: Cognitive map noise robustness

| Setting | Noise Type | SR | SPL | ΔSR |
|---------|------------|----|-----|-----|
| Full M3E | None | 71.92 | 66.96 | - |
| M3E | Edge Drop 10% | 71.65 | 66.58 | -0.27 |
| M3E | Edge Drop 30% | 70.83 | 65.90 | -1.09 |
| M3E | No Frontier Nodes | 70.42 | 65.21 | -1.50 |
| Micro-only | Macro disabled | 66.72 | 62.34 | -5.20 |

**表格说明**：即使 cognitive map 有噪声，Macro Router 仍提供有价值结构先验；完全禁用 Macro 会明显掉点。

### Table 8: Expert count ablation

| # Experts | Trainable Params | AvgSR | AvgSPL | BWT | FWT |
|-----------|------------------|-------|--------|-----|-----|
| 4 | 226.0M | 70.15 | 65.10 | -1.12 | 1.85 |
| **6** | **310.6M** | **71.92** | **66.96** | **0.04** | **2.15** |
| 8 | 395.1M | 72.05 | 67.08 | 0.05 | 2.18 |

**表格说明**：6 个专家已达到较好容量/效率折中，8 个专家收益很小。

### Table 9: Comparison with TTA methods

| Method | Type | Test-time Input | AvgSPL | BWT |
|--------|------|-----------------|--------|-----|
| Finetune | CL | Frozen | 59.08 | -5.42 |
| FSTTA | TTA | Unlabeled Stream | 63.45 | -6.04 |
| FeedTTA | TTA | Stream + Oracle Feedback | 63.84 | -5.78 |
| **M3E** | CL | Frozen | **66.96** | **0.04** |

**表格说明**：TTA 能提高适应，但在 lifelong setting 中破坏共享参数，BWT 更差。

### Table 10: Inference latency

| Map Size | Macro Router | Micro Router | LLM Backbone | Relative Overhead |
|----------|--------------|--------------|--------------|-------------------|
| 10 nodes | 0.80 ms | 0.17 ms | 152.10 ms | +0.5% |
| 50 nodes | 3.07 ms | 0.17 ms | 152.48 ms | +2.0% |
| 100 nodes | 6.10 ms | 0.17 ms | 152.80 ms | +4.0% |

**表格说明**：Macro Router 的 GNN 开销相对 7B LLM decoding 很小，50 节点时约 2%。

---

## 实验

### 设置

| 项目 | 设置 |
|------|------|
| Backbone | [[NaviLLM]] style LLM-based navigation agent |
| Experts | 6 experts |
| Routing | Top-K = 2, fusion weight $\beta=0.3$ |
| Momentum | $\gamma=0.4$ for task-critical experts |
| Adapter | [[LoRA]] rank 16, alpha 32 |
| Optimizer | AdamW |
| LR | $2\times10^{-5}$ |
| Batch size | 4 |
| Training per task | 5 epochs |
| Datasets | [[R2R]], [[REVERIE]] |
| Main metrics | SR, SPL, NE, OSR, BWT, FWT |

### 关键发现

- M3E 在 R2R 上达到 AvgSR 71.92、AvgSPL 66.96、BWT 0.04，是 replay-free 方法中最强。
- 在 REVERIE 上，M3E 比 Finetune 提升 SPL 8.44，并将 BWT 从 -16.91 改善到 -5.91。
- Macro+Micro 双路由若没有 momentum 仍会遗忘；momentum 是稳定性来源，双路由是适应性与泛化来源。
- Macro Router 的专家激活随 domain 变化，Micro Router 的专家激活跨任务稳定，实验证明了宏观/微观解耦。

---

## 批判性思考

### 优点

1. **任务结构建模很贴合 VLN**：全局拓扑推理和局部 token grounding 的拆分比单一路由 MoE 更符合导航任务本质。
2. **Replay-free 有现实意义**：无需存旧轨迹，降低存储、隐私和持续部署成本。
3. **实验覆盖遗忘与迁移**：同时报告 AvgSR/SPL、BWT/FWT、transfer matrix、trend curve，不只看最终导航成功率。
4. **开销可控**：Macro Router 在 50 nodes 下只带来约 3ms，主要计算仍在 7B LLM。

### 局限

1. **仍在离散仿真 VLN setting 中验证**：真实机器人、连续控制、SLAM 噪声和动态障碍仍未充分验证。
2. **专家数量固定**：6 experts 在 R2R/REVERIE 上够用，但开放环境下可能需要动态扩展或合并专家。
3. **Macro Router 依赖 cognitive map 质量**：虽然噪声实验显示鲁棒，但真实场景中的定位漂移、闭环错误会更复杂。
4. **与更大 VLM/LLM agent 的兼容性未充分展开**：7B backbone 下开销可控，但更大模型、多模态输入更多时 routing 的收益/成本还需验证。

### 潜在改进方向

1. 动态专家分配：按新 domain 自动创建、合并或蒸馏专家。
2. 与真实 SLAM / ROS local planner 结合，把 M3E 作为 waypoint-level planner。
3. 混合 privacy-aware replay：只保留抽象地图/专家统计，而不是 raw trajectories。
4. 用 semantic map / object memory 强化 Macro Router 的 scene-level 表达。

---

## 关联笔记

### 方法相关

- [[Vision-and-Language Navigation]]：任务范式。
- [[Continual Learning]]：任务流与 BWT/FWT 评估基础。
- [[Mixture-of-Experts]]：专家路由框架。
- [[MoE-LoRA]]：M3E 替换 LLM FFN 的参数高效专家层。
- [[Macro Router]] / [[Micro Router]]：本文核心路由机制。
- [[Dynamic MoE Momentum Update]]：本文核心知识巩固机制。

### 数据集与评测

- [[R2R]]：主要 VLN benchmark。
- [[REVERIE]]：object-centric VLN benchmark。
- [[Navigation Graph]]：VLN 环境图建模。

### 相关模型

- [[NaviLLM]]：本文采用的 LLM-based navigation backbone 风格。
- [[LLM]]：导航策略核心。
- [[LoRA]]：专家适配器参数化方式。

---

## 速查卡片

> [!summary] M3E
> - **核心**：Macro Router 做全局拓扑/场景策略，Micro Router 做 token 级局部 grounding。
> - **路由**：$w=\beta w^{ma}+(1-\beta)w^{mi}$，实现全局先验与局部适应融合。
> - **持续学习**：根据专家贡献 top-K 做动态动量更新，重要专家快适应，非重要专家保旧知识。
> - **实验**：R2R AvgSR 71.92 / AvgSPL 66.96 / BWT 0.04；REVERIE SPL 48.30。
> - **定位**：replay-free continual VLN，比 rehearsal 方法更省存储且隐私友好。

---

*笔记创建时间: 2026-07-03*
