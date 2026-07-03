---
title: "Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments"
method_name: "Qwen-VLA"
authors: [Qiuyue Wang, Mingsheng Li, Jian Guan, Junyang Lin, Shuai Bai, Jingren Zhou, Qwen Team]
year: 2026
venue: arXiv
tags: [vision-language-action, flow-matching, cross-embodiment, manipulation, vision-language-navigation, diffusion-transformer, reinforcement-learning]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: local
arxiv_html: https://arxiv.org/abs/2605.30280
created: 2026-06-05
---

# 论文笔记：Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Qwen Team (阿里巴巴) |
| 日期 | May 2026 (v2: June 2026) |
| 项目主页 | https://qwen.ai/blog?id=qwenvla |
| 对比基线 | [[Pi05]]、[[GR00T-N1]]、[[OpenVLA]]、[[StreamVLN]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.30280) / [Code](https://github.com/QwenLM/Qwen-VLA) |

---

## 一句话总结

> 用一个统一的 [[Vision-Language-Action]] 模型，通过[[Embodiment-Aware Prompt Conditioning|具身感知提示条件]]把操作、导航、轨迹预测统一到共享的动作-轨迹空间，单一模型即可在多任务、多环境、多机器人形态上媲美甚至超越各自的专家模型。

---

## 核心贡献

1. **统一具身模型 (Unified VLA)**: 把机器人操作、视觉语言导航、第一视角人体动作、轨迹预测统一进同一个**动作-轨迹预测空间**，基于 [[Qwen3.5-VL|Qwen3.5-4B]] 主干 + [[Diffusion Transformer|DiT]] [[Flow Matching|流匹配]]策略头，单模型支持多平台多任务。
2. **具身感知提示条件 (Embodiment-Aware Prompt Conditioning)**: 用一段文本提示描述当前机器人平台、臂构型、控制频率、预测时域，作为唯一的平台特定接口；配合统一动作表示，同一套 DiT 参数即可处理不同控制模式/动作维度，无需 per-embodiment 输出头。
3. **渐进式四阶段训练 (Progressive Training Recipe)**: 从"压缩视角"出发设计 [[Text-to-Action Pretraining|T2A]] → [[Continued Pretraining|CPT]] → [[Supervised Fine-Tuning|SFT]] → [[Reinforcement Learning|RL]] 四阶段，弥合离散视觉语言 token 与连续动作轨迹之间的鸿沟，提升训练稳定性与下游迁移。
4. **大规模异构联合预训练 + 全面评测**: 构建覆盖机器人操作、第一视角人体、合成仿真、导航、自动驾驶 VQA、空间 grounding 的混合语料；在操作/导航/OOD/跨形态多个 benchmark 上验证联合训练不损失专长，且显著提升鲁棒性。

---

## 问题背景

### 要解决的问题
具身智能长期被"专用模型"分割：每个模型只服务单一场景或任务（如操作 vs 导航），导致能力碎片化、跨任务/环境/形态泛化差。本文探讨**能否把这些异构的具身决策问题统一进单一 VLA 模型**。

### 现有方法的局限
- 操作模型 (如 [[Pi0]]) 通常只针对桌面或灵巧操作；导航模型只围绕室内 waypoint/动作预测。
- 不同任务在观测格式、控制频率、预测时域、动作维度、评测协议上差异巨大，难以像通用视觉语言预训练那样规模化。
- 表面异构：操作要预测末端位姿/关节角/夹爪/灵巧手；导航要预测 waypoint/离散运动；第一视角人体提供腕手轨迹而非机器人控制信号。

### 本文的动机
尽管表面异构，这些任务共享**相同的计算结构**：具身智能体都要在视觉观测、语言指令、具身约束的条件下，预测物理与语义对齐的未来动作或轨迹。这一洞察支撑了统一建模——把异构数据吸收进单一 VLA，实现跨形态、跨任务、跨环境泛化。

---

## 方法详解

### 模型架构

<!-- 内联概念链接 -->

Qwen-VLA 采用 **VLM 主干 + 流匹配动作专家 (Action Expert)** 的双模块架构（见 Figure 1）：

- **输入**: 视觉上下文 $o_t$（单/多帧图像、视频或历史窗口）+ 语言指令 $x$ + 具身描述 $e$ + 可选任务标识 $z$
- **Backbone**: [[Qwen3.5-VL|Qwen3.5-4B]]，原生多模态、早期视觉语言融合（ViT 视觉 token 经空间合并后直接插入文本 token 流）；混合注意力设计——多数层用 [[Gated Linear Attention|门控线性注意力]]，间隔层用 [[Grouped-Query Attention|分组查询 softmax 注意力]]
- **核心模块**: [[Diffusion Transformer|DiT]] 风格的 [[Flow Matching|流匹配]]动作专家，用于细粒度连续动作生成
- **输出**: 统一动作-轨迹空间中的目标序列 $y_{t:t+H-1}$（操作动作 / 导航 waypoint / 人体腕手运动）
- **总参数**: 主干 4B + 动作专家约 1.15B

### 核心模块

#### 模块1: 流匹配动作专家 (Flow-Matching Action Expert)

**设计动机**: 用 [[Diffusion Transformer|DiT]] 专门承担细粒度连续动作生成，处理具身动作分布的多模态性与高频动态，同时保护主干预训练能力。

**具体实现**:
- 单流 (single-stream) DiT，将 VLM 隐状态与**含噪动作块 (noisy action chunk)** 拼接成一个序列，做联合 [[Self-Attention|自注意力]]
- 用 [[AdaLN|AdaLN]] 做 timestep 条件注入，多段 [[RoPE]] 与主干对齐
- 训练用[[Flow Matching|流匹配]]目标；推理用少量 [[Euler Integration|欧拉积分]]步从 $\tau=1$ 到 $\tau=0$ 生成动作，低延迟实时控制
- **参数构成**: 16 个 DiT block 占主体（每块 70.8M，合计 1.13B）；动作投影 MLP (4.9M)、VLM 隐状态→DiT 通道线性层 (3.9M)、timestep embedding (2.8M)、输出 AdaLN modulation (4.7M)

#### 模块2: 具身感知提示条件 (Embodiment-Aware Prompt Conditioning)

**设计动机**: 让一套模型服务多机器人形态，把平台差异完全收敛到文本接口。

**具体实现**: 每个训练样本前置一段提示，模板为：

> The robot is {robot_tag} with {single arm / dual arms}[, waist][, and mobile base]. The control frequency is {FPS} Hz. Please predict the next {chunk_size} control actions to execute the following task: {ori_instruction}.

robot_tag 与可选修饰（waist、mobile base）按形态设定，FPS 与 chunk_size 反映数据集原始控制频率与预测时域。这些提示 token 由 VLM 处理，其隐状态再拼接到含噪动作块输入 DiT。

#### 模块3: 统一动作-轨迹表示 (Unified Action and Trajectory Representation)

- 统一**张量接口与掩码方案**，但不强制所有形态进入同一物理动作语义空间；每个数据集保留原生控制约定。
- 每个样本贡献目标张量 $Y \in \mathbb{R}^{H\times K}$，$H$ 为固定预测时域，$K$ 为所有控制模式共享的固定通道维。
- **控制信号两族**: 操作信号（delta 末端位置 $(x,y,z)$、欧拉角/四元数旋转、绝对关节角、夹爪开度、灵巧手关节角）；导航轨迹信号（VLN 约定的 $(x, y, \theta)$ 每 waypoint，表示地面位移与朝向变化）。两族都被动作专家同等对待。
- **通道布局 (Channel Layout)**: 某控制模式用 $c \le K$ 个通道，放在 $Y$ 的前 $c$ 维，其余 $K-c$ 维补零；二值掩码 $M \in \{0,1\}^{H\times K}$ 记录有效通道（$M_{h,k}=1$ 当且仅当 $k<c$ 且 $h$ 落在任务的 chunk 长度 $H_{task} \le H$ 内）。无需 per-embodiment 输出头。

---

## 关键公式

### 公式0: [[Vision-Language-Action|统一条件预测框架]]

$$
p(y_{t:t+H-1} \mid o_t, x, e, z)
$$

**含义**: 所有具身任务统一为给定视觉上下文、语言指令、具身描述、任务标识，预测未来 $H$ 步目标序列的条件分布。

**符号说明**:
- $o_t$: 时刻 $t$ 的视觉上下文（可为单/多帧图像、视频或历史窗口）
- $x$: 语言任务指令；$e$: 具身文本提示；$z$: 可选任务族标识
- $y_{t:t+H-1}$: 任务相关但统一表示在动作-轨迹空间的目标序列

### 公式1: [[Flow Matching|流匹配]]逐通道损失（active channel MSE）

$$
\ell_k = \frac{\sum_{h=1}^{H} M_{h,k}\, \left\| v_\theta(Y^\tau, \tau \mid o_{1:t}, x, e, z) - (Y^1 - Y^0) \right\|_{h,k}^2}{\sum_{h=1}^{H} M_{h,k}}
$$

**含义**: 对每个有效通道 $k<c$，用掩码 $M$ 排除补零位置，计算速度场预测与目标速度 $(Y^1-Y^0)$ 的逐步均方误差（第一层平均：在时间步上按有效位置归一）。

**符号说明**:
- $v_\theta$: 动作专家预测的条件速度场
- $Y^\tau = (1-\tau)Y^0 + \tau Y^1$: 线性插值，$\tau \in [0,1]$；$Y^0$ 干净目标，$Y^1 \sim \mathcal{N}(0, I)$ 噪声
- $M_{h,k}$: 通道-时间步有效性掩码

### 公式2: [[Flow Matching|流匹配]]动作损失（两级平均）

$$
\mathcal{L}_{act} = \mathbb{E}_{\tau, Y^0, Y^1}\left[ \frac{1}{c}\sum_{k=0}^{c-1} \ell_k \right]
$$

**含义**: 第二层平均——在 $c$ 个有效通道上均匀平均，确保每个控制维度对梯度贡献相等（与形态用多少通道无关），且补零位置完全不参与梯度。推理时用少量欧拉积分从 $\tau=1$ 到 $\tau=0$ 生成动作块。

**符号说明**:
- $c$: 当前控制模式激活的通道数
- $\ell_k$: 公式1 的逐通道损失

### 公式3: [[Next-Token Prediction|视觉语言下一 token 损失]]

$$
\mathcal{L}_{vl} = -\sum_{i} \log p_\theta(w_i \mid w_{<i}, o_{1:t})
$$

**含义**: 在辅助视觉语言数据、细粒度具身动作字幕、自动驾驶 VQA、通用 VL 语料上保留标准下一 token 预测，稳定语言 grounding，防止具身联合训练下的灾难性遗忘。

**符号说明**:
- $w_i$: 文本 token；$o_{1:t}$: 视觉观测

### 公式4: 联合训练目标

$$
\mathcal{L} = \lambda_{act}\,\mathcal{L}_{act} + \lambda_{vl}\,\mathcal{L}_{vl}
$$

**含义**: 总损失是两个目标的加权组合，权重用于平衡两者的梯度幅度；每个 mini-batch 按固定采样比混合所有任务族（SFT 阶段权重设为 VL 0.1、操作与导航动作各 1.0）。

**符号说明**:
- $\lambda_{act}, \lambda_{vl}$: 平衡梯度幅度的权重系数

### 公式5: [[Quantile Normalization|分位数归一化]]

$$
\tilde{a}_d = 2 \cdot \frac{a_d - q_{01}^k}{q_{99}^k - q_{01}^k} - 1
$$

**含义**: 每个数据集 $k$ 的每个动作维 $d$，用 1% / 99% 分位数做线性映射后裁剪到 $[-1, 1]$，去除跨形态/动作空间的尺度差异，同时保留各源内部相对运动结构。

**符号说明**:
- $q_{01}^k, q_{99}^k$: 数据集 $k$ 中该维的 1st / 99th 百分位
- $\tilde{a}_d$: 归一化后动作值

### 公式6: [[PPO|PPO 剪裁代理目标]]（RL 阶段）

$$
\mathcal{L}_{actor}(\theta) = -\,\mathbb{E}_t\left[ \min\left( r_t(\theta)\,\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t \right) \right]
$$

**含义**: 以闭环任务成功为优化目标的策略代理损失，$r_t$ 为重要性比，$\hat{A}_t$ 由 [[GAE|广义优势估计]]给出。

**符号说明**:
- $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{old}}(a_t \mid s_t)$: 重要性比；$s_t=(o_t, x, e)$ 当前状态，$a_t$ 时域 $H$ 的预测动作块
- $\hat{A}_t$: [[GAE]] 优势估计，折扣 $\gamma=0.99$、迹衰减 $\lambda=0.95$
- $\epsilon=0.2$: 对称剪裁阈值

### 公式7: RL 总损失（actor + value）

$$
\mathcal{L}(\theta) = \mathcal{L}_{actor}(\theta) + c_v\,\mathcal{L}_{value}(\theta)
$$

**含义**: 策略代理损失与价值函数回归项组合，$c_v=1$；每个 rollout batch 做 4 个优化 epoch。价值头直接挂在 VLM 主干上（mean-pool 隐状态→线性投影），对进入价值头的隐状态做 stop-gradient，价值头用单独学习率（$10^{-4}$，约为 actor $5\times10^{-6}$ 的 20×）。

**符号说明**:
- $\mathcal{L}_{value}$: 价值预测的剪裁 MSE 损失；$c_v$: 价值损失权重

> **流匹配下的 log 概率估计**：PPO 需要 $\log\pi_\theta(a_t\mid s_t)$，但流匹配通过学习的速度场定义隐式密度。作者把确定性 probability-flow ODE 转为对应 SDE（每个欧拉去噪步注入受控噪声），使每步转移成为显式高斯，其 log 概率可解析计算。默认每个 rollout 随机选一个去噪步估计 log 概率，仅需一次额外 DiT 前向。log 概率与优势都在 action-chunk 级（每 $H=16$ 步一个标量奖励与优势）。

---

## 关键图表

<!-- 图片：整页高分辨率渲染，本地 assets/，wikilink 嵌入；Figure 1 另附 GitHub 高清外链 -->

### Figure 1: Overview / 系统概览

![[Qwen-VLA_fig1_overview.png]]

GitHub 高清版（在线外链）：
![Qwen-VLA Overview](https://raw.githubusercontent.com/QwenLM/Qwen-VLA/main/assets/qwenvla_overview.png)

**说明**: Qwen-VLA 整体架构。底部 [[Qwen3.5-VL]] 主干处理观测图像 + 提示，吸收 VLA / VLN / VL 三类数据；其隐状态经 MLP 注入上方的 [[Diffusion Transformer|DiT]]（$N\times$ 块，每块含 Self-Attention + Feed-Forward MLP + [[AdaLN]]，由 timestep 条件化），从含噪动作经[[Flow Matching|流匹配]]去噪生成 Clean Action，可输出导航与操作动作及文本响应。

### Figure 2: Training Recipe / 四阶段训练流程

![[Qwen-VLA_fig2_training_recipe.png]]

**说明**: 四阶段渐进训练。**(I) [[Text-to-Action Pretraining|T2A]]**：冻结 VLM，只训 DiT，仅用文本+具身提示（**刻意不给图像**），让解码器成为"语言→动作解压器"，建立结构化动作先验。**(II) [[Continued Pretraining|CPT]]**：解冻两个模块，把先验 grounding 到视觉观测。**(III) [[Supervised Fine-Tuning|SFT]]**：分两路并行（多任务 SFT / 真机 ALOHA SFT）。**(IV) [[Reinforcement Learning|RL]]**：仅在 SimplerEnv 用稀疏二值成功奖励优化闭环成功率，产出最终 Qwen-VLA-Instruct。

### Figure 3: ROBOINF 合成数据示例

![[Qwen-VLA_fig3_roboinf_data.png]]

**说明**: 用 [[ROBOINF]] 仿真管线生成的数据。上行短时域任务"Place the two green staplers side by side"（reaching→grasping→transporting→placing）；下行长时域任务"Group the drinks together and leave the cleaning sponge by itself"，可分解为多个子任务段（pick/place 各饮料）。**子任务分割 (Subtask Segmentation)** 提供多时间粒度监督。

### Figure 4: 真实世界 ALOHA 评测任务总览

![[Qwen-VLA_fig4_aloha_tasks.png]]

**说明**: ALOHA 双臂平台上的真实评测任务。6 类 in-domain 任务（Pick and Place / Table Cleaning / Bowl Stacking / Bowl and Object Placing / Towel Folding / Fine-grained）+ 5 类 OOD 泛化（颜色、实例、位置、背景、指令），含未见蓝笔、红块、夜间 RGB 光照、绿色背景等变体。

### Figure 5: 真实世界 OOD 定性结果（Qwen-VLA-Base）

![[Qwen-VLA_fig5_realworld_ood.png]]

**说明**: Qwen-VLA-Base 在 ALOHA 双臂上的四类 OOD 定性 rollout。左上：颜色条件抓取（绿/蓝/红/黄球）；右上：抓取新物体（西兰花、玩具鸭）+ 组合式"清理桌面"序列；左下：与完全未见物体（墨镜、毛绒娃娃、玩具鸭）交互；右下：拧笔帽并放置原子动作迁移，对未见黄色背景鲁棒。验证混合 VL–VLA–VLN 持续预训练带来的零样本视觉识别与指令迁移。

### Figure 6: T2A 预训练消融（三子图）

![[Qwen-VLA_fig6_t2a_ablation.png]]

**说明**: T2A 设计消融，指标为从 T2A checkpoint SFT 后在 Simpler-WidowX 的成功率。
- **(a) 数据构成与预测模式**：全序列预测 + 20% 合成 + 80% 真机达最佳 **71.1%**（比无 T2A 的 60.9% 高 +10.2pp）；纯真机 51.0%，纯合成 64.1%；chunk 预测一致弱于全序列；T2A 阶段加入图像反而掉点（-2.9pp），证实 T2A 应完全屏蔽视觉。
- **(b) 流匹配 timestep 分布**：T2A 用 **Sigmoid-Normal** + SFT 用 **Beta** 最优（71.1%）；T2A 换 Beta 降到 65.4%，SFT 换 Sigmoid-Normal 降到 62.8%，双 Beta 仅 59.4%。
- **(c) T2A 训练时长**：2,000 步峰值（71.1%），4k/10k 持平，40k 因过拟合掉到 60.4%。最终采用 2,000 步。

### Figure 7: 视觉语言协同训练消融（两子图）

![[Qwen-VLA_fig7_vl_cotrain.png]]

**说明**:
- **(a) VL 数据对动作学习的影响**：在 Libero / Simpler-WidowX 上 VLA-Only 与 VL+VLA 几乎持平（无干扰）；在需要细粒度物体识别与组合指令解析的 benchmark 上，VL+VLA 明显增益——RoboCasa-GR1 +4.9pp (51.1%→56.0%)、RoboTwin-2.0 +4.6pp (81.8%→86.4%)。
- **(b) 预训练 DiT 的可迁移性**：把预训练 DiT 接到全新 Qwen3.5-4B 主干，相比 from-scratch DiT 收敛更快、峰值更高。

### Table 1: 预训练数据混合构成

| 数据源 | 占比 (%) |
|--------|---------|
| Robot Manipulation Trajectories | 74.2 |
| Human Egocentric Trajectories | 6.0 |
| Navigation Trajectories | 7.5 |
| Synthetic Simulation Trajectories (ours) | 3.7 |
| General Vision-Language Data | 3.4 |
| Spatial Grounding (2D) | 2.5 |
| Autonomous Driving VQA | 2.4 |
| Fine-Grained Embodied Action Caption | 0.2 |
| **Total** | **100.0** |

**说明**: 机器人操作轨迹占绝对主体（74.2%，含 >10,000 小时公开真机数据 + >1,000 小时自采 + >8M 合成）。

### Table 2: 预训练语料中的代表性机器人形态

| Robot | Arms | Action type |
|-------|------|-------------|
| WidowX | Single | EEF + G |
| Google Robot | Single | EEF + G |
| Franka Panda | Single / Dual | EEF + G; Abs Joint + G |
| ARX5 | Dual | EEF + G |
| Fourier GR-1 | Dual | EEF + G |
| Mobile ALOHA | Dual | EEF + G; Abs Joint + G |
| AgiBot A2-D | Dual | Abs Joint + G; Abs Joint + DH |
| Galaxea R1 | Dual | Abs Joint + G |
| AIRBOT MMK2 | Dual | Abs Joint + DH |
| TienKung | Dual | Abs Joint + G; Abs Joint + DH |
| Real Human | Dual | EEF (from MANO) |

**说明**: EEF=末端位姿，Joint=关节角，Δ=delta 相对指令，Abs=绝对指令，G=夹爪，DH=灵巧手。涵盖单臂/双臂/人形/真人多形态。

### Table 3: 粗标签 vs 细粒度动作字幕

| 类型 | 内容 |
|------|------|
| Coarse label | "Pick up, rotate, and place the ceramic bowl." |
| Fine-grained | Step 1: Pick up the ceramic bowl from the right far edge. Step 2: Rotate the bowl clockwise for two full circles. Step 3: Place the bowl at the center of the table. |

**说明**: 沿 13 个维度（动作原语、执行者、物体识别消歧、接触区域、源/目标位置、轨迹朝向、夹爪状态、身体运动等）做密集标注，两阶段管线（Qwen3.6-plus 粗→密集采样细），约 48,000 视频-字幕对（占 0.2%）。

### Table 4: 仿真操作主结果（专家 vs 单一通才）

| Method | Type | LIBERO | RoboCasa-GR1 | Simpler-WidowX | RoboTwin-Easy | RoboTwin-Hard |
|--------|------|:---:|:---:|:---:|:---:|:---:|
| π0 | Specialist | 94.4 | — | — | — | 58.4 |
| StarVLA-OFT | Specialist | 96.6 | 48.8 | 65.9 | — | — |
| GR00T N1.6 | Specialist | 97.2 | 49.9 | — | — | — |
| π0.5 | Specialist | 97.6 | 37.0 | 64.6 | 50.4 | — |
| ABot-M0 | Specialist | 98.6 | 58.3 | — | 86.0 | 85.0 |
| Being-H0.5 | Specialist | 97.6 | 53.3 | 63.2 | 47.6 | 76.8 |
| Qwen-VLA-Base | Generalist | 90.8 | 40.4 | 64.3 | 64.3 | 66.4 |
| **Qwen-VLA-Instruct** | **Generalist** | **97.9** | 56.7 | **73.7** | **86.1** | **87.2** |

**说明**: 单一通才超越多数专家——LIBERO 97.9% 与最佳专家持平；RoboCasa-GR1 56.7% 超 π0.5/GR00T/Being-H0.5；Simpler-WidowX 73.7% 超 StarVLA-OFT；RoboTwin-Easy/Hard 86.1/87.2% 超前最佳专家 ABot-M0。指令微调相对 Base 增益显著（LIBERO +7.1、RoboCasa +16.3、Simpler +9.4、RoboTwin-E +21.8、RoboTwin-H +20.8）。
（注：原文 Table 4 排版交错，数值已按正文论述对齐。）

### Table 5: 真实世界 ALOHA in-domain 成功率 (%)

| Model | Pick & Place | Table Cleaning | Bowl Stacking | Bowl Pick & Place | Towel Folding | Fine-grained | **Avg** |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GR00T N1.6 | 30.8 | 38.5 | 53.8 | 19.2 | 19.2 | 10.3 | 28.6 |
| π0.5 | 73.1 | 84.6 | 88.5 | 69.2 | **80.8** | 33.3 | 71.6 |
| Qwen-VLA-aloha (w/o pretrain) | 30.8 | 53.8 | 61.5 | 64.1 | 50.0 | 30.8 | 48.5 |
| **Qwen-VLA-aloha (w/ pretrain)** | **96.2** | **92.3** | **98.7** | **87.2** | 65.4 | **61.5** | **83.6** |

**说明**: 大规模预训练把均值从 48.5%（从零训）拉到 83.6%，超 GR00T N1.6 与 π0.5。证明性能增益主要来自预训练的 Qwen-VLA-Base 而非架构本身。

### Table 6: 真实世界 ALOHA OOD 成功率 (%)

| Model | Color | Instance | Position | Background | Instruction | **Avg** |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| GR00T N1.6 | 46.2 | 38.5 | 3.8 | 19.2 | 19.2 | 25.4 |
| π0.5 | 57.7 | 61.5 | 19.2 | 26.9 | 42.3 | 41.5 |
| Qwen-VLA-aloha (w/o pretrain) | 42.3 | 30.8 | 34.6 | 30.8 | 42.3 | 36.2 |
| **Qwen-VLA-aloha (w/ pretrain)** | **88.5** | **76.9** | **53.8** | **80.8** | **84.6** | **76.9** |

**说明**: 全部五类泛化均最佳，均值 76.9%，超 π0.5 达 35.4pp、超从零训 40.7pp；背景与指令泛化提升尤为显著。

### Table 7: VLN-CE 导航结果（R2R / RxR Val-Unseen）

| Method | R2R NE↓ | R2R OS | R2R SR | R2R SPL | RxR NE↓ | RxR SR | RxR SPL | RxR nDTW |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| NaVid | 5.7 | 49.2 | 41.9 | 36.5 | 5.7 | 45.7 | 38.2 | — |
| Uni-NaVid | 5.6 | 53.3 | 47.0 | 42.7 | 6.2 | 48.7 | 40.9 | — |
| NaVILA | 5.2 | 62.5 | 54.0 | 49.0 | 6.8 | 49.3 | 44.0 | 58.8 |
| StreamVLN | 5.0 | 64.2 | 56.9 | **51.9** | 6.2 | 52.9 | 46.0 | **61.9** |
| Qwen-VLA-Base | 5.2 | 61.7 | 53.8 | 49.4 | 6.4 | 55.1 | 45.8 | 56.2 |
| **Qwen-VLA-Instruct** | **5.1** | **69.0** | **57.5** | 51.2 | **5.8** | **59.6** | **47.8** | 57.1 |

**说明**: R2R Val-Unseen 上 OS 69.0、SR 57.5 均最高（超 StreamVLN +4.8 / +0.6）；RxR 上 SR 59.6、SPL 47.8 领先明显。联合 VLA+VLN 训练在两个 benchmark 都保持合理表现。

### Table 8: SimplerEnv-OOD 静态操作泛化 (%)

| Method | MoveAway | MoveRight | PlaceNear | PlaceRight | PutFront | StackYellow | Avg |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| π0.5 | 26.1 | 0.0 | 0.0 | 32.1 | 13.0 | 4.2 | 12.6 |
| Qwen-VLA-Base | 31.3 | 31.6 | 16.7 | 47.1 | 6.3 | 18.8 | 25.3 |
| **Qwen-VLA-Instruct** | **43.8** | **33.3** | **39.6** | **47.9** | 4.2 | **22.9** | **32.0** |

**说明**: 仅用 Bridge 简单 pick-and-place 微调，评测未见的空间关系/操作原语。Instruct 均值 32.0% 远超 π0.5 (12.6%)；π0.5 在 MoveRight/PlaceNear 完全失败而本文 33.3%/39.6%。
（注：原文 Table 8 数值排版交错，已按正文论述对齐。）

### Table 9: DOMINO 动态操作（零样本）

| Method | 类别 | SR (%) | MS |
|--------|------|:---:|:---:|
| OpenVLA | Fine-tuned on dynamic | 1.5 | 6.1 |
| RDT-1B | Fine-tuned | 5.3 | 17.7 |
| π0 | Fine-tuned | 8.2 | 24.0 |
| π0.5 | Fine-tuned | 9.6 | 26.2 |
| InternVLA-M1 | Fine-tuned | 5.4 | 27.6 |
| VLA-Adapter | Fine-tuned | 4.4 | 24.3 |
| π0-FAST | Fine-tuned | 3.5 | 20.9 |
| OpenVLA-OFT | Fine-tuned | 9.1 | 24.1 |
| StarVLA-OFT | Fine-tuned | 10.9 | 30.5 |
| PUMA | Fine-tuned | 17.2 | 35.0 |
| OpenVLA-OFT | Zero-shot | 6.7 | 20.0 |
| π0.5 | Zero-shot | 7.5 | 20.4 |
| LingBot-VLA w/ depth | Zero-shot | 11.8 | 26.7 |
| LingBot-VA | Zero-shot | 24.1 | 36.1 |
| Qwen-VLA-Base | Zero-shot | 21.1 | 37.4 |
| **Qwen-VLA-Instruct** | **Zero-shot** | **26.6** | **39.5** |

**说明**: 零样本即取得最佳 SR 26.6% / MS 39.5，零样本类中超 OpenVLA-OFT 与 π0.5 达 19+pp；甚至超越全部针对动态操作专门微调的基线（含依赖 DOMINO 专用微调+时序输入的 PUMA，SR +9.4 / MS +4.5），且仅用当前帧观测、无任何动态操作微调。
（注：原文 Table 9 行列排版交错，类别与数值已按正文论述重排。）

### Table 10: 异构形态投影设计消融（任务均值成功率 %）

| | Bridge Only | Robocasa Only | Multi-MLP | Concat. | Zero-Pad |
|---|:---:|:---:|:---:|:---:|:---:|
| Bridge | 62.8 | — | 63.3 | 63.0 | 63.0 |
| Robocasa | — | 53.4 | 52.1 | 52.8 | 53.2 |

**说明**: 三种投影（Multi-MLP / Concatenation / Zero-Padding）与单形态基线相当（多形态协同无显著惩罚），彼此差异 <1.2pp。Zero-Padding 参数最省（$2h\,d_{max}$ vs 其余 $2h\sum_i d_i$），故作为默认设计。

### Table 11: 后训练各阶段累积效果 (%)

| Stage | Simpler | RoboCasa | RoboTwin-E | RoboTwin-H | LIBERO | SimplerOOD | DOMINO SR | DOMINO MS |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CPT (Base) | 64.3 | 40.4 | 64.3 | 66.4 | 90.8 | 25.3 | 21.1 | 37.4 |
| + SFT | 70.8 | 56.0 | 86.3 | 87.1 | 97.8 | 31.6 | 25.7 | 39.1 |
| + RL (Instruct) | 73.7 | 56.7 | 86.1 | 87.2 | 97.9 | 32.0 | 26.6 | 39.5 |

**说明**: SFT 跨 benchmark 大幅提升；RL 在其训练环境 SimplerEnv 增益最大（+2.9pp）。关键是 RL 增益不局限于训练环境——held-out 任务保持或轻微提升（无灾难性遗忘），DOMINO 动态零样本 SR/MS 也上升，说明仿真中的成功率优化带来温和正迁移。

### Table 12: 状态条件消融（RoboTwin-2.0 成功率 %）

| Conditioning | RoboTwin-Easy | RoboTwin-Hard |
|--------------|:---:|:---:|
| No State | 88.7 | 87.4 |
| State in VLM Prompt | 89.3 | 88.7 |
| State in DiT | 89.4 | 88.3 |

**关键发现**: 引入本体感知状态增益极小（Easy 最多 +0.7、Hard 最多 +1.3），两种注入策略相当。原因：多视角视觉已提供足够构型信息；流匹配预测相对位移而非绝对位姿，降低对显式当前状态的需求。故默认**不**加状态条件，保持具身文本提示为唯一平台特定输入。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 公开真机（RoboSet/AgiBot/DROID/Bridge V2/RT-1/RH20T 等） | >10,000 h | 多形态多任务 | 预训练 |
| 自采真机 | >1,000 h | in-house | 预训练 (~20%) |
| 合成仿真 (ROBOINF + 语言-动作) | >8M 轨迹 / 14,000+ h | 域随机化、子任务分割 | 预训练 / T2A |
| 第一视角人体 (Ego4D/EPIC-KITCHENS via VITRA, EgoDex, EgoVerse, Xperience) | 6.0% | MANO 腕手轨迹 + eigengrasp | 预训练 |
| 导航 (R2R/RxR via VLN-CE) | 7.5% | 长时域 + 指令跟随/搜物/追踪 | 预训练/SFT |
| 自动驾驶 VQA + 空间 grounding + 通用 VL | 8.3% | 防遗忘、空间推理 | 预训练 |

**评测 benchmark**: LIBERO、Simpler-WidowX、RoboCasa-GR1、RoboTwin 2.0（操作）；R2R/RxR Val-Unseen（导航）；SimplerEnv-OOD（静态 OOD）；DOMINO（动态零样本）；真机 ALOHA。

### 实现细节

- **Backbone**: [[Qwen3.5-VL|Qwen3.5-4B]]；**动作专家**: 16-block DiT，约 1.15B 参数
- **第一视角人体动作表示**: 每手腕 SE(3) 相对变换（平移向量 + 轴角，6 维）+ 对 45 维轴角关节做 [[PCA]] 取前 10 个主成分系数（[[Eigengrasp|eigengrasps]]），每手 16 维，双手共 32 维
- **合成语言-动作数据**: 6 个任务模板 × 6 单臂机器人（Franka Panda、UR10e、UR5e、Kinova Gen3、TM12、xArm7），约 7.2M 轨迹，记录关节位置/速度/末端位姿/夹爪 @ 50 Hz
- **多相机**: 用 `<|tag_start|> image <|tag_end|>` 边界 token 标记视角来源（ego / cam_left_wrist / cam_right_wrist）
- **RL**: [[PPO]]+[[GAE]]，[[RLinf]] 框架；N=128 并行环境，每迭代 8 rollout epoch × 128 步 → 8,192 transition chunk；rollout 温度 1.0，评测 0.6；动作块 $H=16$；稀疏二值奖励 R∈{0,1}
- **学习率**: 主干与动作解码器分组 cosine 衰减；actor $5\times10^{-6}$，value head $10^{-4}$
- **流匹配 timestep 分布**: T2A 用 Sigmoid-Normal，CPT/SFT 用 Beta

### 可视化结果

见 Figure 5——Qwen-VLA-Base 在真机上零样本识别未见物体、跟随未见动作动词（"approach"）、对未见黄色背景鲁棒；玩具鸭/墨镜因几何形状而非识别失败导致抓取不稳，体现的是形状泛化的物理边界。

---

## 批判性思考

### 优点
1. **统一性强**：真正把操作/导航/第一视角/轨迹预测纳入单一动作-轨迹空间，且实证"通才不输专家"，工程与科学价值都高。
2. **训练配方有洞察**："压缩视角"下的 T2A 把语言→动作解压先验与视觉 grounding 解耦，配套消融（数据配比、全序列 vs chunk、timestep 分布、训练步数）做得扎实。
3. **OOD 泛化突出**：DOMINO 零样本动态操作超过专门微调基线，SimplerEnv-OOD 与真机 OOD 均大幅领先，说明大规模异构预训练 + 决断性流匹配动作生成的协同。
4. **接口简洁**：具身文本提示是唯一平台特定接口，部署时只换提示即零样本迁移，省去 per-embodiment 头与适配。

### 局限性
1. **具身数据规模仍受限**：远小于视觉语言预训练数据，对长尾物体/环境/形态/富接触交互的鲁棒性有限（作者自陈）。
2. **联合训练有优化权衡**：动作训练可能轻微回退纯 VL 与导航评测，需更好的目标平衡/数据课程/模块化。
3. **评测偏短时域、benchmark 驱动**：缺长时长、易失败的真实部署验证。
4. **RL 范围窄**：仅在单一 SimplerEnv 收集 rollout，虽展示正迁移，但 RL 对更广任务族的可扩展性未验证。

### 潜在改进方向
1. 引入 episodic memory / 持久状态支持长时域规划与失败恢复（作者在框架可扩展性中已点出）。
2. 在输出侧联合预测未来视觉状态，把动作生成与 [[World Model|世界模型]]统一。
3. 融合力/触觉/本体感知等富物理反馈 + 大规模 sim-and-real RL。

### 可复现性评估
- [x] 代码开源（GitHub 仓库，目前主要为信息/Issue，模型权重待确认）
- [ ] 预训练模型（README 未明确放出权重）
- [x] 训练细节完整（四阶段、损失、超参、RL 配置详尽）
- [ ] 数据集可获取（公开数据集可得，但自采 1,000h + 8M 合成数据未必开放）

---

## 关联笔记

### 基于
- [[Qwen3.5-VL]]: 视觉语言主干
- [[Flow Matching]]: 动作专家训练目标
- [[Diffusion Transformer]]: 动作专家骨架
- [[Pi0]] / [[Diffusion Policy]]: 流匹配/扩散策略的前置工作

### 对比
- [[Pi05]]: 主要操作专家基线，多 benchmark 与 OOD 对比
- [[GR00T-N1]]: 人形操作专家基线
- [[OpenVLA]]: 经典 VLA 基线（DOMINO/OFT 变体）
- [[StreamVLN]]: 导航 SOTA 基线

### 方法相关
- [[Embodiment-Aware Prompt Conditioning]]: 跨形态统一核心
- [[Text-to-Action Pretraining]]: 阶段 I 关键创新
- [[PPO]] / [[GAE]]: RL 后训练算法
- [[Eigengrasp]] / [[MANO]]: 人体手部动作表示
- [[Action Chunking]]: 动作块预测

### 硬件/数据相关
- [[ALOHA]]: 真机评测双臂平台
- [[ROBOINF]]: 合成数据生成管线
- [[VLN-CE]]: 连续环境导航评测
- [[RLinf]]: RL 训练框架

---

## 速查卡片

> [!summary] Qwen-VLA
> - **核心**: 单一 VLA 用具身文本提示统一操作/导航/轨迹于共享动作-轨迹空间
> - **方法**: Qwen3.5-4B 主干 + 1.15B DiT 流匹配动作专家 + T2A→CPT→SFT→RL 四阶段
> - **结果**: LIBERO 97.9 / Simpler-WidowX 73.7 / RoboTwin-E/H 86.1/87.2 / R2R OS 69.0 / RxR SR 59.6 / 真机 OOD 76.9 / DOMINO 零样本 26.6
> - **代码**: https://github.com/QwenLM/Qwen-VLA

---

*笔记创建时间: 2026-06-05*
