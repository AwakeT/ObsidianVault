---
title: "ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning"
method_name: "ABot-M0"
authors: [Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, Feng Xiong, Xing Wei, Zhiheng Ma, Mu Xu]
year: 2026
venue: arXiv
tags: [vision-language-action, robotic-manipulation, action-manifold-learning, embodied-ai, cross-embodiment, 3d-perception]
zotero_collection: _待整理
image_source: local
arxiv: https://arxiv.org/abs/2602.11236
arxiv_html: https://arxiv.org/html/2602.11236v2
created: 2026-07-02
---

# 论文笔记：ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[ABot-M0]] |
| 机构 | AMAP CV Lab, Alibaba Group |
| 日期 | February 11, 2026; arXiv v2: April 14, 2026 |
| 会议/来源 | arXiv |
| 项目主页 | [ABot-Manipulation](https://amap-cvlab.github.io/ABot-Manipulation) |
| 代码 | [GitHub](https://github.com/amap-cvlab/ABot-Manipulation) |
| 链接 | [arXiv](https://arxiv.org/abs/2602.11236) / [PDF](https://arxiv.org/pdf/2602.11236) |
| 相关工作 | [[Vision-Language-Action]]、[[OpenVLA]]、[[Diffusion Policy]]、[[Qwen3-VL]]、[[VGGT]] |

---

## 一句话总结

> ABot-M0 用开放机器人数据构建 [[UniACT-dataset]]，并提出 [[Action Manifold Learning]] 让 VLA 直接预测低维动作流形上的动作序列，从数据标准化、动作专家、3D 感知注入三条线一起提升跨机器人泛化。

---

## 核心贡献

1. **开放数据工程贡献**：整合 OXE、OXE-AugE、AgiBot-Beta、RoboCOIN、RoboMind、Galaxea 六个公开数据源，清洗出 600 万+轨迹、9500+ 小时、20+ embodiment 的 [[UniACT-dataset]]。
2. **动作表示统一**：把多源数据统一成 [[Delta Action]] + [[End-Effector Frame]] 表示，旋转统一成 [[Rotation Vector]]，并用 [[Pad-to-Dual-Arm]] 让单臂和双臂任务共享 14 维输出空间。
3. **Action Manifold Learning**：提出 [[Action Manifold Hypothesis]]，认为有效机器人动作位于低维平滑流形；模型直接预测 clean action chunk，而不是预测高维噪声或速度。
4. **VLM + DiT 双组件架构**：用 [[Qwen3-VL]] 作为语义/视觉 backbone，用 0.16B [[Diffusion Transformer]] 动作专家生成动作序列，默认 4 个 denoising steps、action chunk size 16。
5. **可插拔 3D 感知注入**：通过 [[VGGT]] 和 [[Qwen-Image-Edit]] 引入单视图/多视图几何先验，用 cross-attention 把 3D feature 融入动作专家。
6. **实验性能**：在 LIBERO、LIBERO-Plus、RoboCasa GR1、RoboTwin2.0 上分别达到 98.6%、80.5%、58.3%、约 85%+，显著优于 OpenVLA-OFT、π0.5、X-VLA、GR00T 等强基线。

---

## 问题背景

### 要解决的问题

机器人领域长期追求 “one brain, many forms”：同一个 embodied agent 能跨不同机械臂、双臂、半人形/人形平台执行任务。但现有 [[Vision-Language-Action]] 方法通常被三个瓶颈卡住：

- **数据规模不足**：机器人动作标签需要真实交互或高质量仿真，采集成本远高于图文数据。
- **数据标准不统一**：不同数据集的动作空间、坐标系、频率、相机、任务粒度都不同，直接混训会把大量模型容量浪费在记忆格式差异上。
- **VLM 到动作的目标错配**：标准 [[VLM]] 擅长语义识别，却不天然理解 3D 几何、物理接触和低层控制。

### 现有方法的局限

- OpenVLA / π0 系列证明了大规模 VLA 的潜力，但公开数据的跨 embodiment 统一仍不充分。
- diffusion / flow 动作专家常做 noise-prediction 或 velocity-prediction，高维动作空间变大后学习压力很高。
- 许多 VLA 仍主要依赖 2D 图像特征，面对视角扰动、精细插入、双臂协作时空间误差会累积。

### 本文动机

ABot-M0 的核心判断是：**开放数据并非不能训练强 VLA，关键是要把数据治理、动作表示、训练目标和 3D 感知一起工程化**。因此作者没有只提出一个网络模块，而是给出从数据清洗到统一预训练再到 SFT 注入空间先验的一整套 pipeline。

---

## 方法详解

### 总体架构

ABot-M0 采用两组件结构：

- **Perception engine**：[[Qwen3-VL]] 作为预训练 VLM，输入多视角图像序列和自然语言指令，输出最终层 hidden features。
- **Action expert**：0.16B [[Diffusion Transformer]]，接收 VLM feature、可选 3D feature、机器人状态和 noised action，输出 action chunk。
- **训练阶段**：Stage 1 在 UniACT 上大规模预训练，Stage 2 用 SFT 注入 3D/空间任务先验。
- **输出动作**：双臂统一输出 14 维，每只手臂为 $[\Delta x,\Delta y,\Delta z,r,\text{gripper}] \in \mathbb{R}^7$。

```text
多源公开数据
  -> 清洗 / 标准格式 / 均衡采样
  -> UniACT-dataset
  -> Qwen3-VL 语义视觉特征
  -> 可选 3D Modules: VGGT / Qwen-Image-Edit
  -> DiT action expert + Action Manifold Learning
  -> dual-arm action chunk
```

### 数据管线：UniACT-dataset

UniACT-dataset 来自六个主要数据源：

- **OXE / OXE-AugE**：规模大、单臂为主、场景覆盖广，但质量不均、短轨迹多。
- **AgiBot-Beta / Galaxea**：高质量长时序双臂数据，但 embodiment 单一。
- **RoboCOIN / RoboMind**：包含单臂和双臂、多机器人、多层级标注，但规模较小或格式不标准。

作者先做 dataset-specific curation，再统一格式：

- 清理无效指令、乱码、多语言混乱、frame-instruction misalignment。
- 丢弃黑帧、严重模糊、遮挡、视角不足、异常动作序列。
- 把 LeRobot v2、RLDS、AgiBot 等格式转换到统一格式。
- 将动作统一到 [[End-Effector Frame]] 下的 [[Delta Action]]。
- 单臂样本右臂化，并对未使用手臂零填充到双臂动作空间。

### Action Manifold Learning

本文把 manifold hypothesis 扩展到机器人动作：有效动作序列不是高维空间中的任意点，而是受物理、任务、机器人结构约束的低维平滑流形。

传统 diffusion / flow-based action generator 学习的是：

- $\epsilon$-prediction：预测噪声。
- $v$-prediction：预测 flow velocity。

ABot-M0 改为：

- $a$-prediction：模型直接预测 clean action chunk $\hat{A}_t$。
- 训练时仍在 velocity space 计算 loss，使其保留 [[Flow Matching]] 在不同噪声水平调节学习信号的优点。

直觉上，噪声或速度目标往往 off-manifold；直接预测 action 能让动作专家把容量用在“可执行动作的结构”上，尤其适合更长 chunk、更高维双臂/全身控制。

### 两阶段训练

**Stage 1: Large-scale pre-training**

- 使用 UniACT-dataset 学习跨任务、跨 embodiment 的通用动作先验。
- 用 fast token head 和 action classification loss 帮助连续动作离散化建模，提高收敛稳定性。
- 对任务类别和机器人形态做 dual-weighted sampling，减少长尾任务和单一 embodiment 的偏置。

**Stage 2: Space-aware SFT**

- 针对插入、折叠、双臂协作等精细空间任务做监督微调。
- 联合微调 VLM 和 action expert，小学习率、dropout、action noise perturbation 提升鲁棒性。
- 可插拔接入 3D feature，不修改核心 backbone。

### VLM Feature Interaction

作者比较 VLM 最后一层、中间层、多层聚合，以及 action query 的不同使用方式。结论很实用：

- robotics pre-training 后，**final hidden layer feature 最好**。
- action query 单独使用略逊于原始 feature。
- feature + query concat 反而最差，可能破坏预训练 VLM 已形成的动作语义结构。

### 3D Information Injection

3D 模块不替换 VLM，而是以专家形式补偿 VLM 的几何短板：

- **单视图 3D**：用 [[VGGT]] 从 RGB 中提取 scene-level geometric feature。
- **隐式多视图**：用 [[Qwen-Image-Edit]] 合成额外视角，缓解固定视角依赖。
- **融合策略**：比较 concatenation、cross-attention、Q-Former；最终 single-layer cross-attention 最好。

---

## 关键公式

### 公式1：[[Action Manifold Learning|Action chunk 直接预测]]

$$
\hat{A}_t = V_\theta(\phi_t, A_t^\tau, q_t)
$$

**含义**：动作专家 $V_\theta$ 不预测噪声，而是直接预测 clean action chunk $\hat{A}_t$。

**符号说明**：

- $\hat{A}_t=[\hat{a}_t,\hat{a}_{t+1},\ldots,\hat{a}_{t+H-1}]$：长度为 $H$ 的预测动作块。
- $\phi_t$：来自 VLM 和可选 3D modules 的上下文特征。
- $q_t$：当前机器人状态。
- $A_t^\tau$：扩散时间步 $\tau$ 下的 noisy action。
- $V_\theta$：基于 DiT 的动作生成器。

### 公式2：[[Flow Matching|Noisy action 构造]]

$$
A_t^\tau = \tau A_t + (1-\tau)\epsilon,\qquad \tau\in[0,1],\quad \epsilon\sim\mathcal{N}(0,I)
$$

**含义**：在 ground-truth action chunk $A_t$ 和标准高斯噪声 $\epsilon$ 之间线性插值得到 noisy action。

### 公式3：[[Flow Matching|速度估计]]

$$
\hat{v} = \frac{\hat{A}_t - A_t^\tau}{1-\tau},
\qquad
v = \frac{A_t - A_t^\tau}{1-\tau}
$$

**含义**：虽然模型输出 action，但训练 loss 可转到 velocity space，从而对不同噪声水平做 reweighting。

**符号说明**：

- $\hat{v}$：由预测动作反推的估计速度。
- $v$：由真实动作反推的目标速度。
- $1-\tau$：从 noisy action 投影回 clean action 的距离尺度。

### 公式4：[[Action Manifold Learning|重加权动作损失]]

$$
\begin{aligned}
\mathcal{L}(\theta)
&= \mathbb{E}\left\|v_{\text{pred}}-v_{\text{target}}\right\|^2 \\
&= \mathbb{E}\left[
w(\tau)\left\|V_\theta(\phi_t,A_t^\tau,q_t)-A_t\right\|^2
\right],
\qquad
w(\tau)=\frac{1}{(1-\tau)^2}
\end{aligned}
$$

**含义**：低噪声时 $\tau\to1$，权重变大，促使模型做精细动作修正；高噪声时权重较小，允许较大幅度 denoising。

### 公式5：[[Euler Integration|推理更新]]

$$
A_t^{\tau+\Delta\tau}=A_t^\tau+\Delta\tau\cdot \hat{v}
$$

**含义**：推理从 $A_t^0\sim\mathcal{N}(0,I)$ 开始，迭代预测 clean action 并沿 ODE 轨迹更新 action。

---

## 关键图表

> 图表来源：论文 PDF v2。本笔记已从 PDF 裁剪本地图片，共 Figure 10 张、Table 10 张。

### Figure 1: Data cleaning and preprocessing pipeline

![[ABot-M0_fig1_data_cleaning.png]]

**说明**：展示从多源数据到 UniACT-dataset 的清洗流程：源格式转换、无效样本过滤、动作表示统一、旋转表示转换等。约 16% 轨迹被丢弃，其余轨迹 refined 后合并。

### Figure 2: Overview of UniACT-dataset

![[ABot-M0_fig2_uniact_dataset.png]]

**说明**：中心饼图展示六个数据源及 embodiment 比例；周围展示不同机器人、场景、任务、视角和视觉风格。OXE-AugE 占比最大，双臂数据约 17.2%。

### Figure 3: Model architecture of ABot-M0

![[ABot-M0_fig3_architecture.png]]

**说明**：ABot-M0 从 UniACT 数据开始，经 Qwen3-VL 和 action expert 输出动作；右侧展示 Stage 1 fast token 预训练与 Stage 2 Action Manifold Learning SFT。

### Figure 4: Action Manifold

![[ABot-M0_fig4_action_manifold.png]]

**说明**：左图解释 [[Action Manifold Hypothesis]]：合理动作位于低维动作流形，噪声/速度预测容易偏离流形；右图展示 DiT 接收 noised action、VLM feature、3D feature、robot state 并直接预测动作。

### Figure 5: Embodiment distribution under sampling strategies

![[ABot-M0_fig5_sampling_distribution.png]]

**说明**：比较 Trajectory-Uniform、Task-Uniform、Embodiment-Uniform 三种采样策略。Task-Uniform 在数据规模与 embodiment 多样性之间取得更好的折中。

### Figure 6: Skill sampling characteristics

![[ABot-M0_fig6_skill_sampling.png]]

**说明**：用 rank-probability、Lorenz curve、Coverage@T 分析 skill 采样。Task-Uniform 的概率集中度更低、覆盖新技能更快。

### Figure 7: Validation performance across bimanual embodiments

![[ABot-M0_fig7_bimanual_validation.png]]

**说明**：在 RoboCOIN 双臂 embodiment 上，Task-Uniform 和 Embodiment-Uniform 的 MAE 相近，Trajectory-Uniform 明显更差。

### Figure 8: Dataset-wise validation performance

![[ABot-M0_fig8_dataset_validation.png]]

**说明**：Task-Uniform 在 OXE、AgiBot-Beta、RoboCOIN 三个数据源上整体 MAE 最低，说明 task-level 组织比简单按轨迹或 embodiment 均匀更稳。

### Figure 9: VLM feature interaction

![[ABot-M0_fig9_vlm_feature_interaction.png]]

**说明**：比较 final layer、intermediate layer、multi-layer 以及 action query。最终层原始特征最好，query/concat 不能稳定带来收益。

### Figure 10: 3D information injection

![[ABot-M0_fig10_3d_injection.png]]

**说明**：3D feature 在 action expert 前融合；三种策略为 concat、cross-attention、Q-Former。实验表明 single-layer cross-attention 最优。

### Table 1: Open-source VLA datasets

| Dataset | Scale | Emb. Adopted | 类型 | Strengths | Limitations |
|---------|-------|--------------|------|-----------|-------------|
| OXE | >1.0M | 11 | Single | Large scale, diverse scene | Low complexity, short trajectory, poor quality |
| OXE-AugE | 4.4M | 9 | Single | Inherited from OXE, balanced embodiment | Synthetic video, visual artifacts, physical unreality |
| AgiBot-Beta | >1.0M | 1 | Dual | Long-horizon, rich atomic skill, high quality | Single embodiment |
| RoboCOIN | 180K | 8 | Single+Dual | Hierarchical label, diverse dual-arm | Limited scale, no single-arm |
| RoboMind | 107K | 7 | Single+Dual | Unified protocol, long-horizon task, single & dual-arm | Nonstandard data format |
| Galaxea | 100K | 1 | Dual | Long-horizon, subtask annotation, rich sensor data | Single embodiment |
| Others | <=100K | - | Task-specific | Task-specific design | Fixed embodiment, inconsistent format, adaptation cost |

**表格说明**：没有单一数据集同时满足规模、质量、embodiment 多样性三点，因此本文主张通过系统数据工程构建 unified pre-training set。

### Table 2: Sampling strategy on LIBERO-Plus

| Method | Total |
|--------|-------|
| Trajectory-Uniform | 71.3 |
| Embodiment-Uniform | 71.6 |
| **Task-Uniform** | **72.4** |

**表格说明**：Task-Uniform 的 downstream 表现最好，和 MAE 分析一致。

### Table 3: LIBERO main results

| Method | L-Spatial | L-Object | L-Goal | L-Long | Average |
|--------|-----------|----------|--------|--------|---------|
| Diffusion Policy | 78.5 | 87.5 | 73.5 | 64.8 | 76.1 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0-Fast | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0 | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| F1 | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| InternVLA-M1 | 98.0 | 99.0 | 93.8 | 92.6 | 95.9 |
| Discrete Diffusion VLA | 97.2 | 98.6 | 97.4 | 92.0 | 96.3 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T-N1.6 | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| **ABot-M0** | **98.8** | **99.8** | **99.0** | 96.6 | **98.6** |

**表格说明**：ABot-M0 在平均成功率上达到 98.6%，在 Object/Goal 任务上特别强。

### Table 4: Zero-shot LIBERO-Plus

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|--------|--------|-------|----------|-------|------------|-------|--------|-------|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| π0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| **ABot-M0** | 60.4 | **67.9** | **86.4** | **96.2** | 91.6 | **86.4** | **82.6** | **80.5** |

**表格说明**：ABot-M0 的跨扰动 zero-shot robustness 明显领先，尤其在 robot / language / noise / layout 维度优势大。

### Table 5: RoboCasa GR1 tabletop tasks

| 指标 | GR00T-N1.6 | Qwen3GR00T | Qwen3PI | Qwen3OFT | Qwen3FAST | ABot-M0 |
|------|------------|------------|---------|----------|-----------|---------|
| Average over 24 tasks | 47.6 | 47.8 | 43.9 | 48.8 | 39.0 | **58.3** |

**表格说明**：ABot-M0 在 29 维动作空间、chunk size 16 的高维动作预测下仍显著优于噪声预测专家和 action regression 基线。原表 24 个任务中，ABot-M0 在 PnPBottleToCabinetClose、PnPCanToDrawerClose、PnPWineToCabinetClose 等任务提升尤其明显。

### Table 6: RoboTwin 2.0 benchmark

| Method | Clean Average | Randomized Average |
|--------|---------------|--------------------|
| π0.5 | 42.98 | 43.84 |
| X-VLA | 72.80 | 72.84 |
| **ABot-M0** | **86.06** | **85.08** |

**表格说明**：ABot-M0 在 clean 与 randomized 两种设置下都超过 80%，说明其对背景、桌面杂物、桌高、光照扰动有较强泛化。

### Table 7: Action Manifold Learning ablation

| Setting | Qwen3-VL-GR00T Total | ABot-M0 Total | 结论 |
|---------|----------------------|---------------|------|
| Steps 4, chunk 8 | 69.3 | **71.0** | 默认设置 AML 更强 |
| Steps 2 | 67.2 | **69.7** | 极少步数下直接预测动作更稳 |
| Steps 10 | 68.6 | **70.2** | 更多步数没有明显收益 |
| Chunk 10 | 69.3 | **72.4** | 更合适的 chunk 进一步提升 AML |
| Chunk 30 | 45.7 | **62.8** | 高维长 chunk 下 noise prediction 崩得更明显 |

**表格说明**：最关键的是 chunk 30：GR00T 相比默认下降 23.6，ABot-M0 只下降 8.2，支持 Action Manifold Hypothesis。

### Table 8: VLM feature interaction ablation

| Layers | Feature | Query | Total |
|--------|---------|-------|-------|
| Last | Yes | No | **71.0** |
| Last | No | Yes | 70.0 |
| Intermediate | Yes | No | 69.0 |
| Intermediate | No | Yes | 68.3 |
| Last 16 layers | Yes | No | 67.4 |
| Last 16 layers | No | Yes | 65.2 |
| Last 16 layers | Yes | Yes | 63.8 |

**表格说明**：最终层原始特征最优，多层聚合和 query concat 不一定有益。

### Table 9: 3D feature injection on LIBERO

| Method | L-Spatial | L-Object | L-Goal | L-Long | Average |
|--------|-----------|----------|--------|--------|---------|
| Baseline | 97.8 | 98.2 | 94.6 | 90.8 | 95.4 |
| VGGT cross attention | 99.4 | 99.2 | 96.2 | 95.6 | 97.6 |
| VGGT concat | 99.4 | 98.6 | 96.2 | 93.0 | 96.8 |
| VGGT Q-former | 99.2 | 99.2 | 97.2 | 94.0 | 97.4 |
| Qwen-Image-Edit 1 view | 99.4 | 99.4 | 97.2 | 96.4 | 98.1 |
| **Qwen-Image-Edit 2 views** | **99.6** | 99.0 | **97.4** | **96.6** | **98.2** |

**表格说明**：3D prior 对 LIBERO 长时序和目标任务有明显帮助，多视图合成进一步提升平均成功率。

### Table 10: 3D feature injection on LIBERO-Plus

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|--------|--------|-------|----------|-------|------------|-------|--------|-------|
| Baseline | 32.9 | 50.8 | 86.3 | 85.7 | 86.3 | 62.0 | 73.6 | 66.4 |
| **VGGT cross attention** | 45.8 | **53.8** | 86.8 | **97.2** | **93.9** | **65.5** | 69.8 | **71.1** |
| VGGT concat | 41.2 | 51.2 | 87.0 | 93.7 | 88.4 | 63.9 | 70.6 | 68.9 |
| VGGT Q-former | 44.3 | 51.0 | 87.2 | 95.1 | 91.3 | 62.0 | 71.0 | 69.6 |
| Qwen-Image-Edit 1 view | 38.5 | 52.6 | 87.5 | 94.9 | 90.3 | 62.5 | 64.6 | 68.0 |
| Qwen-Image-Edit 2 views | **46.7** | 53.5 | **87.6** | 96.6 | 91.8 | 60.6 | 69.5 | 70.2 |

**表格说明**：VGGT + cross-attention 是整体最优；多视图在 camera 扰动上提升最大，验证几何先验确实缓解视角依赖。

---

## 实验

### 数据集与设置

| 项目 | 设置 |
|------|------|
| VLM backbone | [[Qwen3-VL]] 4B |
| Action expert | 0.16B [[Diffusion Transformer]] + [[Action Manifold Learning]] |
| 3D module | [[VGGT]] optional |
| Denoising steps | 4 |
| Action chunk size | 16 |
| Image size | 224 x 224 |
| Pre-training LR | 1e-5 |
| Batch size | 1024 |
| Training steps | 100K |
| Benchmarks | [[LIBERO]]、[[LIBERO-Plus]]、[[RoboCasa]]、[[RoboTwin 2.0]] |

### 关键结果

- **LIBERO**：平均 98.6%，优于 X-VLA 98.1 和 OpenVLA-OFT 97.1。
- **LIBERO-Plus zero-shot**：80.5%，在 robot、language、noise、layout 等扰动上优势明显。
- **RoboCasa GR1**：平均 58.3%，比最强基线 Qwen3OFT 48.8 高 9.5。
- **RoboTwin2.0**：clean 86.06、randomized 85.08，对随机背景、杂物、桌高、光照仍稳定。

### 消融结论

1. **AML 优于 noise prediction**：尤其在 action chunk 变长时，直接预测动作的性能退化更小。
2. **最终层 VLM feature 最好**：机器人预训练后，深层特征已经编码动作相关语义。
3. **3D feature 有效**：VGGT cross-attention 和 Qwen-Image-Edit 多视图能提升空间任务与视角扰动鲁棒性。
4. **Task-Uniform sampling 更稳**：它在 embodiment 覆盖和 skill 覆盖之间平衡更好。

---

## 批判性思考

### 优点

1. **不是单点 trick，而是完整系统工程**：数据清洗、动作标准化、采样策略、动作目标、3D 感知全部对齐。
2. **Action Manifold Learning 的动机清楚**：把动作视作低维可执行结构，解释了为什么长 chunk 和高维控制下 noise prediction 更吃力。
3. **公开数据路线有价值**：强调不依赖 proprietary data，也能通过系统整合获得强 VLA。
4. **消融较完整**：AML、VLM feature、3D injection、sampling 都有对应实验。

### 局限

1. **真实机器人结果缺失或不足**：主结果集中在仿真 benchmark，实际硬件迁移仍需验证。
2. **UniACT 数据质量仍强依赖规则管线**：清洗、标准化、过滤策略是否会引入偏差，需要更多透明统计。
3. **Action Manifold Hypothesis 仍偏经验性**：实验支持直接动作预测，但还没有严格刻画动作流形维度、曲率或可达性约束。
4. **3D 模块是后验注入**：作者自己也指出未来应把 3D 表示学习内化到预训练，而不是 SFT 阶段外挂。
5. **部分 benchmark 提升可能来自数据覆盖**：需要进一步确认公开训练数据与评测任务之间的语义/场景重叠程度。

### 潜在改进方向

1. 用自监督深度、pose、scene flow 任务在 VLA pre-training 阶段内建 3D prior。
2. 将触觉、力矩、温度等主动交互传感器统一到动作空间。
3. 将 action manifold 从经验假设转化为显式约束，例如可达集、动力学约束、接触稳定性。
4. 在真实双臂/灵巧手/移动操作场景中验证长 horizon 和高维 action chunk。

---

## 关联笔记

### 方法相关

- [[Vision-Language-Action]]：ABot-M0 所属的大模型机器人策略范式。
- [[Action Manifold Learning]]：本文核心动作生成目标。
- [[Diffusion Transformer]]：动作专家 backbone。
- [[Flow Matching]]：训练与推理中的连续流建模基础。
- [[Action Chunking]]：一次预测多步动作序列。

### 数据与评测相关

- [[UniACT-dataset]]：本文构建的统一训练数据集。
- [[LIBERO]] / [[LIBERO-Plus]]：主要单臂仿真评测。
- [[RoboCasa]]：GR1 tabletop 双臂/全身动作评测。
- [[RoboTwin 2.0]]：多任务随机化仿真评测。

### 感知相关

- [[Qwen3-VL]]：VLM backbone。
- [[VGGT]]：单视图 3D feature 模块。
- [[Qwen-Image-Edit]]：多视图合成与隐式 3D feature 来源。

---

## 速查卡片

> [!summary] ABot-M0
> - **核心**：开放机器人数据 + 统一动作表示 + Action Manifold Learning + 可插拔 3D 感知
> - **数据**：UniACT-dataset，6M+ trajectories，9500+ hours，20+ embodiments
> - **动作**：EEF delta action；双臂统一 14D；rotation vector；pad-to-dual-arm
> - **模型**：Qwen3-VL 4B + 0.16B DiT action expert；默认 4 denoising steps，chunk size 16
> - **关键结论**：直接预测 action 在高维/长 chunk 下比 noise prediction 更稳
> - **结果**：LIBERO 98.6，LIBERO-Plus 80.5，RoboCasa 58.3，RoboTwin2.0 clean 86.06 / randomized 85.08
> - **代码**：github.com/amap-cvlab/ABot-Manipulation

---

*笔记创建时间: 2026-07-02*
