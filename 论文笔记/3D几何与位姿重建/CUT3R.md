---
title: "Continuous 3D Perception Model with Persistent State"
method_name: "CUT3R"
authors: [Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa]
year: 2025
venue: CVPR 2025 (Oral)
tags: [3d-reconstruction, persistent-state, recurrent-model, online-perception, pointmap, dynamic-scene]
zotero_collection: 6-3D视觉
image_source: online
arxiv: https://arxiv.org/abs/2501.12387
arxiv_html: https://arxiv.org/html/2501.12387v1
created: 2026-07-02
---

# 论文笔记：Continuous 3D Perception Model with Persistent State (CUT3R)

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[CUT3R]]（Continuous Updating Transformer for 3D Reconstruction）|
| 机构 | UC Berkeley，Google DeepMind |
| 日期 | 2025-01-21 (arXiv v1) |
| 会议/来源 | CVPR 2025 (Oral) |
| 项目主页 | [cut3r.github.io](https://cut3r.github.io/) |
| 代码 | [github.com/CUT3R/CUT3R](https://github.com/CUT3R/CUT3R) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.12387) / [HTML](https://arxiv.org/html/2501.12387v1) |
| 相关工作 | [[DUSt3R]]、[[MASt3R]]、[[Spann3R]]、[[MonST3R]]、[[Point Map]]、[[Global Scene Representation]] |

---

## 一句话总结

> CUT3R 用一个**持久状态（persistent state）的循环模型**在线处理任意长度的图像流或无序照片，随每帧读取/更新状态并输出统一坐标系下的度量尺度 pointmap，无需全局对齐优化，同时支持静态与动态场景，还能对未观测视角"探测"出未见结构。

---

## 核心贡献

1. **持久状态循环模型**：用一组 state token 在线编码场景全局信息，随每帧读取/更新，实现**隐式对齐**而非优化对齐。
2. **state–image 双向交互机制**：state-update（把新图像信息写入状态）与 state-readout（从状态读出历史上下文），由两个互联的 transformer decoder 联合实现。
3. **统一在线框架**：处理视频流与无序照片、静态与动态场景，无需 [[Global Optimization|全局对齐]]。
4. **[[Raymap]] 查询能力**：可在虚拟未观测视角处"幻想（hallucinate）"出未见结构，类似图像级 MAE 但利用状态中的 3D 全局上下文。
5. **单次前向多输出**：一次前向即输出 self/world 两个 pointmap + 6-DoF 相机位姿，约 17 FPS（约为 DUSt3R-GA 的 25×）。

---

## 问题背景

### 要解决的问题

像人类连续观察世界那样，用一个统一模型从纯 RGB 输入**在线**求解广泛的 3D/4D 任务（深度、位姿、稠密重建），并随每次观测持续更新场景理解。

### 现有方法的局限

- **[[DUSt3R]] / [[MASt3R]]**：成对预测 pointmap，多帧需 [[Global Optimization|全局对齐优化]]，且假设场景静态，速度慢、非在线。
- **[[Spann3R]]**：引入空间记忆做增量式重建，但仍有累积漂移、动态处理不佳。
- **[[MonST3R]]**：面向动态场景，但依赖全局对齐优化，非在线。

### 本文动机

需要一个**持续的 recurrent state** 做在线感知——随每帧更新对场景的内部理解，通过状态实现隐式对齐，从而同时适应静态与动态场景并保持在线。

---

## 方法详解

### 总体架构

- **输入**：任意长度的图像流或无序照片集合。
- **编码**：共享 [[Vision Transformer|ViT-Large]] 编码器把每张图像编码为 tokens（16×16 patch）。
- **状态**：一组 state token（**768 个 token，每个维度 768**），输入前初始化为跨场景共享的**可学习 token**。
- **state–input 双向交互**：图像 token 与状态双向交互——state-update 写入、state-readout 读出——由两个互联的 [[Vision Transformer|ViT-Base]] decoder 联合作用，每个 block 做 [[Cross-Attention]]。
- **pose token**：图像 token 前预置一个可学习 pose token $z$，其输出捕获图像级场景信息（如 ego-motion）。
- **输出**：两个带置信度的 pointmap——一个在图像自身坐标系（"self"），一个在"world"坐标系（首帧坐标系）；同时预测帧间 6-DoF 相对位姿。

作者指出同时预测 self/world pointmap 加位姿的**冗余设计简化训练**，可直接监督并在部分标注数据集上训练。

### 用未观测视角查询状态（[[Raymap]]）

- 虚拟相机作为**查询**来抽取状态信息，其内外参编码为 **raymap**——逐像素编码光线起点与方向的 6 通道图像。
- 一个独立轻量 transformer（2 个 block）编码 raymap，经同一共享权重 decoder 与当前状态交互，但**不更新状态**（raymap 仅查询，不引入新内容）。
- **color head** 解码每条光线的颜色。类比 MAE：补全发生在图像级，利用状态中捕获的 3D 场景全局上下文。

### 训练策略——四阶段课程

- **Stage 1**：4 视角序列，主要静态数据集（224×224）。
- **Stage 2**：加入动态场景 + 部分标注数据集（224×224）。
- **Stage 3**：更高分辨率、多样长宽比，最大边 512px。
- **Stage 4**：**冻结 encoder**，在 **4–64 视角**序列上训练 decoder + heads，学习长上下文/跨场景推理。

### 静态 vs 动态处理

与假设静态场景的全局对齐方法不同，CUT3R **通过状态做隐式对齐**，同时适应静态与动态场景并保持在线。

---

## 关键公式

### 公式1：[[Vision Transformer|图像编码]]

$$
\mathbf{F}_t = \mathrm{Encoder}_i(\mathbf{I}_t)
$$

**含义**：共享 ViT 把第 $t$ 帧编码为图像 token $\mathbf{F}_t$。

### 公式2：[[Global Scene Representation|状态–输入交互]]

$$
[\mathbf{z}_t', \mathbf{F}_t'],\ \mathbf{s}_t = \mathrm{Decoders}([\mathbf{z}, \mathbf{F}_t], \mathbf{s}_{t-1})
$$

**含义**：两个互联 decoder 同时更新状态并读出上下文。

**符号说明**：

- $\mathbf{s}_{t-1}, \mathbf{s}_t$：交互前/后的 state token。
- $\mathbf{F}_t'$：被状态信息丰富后的图像 token。
- $\mathbf{z}_t'$：捕获 ego-motion 的 pose token 输出。

### 公式3：self 坐标系 pointmap head

$$
\hat{\mathbf{X}}_t^{\text{self}}, \mathbf{C}_t^{\text{self}} = \mathrm{Head}_{\text{self}}(\mathbf{F}_t')
$$

**含义**：DPT head 输出输入图像自身坐标系下的 pointmap 与置信度。

### 公式4：world 坐标系 pointmap head

$$
\hat{\mathbf{X}}_t^{\text{world}}, \mathbf{C}_t^{\text{world}} = \mathrm{Head}_{\text{world}}(\mathbf{F}_t', \mathbf{z}_t')
$$

**含义**：输出世界坐标系（首帧坐标系）下的 pointmap 与置信度。

### 公式5：位姿 head

$$
\hat{\mathbf{P}}_t = \mathrm{Head}_{\text{pose}}(\mathbf{z}_t')
$$

**含义**：MLP 从 pose token 回归 camera-to-world 的 6-DoF 位姿。

### 公式6：[[Raymap|raymap 编码]]

$$
\mathbf{F}_r = \mathrm{Encoder}_r(\mathbf{R}),
\qquad
\hat{\mathbf{I}}_r = \mathrm{Head}_{\text{color}}(\mathbf{F}_r')
$$

**含义**：轻量 transformer 编码虚拟相机 raymap $\mathbf{R}$，与状态交互后由 color head 解出颜色，实现未观测视角查询。

### 公式7：[[Point Map|置信度加权 3D 回归损失]]

$$
\mathcal{L}_{\text{conf}} = \sum_{(\hat{\mathbf{x}},c)\in(\hat{\mathcal{X}},\mathcal{C})}\left(c\cdot\left\|\frac{\hat{\mathbf{x}}}{\hat{s}}-\frac{\mathbf{x}}{s}\right\|_2 - \alpha\log c\right)
$$

**含义**：延续 [[MASt3R]] 的置信度加权回归；$\hat{s}, s$ 为尺度归一化因子，度量真值时令 $\hat{s}:=s$。

### 公式8：[[Camera Pose Estimation|位姿损失]]

$$
\mathcal{L}_{\text{pose}} = \sum_{t=1}^{N}\left(\left\|\hat{\mathbf{q}}_t - \mathbf{q}_t\right\|_2 + \left\|\frac{\hat{\boldsymbol{\tau}}_t}{\hat{s}} - \frac{\boldsymbol{\tau}_t}{s}\right\|_2\right)
$$

**含义**：分别监督四元数 $\mathbf{q}$（旋转）与平移 $\boldsymbol{\tau}$；平移做尺度归一化。

### 公式9：RGB 损失（raymap 输入）

$$
\mathcal{L}_{rgb} = \|\hat{\mathbf{I}}_r - \mathbf{I}_r\|_2^2
$$

**含义**：监督从虚拟视角查询状态解码出的颜色。

---

## 关键图表

> 图表来源：论文 arXiv HTML v1。共 6 张 Figure、6 张 Table。图片 URL 前缀 `https://arxiv.org/html/2501.12387v1/`。

### Figure 1: Continuous 3D Perception

![Figure 1](https://arxiv.org/html/2501.12387v1/x1.png)

**说明**：给定 RGB 图像流，在线连续稠密 3D 重建，每帧同时估计相机参数与稠密几何；支持视频/稀疏照片、静态与动态场景。

### Figure 2: Querying Unseen Regions

![Figure 2](https://arxiv.org/html/2501.12387v1/x2.png)

**说明**：给定虚拟相机查询（蓝色所示），可推断场景中**未观测**部分的结构。

### Figure 3: Method Overview

![Figure 3](https://arxiv.org/html/2501.12387v1/x3.png)

**说明**：用持久状态从图像流在线稠密重建；共享 ViT 编码 → state update + state readout（两个互联 decoder）→ 输出 world/camera pointmap 与 camera-to-world 变换；[[Raymap]] 查询预测未观测区域（不更新状态，蓝框标记）。

### Figure 4: In-the-wild Videos

![Figure 4](https://arxiv.org/html/2501.12387v1/x4.png)

**说明**：野外互联网视频的定性结果，与 [[Spann3R]]、[[MonST3R]] 对比。

### Figure 5: State Update Analysis

![Figure 5](https://arxiv.org/html/2501.12387v1/x5.png)

**说明**：相比纯 online，revisiting（用完整上下文的最终状态重新处理）引入全局上下文改善重建。

### Figure 6: Inferring New Structure via State Readout

![Figure 6](https://arxiv.org/html/2501.12387v1/x6.png)

**说明**：从状态读出推断未观测结构。各行：输入图像；GT 图像（用于查询，不给模型）；预测深度；仅输入的 pointmap；共享坐标系下合并的 pointmap。

### Table 1: 单帧深度评估（Abs Rel↓ / δ<1.25↑）

| Method | Sintel | Bonn | KITTI | NYU |
|---|---|---|---|---|
| [[DUSt3R]] | 0.424 | 0.141 | 0.112 | 0.080 |
| [[MASt3R]] | 0.340 | 0.142 | 0.079 | 0.129 |
| [[MonST3R]] | 0.358 | 0.076 | 0.100 | 0.102 |
| [[Spann3R]] | 0.470 | 0.118 | 0.128 | 0.122 |
| **Ours** | 0.428 | **0.063** | 0.092 | 0.086 |

**表格说明**：Bonn、NYU-v2 领先；KITTI 第二（列为 Abs Rel）。

### Table 2: 视频深度评估（per-sequence scale）

| Method | Optim/Onl | Sintel AbsRel | Bonn AbsRel | KITTI AbsRel | FPS |
|---|---|---|---|---|---|
| DUSt3R-GA | Optim | 0.656 | 0.155 | 0.144 | 0.76 |
| MonST3R-GA | Optim | 0.378 | 0.067 | 0.168 | 0.35 |
| [[Spann3R]] | Onl | 0.622 | 0.144 | 0.198 | 13.55 |
| **Ours** | Onl | **0.421** | 0.078 | **0.118** | **16.58** |

**表格说明**：在线方法中显著超过 Spann3R，接近或超过需全局对齐的 MonST3R-GA；度量尺度下也大幅优于 MASt3R-GA。

### Table 3: 相机位姿估计（ATE↓ / RPE trans↓ / RPE rot↓）

| Method | Onl | Sintel ATE | TUM ATE | ScanNet ATE |
|---|---|---|---|---|
| MonST3R-GA | Optim | 0.111 | 0.098 | 0.077 |
| DUSt3R | Onl | 0.290 | 0.140 | 0.246 |
| [[Spann3R]] | Onl | 0.329 | 0.056 | 0.096 |
| **Ours** | Onl | **0.213** | **0.046** | 0.099 |

**表格说明**：Sim(3) 对齐后评估，无需相机标定；在线方法中最佳。

### Table 4: 3D 重建（7-Scenes & NRGBD；Acc↓/Comp↓/NC↑）

| Method | O/On | 7S Acc | 7S Comp | NR Acc | NR Comp | FPS |
|---|---|---|---|---|---|---|
| DUSt3R-GA | Optim | 0.146 | 0.181 | 0.144 | 0.154 | 0.68 |
| MASt3R-GA | Optim | 0.185 | 0.180 | 0.085 | 0.063 | 0.34 |
| [[Spann3R]] | Onl | 0.298 | 0.205 | 0.416 | 0.417 | 12.97 |
| **Ours** | Onl | **0.126** | **0.154** | **0.099** | **0.076** | **17.00** |

**表格说明**：稀疏视角评估（7-Scenes 3–5 帧，NRGBD 2–4 帧），超过所有对比方法（含离线 GA），速度约 DUSt3R-GA 的 25×。

### Table 5: State Update 分析（消融）

| Method | 7S Acc | 7S Comp | NR Acc |
|---|---|---|---|
| DUSt3R-GA | 0.146 | 0.181 | 0.144 |
| Ours | 0.126 | 0.154 | 0.099 |
| **Ours Revisit** | **0.113** | **0.107** | **0.094** |

**表格说明**：Revisiting（先看完所有图像冻结最终状态再重新处理）进一步提升精度，证明状态携带全局上下文的作用。

### Table 6: 训练数据集

共 **32 个**数据集，覆盖 Indoor/Outdoor/Object-Centric/Mixed，含 metric/non-metric、real/synthetic、static/dynamic、cam-only、single-view 多种属性（如 ARKitScenes、ScanNet++、TartanAir、Waymo、BEDLAM、PointOdyssey、CO3Dv2、RealEstate10K、HOI4D 等）。部分数据集仅用相机参数监督（无稠密深度）。

---

## 实验

### 数据集与设置

| 项目 | 设置 |
|------|------|
| 编码器 | [[Vision Transformer|ViT-Large]]（[[DUSt3R]] 权重初始化）|
| 解码器 | ViT-Base；state = 768 token × 768 dim |
| raymap 编码器 | 2 个 block |
| head | Head_self / Head_world 为 DPT；Head_pose 为 MLP |
| 优化器 | Adam-W，初始 lr 1e-4，warmup + 余弦衰减 |
| 硬件 | 8× A100 (80GB) |
| 评测 | Sintel、Bonn、KITTI、NYU-v2（深度）；Sintel/TUM/ScanNet（位姿）；7-Scenes、NRGBD（重建）|

### 关键结果

- 在线方法中位姿估计最佳；视频深度显著超过 [[Spann3R]]，接近/超过需全局对齐的 MonST3R-GA。
- 3D 重建在 7-Scenes/NRGBD 上超过所有对比方法（含离线 GA），速度 ~17 FPS（约 25× DUSt3R-GA）。
- 度量尺度深度大幅优于 MASt3R-GA。
- **消融**：Revisiting 进一步提升重建精度，证明状态携带全局上下文。

---

## 批判性思考

### 优点

1. **持久状态实现隐式对齐**：无需全局优化即可在线拼接，速度快 25×。
2. **静态 + 动态统一**：状态机制天然适配动态场景，无需专门分支。
3. **可查询未观测视角**：raymap + color head 让模型能"幻想"未见结构，独特能力。
4. **单模型多任务**：深度、位姿、重建、4D 一次前向搞定，性能全面 SOTA/competitive。

### 局限

1. **缺少全局对齐 → 长序列漂移（drift）**：这也正是后续 [[TTT3R]] 要修的核心问题。
2. **结构生成确定性 → 可能模糊**：查询未观测区域时结果 blurry。
3. **循环网络训练耗时**：四阶段课程 + 8×A100，训练成本高。

### 潜在改进方向

1. 引入更好的状态更新规则缓解长序列遗忘（[[TTT3R]] 的方向）。
2. 生成式建模替代确定性结构预测，减少模糊。
3. 更高效的循环训练策略。

---

## 关联笔记

### 基于

- [[DUSt3R]]：编码器权重初始化来源，pointmap 范式。
- [[MASt3R]]：置信度加权回归损失沿用。
- [[Spann3R]]：同为在线增量式，本文用持久状态替代显式空间记忆。

### 对比

- [[MonST3R]]：动态场景方法，但依赖全局对齐、非在线。
- [[Global Optimization]]：本文用隐式状态对齐消除的对象。

### 方法相关

- [[Global Scene Representation]]：持久 state token 编码的全局场景表示。
- [[Raymap]]：未观测视角查询的核心表示。
- [[Cross-Attention]]：state–image 交互机制。
- [[Point Map]]：self/world 双 pointmap 输出。

### 谱系相关

- [[TTT3R]]：直接在 CUT3R 状态更新上做 test-time training 改进，修复其长序列遗忘。
- [[MUSt3R]]：DUSt3R 谱系的多视图记忆方法。

---

## 速查卡片

> [!summary] CUT3R
> - **核心**：持久状态循环模型，在线读写 state，隐式对齐输出统一坐标系 pointmap
> - **状态**：768 token × 768 dim；state-update + state-readout 双 decoder
> - **能力**：单模型多任务；raymap 查询未观测视角；静态+动态统一
> - **输出**：self/world 双 pointmap + 6-DoF 位姿，一次前向
> - **关键结论**：在线 SOTA，~17 FPS（约 25× DUSt3R-GA）
> - **结果**：7S Acc 0.126 / Bonn 深度 0.063 / TUM ATE 0.046
> - **局限**：无全局对齐 → 长序列漂移（TTT3R 修复）
> - **项目**：cut3r.github.io

---

*笔记创建时间: 2026-07-02*
