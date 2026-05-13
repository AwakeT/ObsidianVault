---
title: "MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion"
method_name: "MonST3R"
authors: [Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, Ming-Hsuan Yang]
year: 2025
venue: ICLR 2025 (Spotlight)
tags: [dynamic-3d-reconstruction, video-depth-estimation, camera-pose-estimation, pointmap, 4d-reconstruction, visual-odometry]
image_source: mixed
arxiv_html: https://arxiv.org/html/2410.03825v1
created: 2026-05-13
---

# 论文笔记：MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UC Berkeley; Google DeepMind; Stability AI; UC Merced |
| 日期 | October 2024 |
| 项目主页 | https://monst3r-project.github.io/ |
| 对比基线 | [[DUSt3R]], [[DepthCrafter]], [[CasualSAM]], [[DROID-SLAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2410.03825) / [Code](https://github.com/junyi42/monst3r) |

---

## 一句话总结

> 通过对 DUSt3R 进行小规模微调，使其 pointmap 表示适配动态场景，实现时变几何估计、视频深度和相机位姿的联合推断。

---

## 核心贡献

1. **动态场景的 Pointmap 表示**: 发现 [[DUSt3R]] 的 [[Point Map|pointmap]] 表示可以自然地扩展到动态场景——为每个时间步估计独立的 pointmap
2. **小规模高效微调策略**: 仅冻结 encoder 微调 decoder 和预测头，在有限数据集上即可学会处理动态场景
3. **动态全局优化**: 引入对齐损失、轨迹平滑损失和光流投影损失，从成对预测恢复全局一致的动态点云和相机轨迹

---

## 问题背景

### 要解决的问题
从包含运动物体的动态视频中估计场景几何（深度、相机位姿、3D 结构），这是视频理解中的基本挑战。

### 现有方法的局限
- 传统管线将问题分解为深度、光流、轨迹估计等子任务，各阶段误差累积
- [[Structure from Motion|SfM]] 和 [[Visual SLAM]] 假设静态场景，运动物体违反 [[Epipolar Constraint|对极约束]]
- [[DUSt3R]] 仅在静态场景上训练，面对运动物体时：(1) 错误地以前景物体为参考对齐背景，(2) 无法正确估计运动前景的深度
- 视频深度方法如 [[DepthCrafter]] 仅输出 scale/shift-invariant 深度，不适合 3D 应用

### 本文的动机
[[DUSt3R]] 的 pointmap 表示天然支持"每时间步估计几何"——动态物体在点云中按其运动出现在不同位置，多帧对齐时利用静态元素即可。关键挑战在于缺乏带动态标注的训练数据。

---

## 方法详解

### 模型架构

MonST3R 基于 **[[DUSt3R]]** 架构进行微调：
- **输入**: 视频帧对 $(I^t, I^{t'})$
- **Backbone**: [[CroCo]] 预训练的 [[Vision Transformer|ViT]] encoder（冻结）
- **核心模块**: ViT decoder + [[Cross-Attention]]（可训练）+ [[DPT]] 预测头（可训练）
- **输出**: 每时间步的 [[Point Map|pointmap]] $X^t \in \mathbb{R}^{H \times W \times 3}$ + 置信度图
- **关键创新**: 每个 pointmap 对应单一时间点，而非 DUSt3R 的静态场景假设

### 核心模块

#### 模块1: 动态场景微调策略

**设计动机**: 保留 [[CroCo]] encoder 中的丰富几何先验，减少数据需求

**具体实现**:
- **冻结 encoder**: 仅微调 decoder 和 [[DPT]] 预测头
- **时间步长采样**: 训练对以步长 1-9 采样，步长 9 概率为步长 1 的 2 倍，增加运动多样性
- **FoV 增强**: 中心裁剪不同尺度，泛化到不同相机内参
- 在 4 个数据集上仅需 2 天 (2× RTX 6000) 即可完成微调

#### 模块2: 静态区域检测

**设计动机**: 在全局优化中需要区分静态和动态区域，以利用静态元素进行相机位姿估计

**具体实现**:
- 比较估计的 [[Optical Flow|光流]] $F_{\text{est}}$ 与相机运动诱导的光流 $F_{\text{cam}}$
- 相机运动光流从 pointmap 推导的深度和相机参数计算
- 通过阈值化差异获得静态掩码 $S^{t \to t'}$

#### 模块3: 动态全局优化

**设计动机**: 将成对 pointmap 预测融合为全局一致的动态点云和相机轨迹

**具体实现**:
- 使用滑动时间窗口（窗口大小 $w$）构建视频图，避免全连接图的高计算开销
- 全局 pointmap 用相机参数 $P^t = [R^t | T^t]$、内参 $K^t$ 和逐帧深度图 $D^t$ 参数化
- 联合优化三项损失

---

## 关键公式

### 公式1: [[Optical Flow|相机运动光流]]

$$
F_{\text{cam}}^{t \to t'} = \pi\bigl(D^{t;t \leftrightarrow t'} \cdot K^{t'} \cdot R^{t \to t'} \cdot (K^t)^{-1} \cdot \hat{x} + K^{t'} \cdot T^{t \to t'}\bigr) - x
$$

**含义**: 假设场景静态时，仅由相机运动产生的光流。用于与实际光流对比检测动态物体。

**符号说明**:
- $x$: 像素坐标矩阵
- $\hat{x}$: 齐次坐标
- $\pi(\cdot)$: 投影函数 $(x,y,z) \to (x/z, y/z)$
- $D^{t;t \leftrightarrow t'}$: 从 pointmap 估计的深度
- $K^t, K^{t'}$: 相机内参
- $R^{t \to t'}, T^{t \to t'}$: 帧间相对旋转和平移

### 公式2: [[Motion Segmentation|静态掩码]]

$$
S^{t \to t'} = \bigl[\alpha > \|F_{\text{cam}}^{t \to t'} - F_{\text{est}}^{t \to t'}\|_{L1}\bigr]
$$

**含义**: 通过比较相机运动光流和实际光流识别静态区域

**符号说明**:
- $\alpha$: 阈值
- $[\cdot]$: Iverson 括号
- $\|\cdot\|_{L1}$: smooth-L1 范数

### 公式3: [[Point Cloud Alignment|对齐损失]]

$$
\mathcal{L}_{\text{align}}(X, \sigma, P_W) = \sum_{W^i \in W} \sum_{e \in W^i} \sum_{t \in e} \|C^{t;e} \cdot (X^t - \sigma^e \cdot P^{t;e} \cdot X^{t;e})\|_1
$$

**含义**: 全局 pointmap 与成对预测 pointmap 之间的对齐约束

**符号说明**:
- $X^t$: 全局 pointmap
- $X^{t;e}$: 边 $e$ 对应的成对 pointmap 预测
- $\sigma^e$: 成对尺度因子
- $C^{t;e}$: 置信度权重
- $P^{t;e}$: 成对相对变换

### 公式4: [[Trajectory Smoothness|相机轨迹平滑损失]]

$$
\mathcal{L}_{\text{smooth}}(X) = \sum_{t=0}^N \bigl(\|(R^t)^\top \cdot R^{t+1} - I\|_F + \|(R^t)^\top \cdot (T^{t+1} - T^t)\|_2\bigr)
$$

**含义**: 约束相邻帧相机位姿的平滑性，防止轨迹突变

**符号说明**:
- $R^t, T^t$: 第 $t$ 帧相机旋转和平移
- $\|\cdot\|_F$: Frobenius 范数
- $I$: 单位矩阵

### 公式5: [[Optical Flow|光流投影损失]]

$$
\mathcal{L}_{\text{flow}}(X) = \sum_{W^i \in W} \sum_{t \to t' \in W^i} \|S^{\text{global};t \to t'} \cdot (F_{\text{cam}}^{\text{global};t \to t'} - F_{\text{est}}^{t \to t'})\|_1
$$

**含义**: 全局 pointmap 和相机位姿在静态区域应与光流估计一致

### 公式6: [[Global Optimization|完整优化目标]]

$$
\hat{X} = \arg\min_{X, P_W, \sigma} \mathcal{L}_{\text{align}} + w_{\text{smooth}} \cdot \mathcal{L}_{\text{smooth}} + w_{\text{flow}} \cdot \mathcal{L}_{\text{flow}}
$$

**含义**: 联合优化全局点云、相机位姿和尺度因子

**符号说明**:
- $w_{\text{smooth}} = 0.01$: 平滑损失权重
- $w_{\text{flow}} = 0.01$: 光流损失权重

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2410.03825v1/x1.png)

**说明**: MonST3R 处理动态视频，输出时变动态点云以及逐帧相机位姿和内参。该表示支持高效计算下游任务，如视频深度估计和动态/静态场景分割。

### Figure 2: DUSt3R 在动态场景的局限

![[MonST3R_fig2_limitation.png]]

**说明**: [[DUSt3R]] 在动态场景中的局限。DUSt3R 以运动前景为参考对齐，导致背景点错位；且无法正确估计前景物体深度，将其放置在背景中。

### Figure 3: 动态全局点云和相机位姿估计

![Figure 3](https://arxiv.org/html/2410.03825v1/x3.png)

**说明**: 给定固定时间窗口，MonST3R 计算成对 pointmap 和光流，然后优化全局点云和逐帧相机位姿。视频深度可直接从统一表示中导出。

### Figure 4: 定性对比

![Figure 4](https://arxiv.org/html/2410.03825v1/x4.png)

**说明**: 与 [[CasualSAM]] 和 [[DUSt3R]] 对比。CasualSAM 相机轨迹可靠但前景几何偶尔失败；DUSt3R 在动态前景处挣扎。MonST3R 同时输出可靠的相机轨迹和完整场景重建。

### Table 1: 训练数据集

| Dataset | Domain | Dynamics | Ratio |
|---------|--------|----------|-------|
| PointOdyssey | Synthetic | Realistic | 50% |
| TartanAir | Synthetic | None | 25% |
| Spring | Synthetic | Realistic | 5% |
| Waymo Perception | Real (Driving) | Driving | 20% |

**说明**: 混合使用合成和真实数据，PointOdyssey 占比最大因其具有逼真的关节动态物体。

### Table 2: 视频深度评估

| Alignment | Method | Sintel AbsRel↓ | Bonn AbsRel↓ | KITTI AbsRel↓ |
|-----------|--------|----------------|--------------|---------------|
| Per-seq s&s | Depth-Anything-V2 | 0.367 | 0.106 | 0.140 |
| Per-seq s&s | DepthCrafter | 0.292 | 0.075 | 0.110 |
| Per-seq s&s | **MonST3R** | **0.335** | **0.063** | **0.104** |
| Per-seq s | DepthCrafter | 0.692 | 0.217 | 0.141 |
| Per-seq s | **MonST3R** | **0.345** | **0.065** | **0.106** |

**说明**: MonST3R 在 metric depth (scale-only) 条件下显著优于 [[DepthCrafter]]，证明其输出的深度具有几何一致性。在 Bonn 和 KITTI 上表现最优。

### Table 3: 单帧深度评估

| Method | Sintel AbsRel↓ | Bonn AbsRel↓ | KITTI AbsRel↓ | NYU-v2 AbsRel↓ |
|--------|----------------|--------------|---------------|----------------|
| DUSt3R | 0.424 | 0.141 | 0.112 | 0.080 |
| MonST3R | 0.345 | 0.076 | 0.101 | 0.091 |

**说明**: 微调后单帧深度性能在动态场景 (Sintel, Bonn, KITTI) 上提升，静态场景 (NYU-v2) 上略有下降但仍具竞争力。

### Table 4: 相机位姿评估

| Category | Method | Sintel ATE↓ | Sintel RPE_t↓ | TUM ATE↓ | ScanNet ATE↓ |
|----------|--------|-------------|---------------|----------|--------------|
| Pose only | DROID-SLAM* | 0.175 | 0.084 | - | - |
| Pose only | LEAP-VO* | 0.089 | 0.066 | 0.068 | 0.070 |
| Joint | CasualSAM | 0.141 | 0.035 | 0.071 | 0.158 |
| Joint | DUSt3R w/mask | 0.417 | 0.250 | 0.083 | 0.081 |
| Joint | **MonST3R** | **0.108** | **0.042** | **0.063** | **0.068** |

**说明**: MonST3R 在联合深度位姿估计中达到最佳精度，在 TUM 上甚至优于需要 GT 内参的 LEAP-VO。*表示需要 GT 相机内参。

### Table 5: 消融实验 (Sintel)

| Variants | ATE↓ | RPE_t↓ | AbsRel↓ | δ<1.25↑ |
|----------|------|--------|---------|---------|
| No finetune (DUSt3R) | 0.354 | 0.167 | 0.482 | 56.5 |
| w/ PO only | 0.220 | 0.129 | 0.378 | 53.7 |
| w/ all 4 datasets | **0.108** | **0.042** | **0.335** | **58.5** |
| Full model finetune | 0.181 | 0.110 | 0.352 | 55.4 |
| Finetune decoder & head | **0.108** | **0.042** | **0.335** | **58.5** |
| w/o flow loss | 0.140 | 0.051 | 0.339 | 57.7 |
| w/o smoothness loss | 0.127 | 0.060 | 0.333 | 58.4 |
| Full | **0.108** | **0.042** | **0.335** | **58.5** |

**关键发现**: (1) 四个数据集全部使用效果最佳；(2) 冻结 encoder 仅微调 decoder+head 是最优策略；(3) 光流损失和平滑损失对位姿估计贡献显著。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| PointOdyssey | 200K 帧 | 合成，逼真动态 | 训练（50%） |
| TartanAir | 1M 帧 | 合成，无动态 | 训练（25%） |
| Spring | 6K 帧 | 合成，户外动态 | 训练（5%） |
| Waymo Perception | 160K 帧 | 真实驾驶，LiDAR 深度 | 训练（20%） |
| Sintel | 合成 | 电影级动态 | 评估 |
| Bonn, KITTI | 真实 | 室内/驾驶 | 深度评估 |
| TUM-dynamics | 真实 | 室内动态 | 位姿评估 |
| ScanNet | 真实 | 室内静态 | 位姿评估 |

### 实现细节

- **Backbone**: CroCo 预训练的 ViT-Base encoder（冻结）
- **优化器**: [[AdamW]]，学习率 5×10⁻⁵，mini-batch size 4/GPU
- **训练**: 25 epochs，每 epoch 20K 样本对
- **硬件**: 2× RTX 6000 48GB，训练 1 天
- **推理**: ~30 秒/60 帧视频（w=9, stride 2），全局优化 ~1 分钟
- **全局优化**: Adam optimizer，300 iterations，学习率 0.01

### 可视化结果

- DAVIS 数据集上的联合重建展示了对运动人体和物体的准确几何估计
- 静态/动态分割掩码准确地区分运动前景和静态背景
- 视频深度时间一致性明显优于单帧方法

---

## 批判性思考

### 优点
1. **优雅的核心洞察**: pointmap 表示天然适配动态场景，无需显式运动表示
2. **极高数据效率**: 仅用 4 个小数据集微调 1 天即可学会处理动态场景
3. **多任务统一**: 单一模型输出深度、位姿、动态/静态分割和 4D 重建
4. **鲁棒性强**: 在 Sintel（合成电影）、TUM（室内动态）、KITTI（驾驶）等多领域均表现优异

### 局限性
1. 全局优化仍需约 1 分钟，非实时
2. 对 out-of-distribution 场景（如开阔地带）效果下降
3. 动态相机内参变化时需要仔细调参
4. 训练数据中缺乏真实室内动态场景，泛化可能受限

### 潜在改进方向
1. 扩大训练数据集规模和多样性
2. 前馈式全局对齐替代测试时优化
3. 与 [[4D Gaussian Splatting]] 结合进行高质量 4D 重建

### 可复现性评估
- [x] 代码开源（github.com/junyi42/monst3r）
- [x] 预训练模型（Google Drive / Hugging Face）
- [x] 训练细节完整
- [x] 数据集可获取

---

## 关联笔记

### 基于
- [[DUSt3R]]: 核心架构和 pointmap 表示来源
- [[CroCo]]: 自监督预训练的 encoder

### 对比
- [[DepthCrafter]]: 专用视频深度方法，输出 scale/shift-invariant 深度
- [[CasualSAM]]: 测试时微调深度的方法
- [[DROID-SLAM]]: 需要 GT 内参的 SLAM 方法
- [[Robust-CVD]]: 光流+掩码联合优化深度位姿

### 方法相关
- [[Point Map]]: 核心表示——将每个像素映射到 3D 空间
- [[Optical Flow]]: 用于构建静态掩码和光流投影损失
- [[PnP]]: 用于从 pointmap 恢复相对位姿
- [[RANSAC]]: 鲁棒位姿估计中使用
- [[Motion Segmentation]]: 隐式实现的动态/静态分割

### 硬件/数据相关
- [[PointOdyssey]]: 主要训练数据集（含逼真动态物体）
- [[TartanAir]]: 无人机飞行合成数据
- [[Waymo Perception]]: 真实驾驶场景数据

---

## 速查卡片

> [!summary] MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion
> - **核心**: 将 DUSt3R 的 pointmap 表示扩展到动态场景，每时间步估计独立 pointmap
> - **方法**: 冻结 encoder + 小规模微调 decoder，滑动窗口全局优化（对齐+平滑+光流损失）
> - **结果**: ICLR 2025 Spotlight；Sintel ATE 0.108，Bonn AbsRel 0.063，KITTI AbsRel 0.104
> - **代码**: https://github.com/junyi42/monst3r

---

*笔记创建时间: 2026-05-13*
