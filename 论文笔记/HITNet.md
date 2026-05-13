---
title: "HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching"
method_name: "HITNet"
authors: [Vladimir Tankovich, Christian Häne, Yinda Zhang, Adarsh Kowdle, Sean Fanello, Sofien Bouaziz]
year: 2021
venue: CVPR 2021
tags: [stereo-matching, real-time-stereo, tile-hypothesis, slanted-plane, depth-estimation, hierarchical-refinement]
image_source: mixed
arxiv_html: https://arxiv.org/html/2007.12140
created: 2026-05-12
---

# 论文笔记：HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Google |
| 日期 | July 2020 (CVPR 2021) |
| 项目主页 | — |
| 对比基线 | [[LEAStereo]], [[GANet]], [[PSMNet]], [[EdgeStereo]], [[StereoNet]] |
| 链接 | [arXiv](https://arxiv.org/abs/2007.12140) / [Code](https://github.com/google-research/google-research/tree/master/hitnet) |

---

## 一句话总结

> 将视差图表示为带倾斜平面假设的 tile 图，通过快速多分辨率初始化 + 可微 2D 空间传播迭代精化，实现 0.02s 实时立体匹配。

---

## 核心贡献

1. **Tile Hypothesis 表示**: 将图像划分为 tile，每个 tile 用倾斜平面 $(d, d_x, d_y)$ + 可学习特征描述符表示，实现紧凑且几何感知的视差表示
2. **快速多分辨率初始化**: 在多尺度特征图上高分辨率匹配并选择最低代价的视差假设，无需构建完整 3D cost volume
3. **可微 2D 传播与精化**: 通过 image warping 构建局部 cost volume，用 CNN 预测 tile 假设的更新量和置信度，层次化从低分辨率向高分辨率精化

---

## 问题背景

### 要解决的问题
实时立体匹配需要在毫秒级延迟内完成视差估计，现有深度学习方法虽然精度高但计算量巨大（通常 >100ms），难以满足移动机器人和自动驾驶的实时需求。

### 现有方法的局限
- 基于 3D cost volume 的方法（[[GCNet]], [[PSMNet]]）依赖 3D 卷积，计算量与视差范围成正比
- 下采样 cost volume 的方法虽快但丢失细节，尤其是薄结构
- 高效方法（如 StereoNet）牺牲精度换速度，边缘和细节恢复差
- 传统高效方法（hash-based matching）不可微分，无法端到端学习

### 本文的动机
传统高效立体匹配方法（slanted support windows、spatial propagation）虽非可学习但策略有效。将这些几何先验（倾斜平面表示、空间传播、层次化精化）融入可学习的神经网络框架，可以同时实现高精度和实时速度。

---

## 方法详解

### 模型架构

HITNet 由三个阶段构成：

- **Feature Extraction**: 小型 [[U-Net]] 提取多尺度特征 $\mathcal{E} = \{\mathbf{e}_0, \dots, \mathbf{e}_M\}$，$M$ 为最低分辨率层级（默认 $M=4$，即 $2^4=16\times$ 下采样）
- **Initialization**: 在各尺度上对 $4 \times 4$ tile 进行逐视差匹配，选择代价最低的视差初始化 tile 假设
- **Propagation**: 从最低分辨率开始，通过 warping + 局部 cost volume + CNN 迭代更新 tile 假设，逐层上采样至全分辨率
- **输出**: 全分辨率视差图 + 倾斜平面参数

### 核心模块

#### 模块1: Tile Hypothesis

**设计动机**: 用紧凑的几何表示（倾斜平面 + 特征描述符）替代密集 cost volume，大幅降低存储和计算需求。

**具体实现**:
- 每个 tile 假设为向量 $\mathbf{h} = [d, d_x, d_y, \mathbf{p}]$
  - $d$: tile 中心的视差
  - $(d_x, d_y)$: 视差在 $x, y$ 方向的梯度（描述倾斜平面）
  - $\mathbf{p}$: 可学习的特征描述符
- tile 内每个像素的视差通过平面方程计算

#### 模块2: 多分辨率初始化

**设计动机**: 在高分辨率特征上直接匹配可获得精确初始值，无需存储完整 cost volume。

**具体实现**:
- 对左右图特征用 $4 \times 4$ 卷积提取 tile 特征，左图 stride $4 \times 4$，右图 stride $4 \times 1$（保持视差分辨率）
- 匹配代价：左右 tile 特征的 $\ell_1$ 距离
- 选择代价最低的视差作为初始值
- 通过感知机（$1 \times 1$ conv + leaky ReLU）预测初始特征描述符

#### 模块3: Propagation（空间传播与精化）

**设计动机**: 利用 image warping 构建高效的局部 cost volume，通过 CNN 在空间邻域内传播和融合信息。

**具体实现**:
- **Warping**: 根据当前 tile 假设的局部视差 $\mathbf{d}'$ 将右图特征 warp 到左图坐标系
- **局部 cost volume**: 在当前视差 $\pm 1$ 范围内计算 3 个方向的匹配代价（16 维向量）
- **Tile 增广**: 将 tile 假设与局部 cost volume 拼接作为 CNN 输入
- **Tile Update CNN** $\mathcal{U}_l$: 使用 [[Dilated Convolution|空洞卷积]] 的残差块（无 BN），预测每个假设的更新量 $\Delta \mathbf{h}$ 和置信度 $w$
- **层次化**: 从 $l=M$（最低分辨率，单假设）开始，上采样到 $l=M-1$ 后与该层初始化假设合并（取置信度更高的），迭代至 $l=0$（全分辨率 $4 \times 4$ tile）
- 最终对 $4 \times 4$ tile 做 3 轮额外精化（$4 \times 4 \to 2 \times 2 \to 1 \times 1$ tile），输出逐像素视差

---

## 关键公式

### 公式1: [[Slanted Plane|Tile Hypothesis 定义]]

$$
\mathbf{h} = [\underbrace{d, d_x, d_y}_{\text{plane}}, \underbrace{\mathbf{p}}_{\text{descriptor}}]
$$

**含义**: 每个 tile 用一个倾斜平面（视差 + 梯度）和可学习特征描述符完整表示

**符号说明**:
- $d$: tile 中心的视差值
- $d_x, d_y$: 视差在水平和垂直方向的梯度
- $\mathbf{p}$: 可学习的 tile 特征描述符

### 公式2: [[Stereo Matching|匹配代价]]

$$
\varrho(l, x, y, d) = \|\tilde{\mathbf{e}}^L_{l,x,y} - \tilde{\mathbf{e}}^R_{l,4x-d,y}\|_1
$$

**含义**: 在分辨率 $l$ 和位置 $(x,y)$ 处，视差 $d$ 的匹配代价为左右 tile 特征的 L1 距离

**符号说明**:
- $\tilde{\mathbf{e}}^L_{l,x,y}$: 左图在 $(x,y)$ 处经 $4 \times 4$ 卷积后的 tile 特征
- $\tilde{\mathbf{e}}^R_{l,4x-d,y}$: 右图在视差偏移位置的 tile 特征（stride 1 保持高分辨率）

### 公式3: [[Stereo Matching|初始视差选择]]

$$
d^{\text{init}}_{l,x,y} = \operatorname{argmin}_{d \in [0,D]} \varrho(l, x, y, d)
$$

**含义**: 选择匹配代价最低的视差作为该 tile 的初始值

### 公式4: [[Stereo Matching|初始特征描述符]]

$$
\mathbf{p}^{\text{init}}_{l,x,y} = \mathcal{D}(\varrho(d^{\text{init}}_{l,x,y}), \tilde{\mathbf{e}}^L_{l,x,y}; \boldsymbol{\theta}_{\mathcal{D}_l})
$$

**含义**: 基于最佳匹配代价和左图特征，通过感知机预测初始特征描述符

### 公式5: [[Slanted Plane|Tile 内局部视差]]

$$
\mathbf{d}'_{i,j} = d + (i - 1.5)d_x + (j - 1.5)d_y
$$

**含义**: tile 内像素 $(i,j)$ 的视差由平面方程给出（$i,j \in \{0,...,3\}$）

### 公式6: [[Stereo Matching|局部 Cost Volume]]

$$
\boldsymbol{\phi}(\mathbf{e}, \mathbf{d}') = [c_{0,0}, c_{0,1}, \dots, c_{3,3}]
$$

其中 $c_{i,j} = \|\mathbf{e}^L_{l,4x+i,4y+j} - \mathbf{e}^R_{l,4x+i-\mathbf{d}'_{i,j},4y+j}\|_1$

**含义**: warping 后左右特征在 $4 \times 4$ tile 内的逐像素 L1 距离，形成 16 维 cost 向量

### 公式7: [[Stereo Matching|增广 Tile 输入]]

$$
\mathbf{a}_{l,x,y} = [\mathbf{h}_{l,x,y}, \underbrace{\boldsymbol{\phi}(\mathbf{e}_l, \mathbf{d}'-1), \boldsymbol{\phi}(\mathbf{e}_l, \mathbf{d}'), \boldsymbol{\phi}(\mathbf{e}_l, \mathbf{d}'+1)}_{\text{local cost volume}}]
$$

**含义**: 将 tile 假设与 3 个方向的局部 cost volume 拼接，作为更新 CNN 的输入

### 公式8: [[Stereo Matching|Tile 更新预测]]

$$
(\Delta \mathbf{h}^1_l, w^1, \dots, \Delta \mathbf{h}^n_l, w^n) = \mathcal{U}_l(\mathbf{a}^1_l, \dots, \mathbf{a}^n_l; \boldsymbol{\theta}_{\mathcal{U}_l})
$$

**含义**: CNN 对 $n$ 个 tile 假设同时预测更新量和置信度

### 公式9: [[Contrastive Loss|初始化损失]]

$$
L^{\text{init}}(d^{\text{gt}}, d^{\text{nm}}) = \psi(d^{\text{gt}}) + \max(\beta - \psi(d^{\text{nm}}), 0)
$$

**含义**: 对比损失——最小化正确匹配的代价，同时确保最近非匹配的代价高于 margin $\beta$

**符号说明**:
- $\psi(d) = (d - \lfloor d \rfloor)\varrho(\lfloor d \rfloor + 1) + (\lfloor d \rfloor + 1 - d)\varrho(\lfloor d \rfloor)$: 亚像素线性插值代价
- $d^{\text{gt}}$: 真值视差
- $d^{\text{nm}}$: 排除正确匹配邻域后代价最低的非匹配视差
- $\beta = 1$: margin

### 公式10: [[Robust Loss|传播损失]]

$$
L^{\text{prop}}(d, d_x, d_y) = \rho(\min(|d^{\text{diff}}|, A), \alpha, c)
$$

**含义**: 对视差预测误差施加截断的鲁棒损失（类似 smooth L1 / Huber loss）

### 公式11: [[Slanted Plane|倾斜平面损失]]

$$
L^{\text{slant}}(d_x, d_y) = \left\| \begin{pmatrix} d^{\text{gt}}_x - d_x \\ d^{\text{gt}}_y - d_y \end{pmatrix} \right\|_1 \chi_{|d^{\text{diff}}| < B}
$$

**含义**: 仅在视差预测接近真值时（$|d^{\text{diff}}| < B$），对倾斜平面梯度施加监督

### 公式12: [[Confidence Loss|置信度损失]]

$$
L^w(w) = \max(1-w, 0)\chi_{|d^{\text{diff}}| < C_1} + \max(w, 0)\chi_{|d^{\text{diff}}| > C_2}
$$

**含义**: 正确预测时置信度应高（推向 1），错误预测时应低（推向 0），中间区域不施加损失

**符号说明**:
- $C_1 = 1$: 正确阈值
- $C_2 = 1.5$: 错误阈值

---

## 关键图表

### Figure 1: Example Results / 示例结果

![[HITNet_fig1.png|600]]

**说明**: HITNet 在 SceneFlow、KITTI、ETH3D 和 Middlebury 上的示例结果。预测视差具有准确的深度和清晰的边缘。右列为预测的倾斜平面参数可视化。

### Figure 2: Overview / 方法概览

![[HITNet_fig2.png|600]]

**说明**: HITNet 框架概览。上：U-Net 提取多尺度特征后，初始化阶段在各尺度 $4 \times 4$ tile 上匹配并选择最低代价假设。下：传播阶段从最低分辨率开始，通过 Tile Update CNN 迭代精化假设并逐层上采样。

### Figure 3: Qualitative Results / 定性结果

![[HITNet_fig3.png|600]]

**说明**: SceneFlow 和 KITTI 2012 上的定性结果。展示左图、初始化结果、最终结果、预测倾斜平面和真值。模型能恢复精细细节、无纹理区域和清晰边缘。

### Table 1: SceneFlow 对比

| Method | EPE (px) | Runtime (s) |
|--------|----------|-------------|
| **HITNet XL** | **0.36** | 0.114 |
| HITNet L | 0.43 | 0.054 |
| EdgeStereo | 0.74 | 0.32 |
| LEAStereo | 0.78 | 0.3 |
| GA-Net | 0.84 | 1.6 |
| PSMNet | 1.09 | 0.41 |
| StereoNet | 1.1 | 0.015 |

**说明**: SceneFlow finalpass 上的对比。HITNet XL 达到 0.36 EPE（比次优好 2x），HITNet L 在 0.054s 下达到 0.43 EPE。

### Table 2: ETH3D 对比

| Method | EPE (px) | Bad 1 | Bad 2 | Runtime |
|--------|----------|-------|-------|---------|
| **HITNet (ours)** | **0.20** | **2.79** | **0.80** | **0.02 s** |
| R-Stereo | 0.18 | 2.44 | 0.44 | 0.81 s |
| DN-CSS | 0.22 | 2.69 | 0.77 | 0.31 s |
| AdaStereo | 0.26 | 3.41 | 0.74 | 0.40 s |
| Deep-Pruner | 0.26 | 3.52 | 0.86 | 0.16 s |
| iResNet | 0.24 | 3.68 | 1.00 | 0.20 s |
| Stereo-DRNet | 0.27 | 4.46 | 0.83 | 0.33 s |
| PSMNet | 0.33 | 5.02 | 1.09 | 0.54 s |

**说明**: ETH3D 数据集上 HITNet 排名 1st-4th，Bad 2 指标最优 (0.80%)，运行时间仅 0.02s。

### Table 3: KITTI 2012 & 2015 对比

**KITTI 2012:**

| Method | 2-noc | 2-all | 3-noc | 3-all | EPE-noc | EPE-all |
|--------|-------|-------|-------|-------|---------|---------|
| **HITNet** | **2.00** | 2.65 | **1.41** | **1.89** | **0.4** | **0.5** |
| LEAStereo | 1.90 | 2.39 | 1.13 | 1.45 | 0.4 | 0.5 |
| GANet-deep | 1.89 | 2.50 | 1.19 | 1.6 | 0.4 | 0.5 |
| EdgeStereo-V2 | 2.32 | 2.88 | 1.46 | 1.83 | 0.4 | 0.5 |
| GC-Net | 2.71 | 3.46 | 1.77 | 2.30 | 0.6 | 0.7 |

**KITTI 2015:**

| Method | D1-bg | D1-fg | D1-all | Run-time |
|--------|-------|-------|--------|----------|
| **HITNet** | **1.74** | 3.20 | **1.98** | **0.02s** |
| LEAStereo | 1.40 | 2.91 | 1.65 | 0.3s |
| GANet-deep | 1.48 | 3.46 | 1.81 | 1.8s |
| EdgeStereo-V2 | 1.84 | 3.30 | 2.08 | 0.32s |
| GC-Net | 2.21 | 6.16 | 2.87 | 0.9s |

**说明**: HITNet 在 <100ms 的方法中排名第一。相比 GA-Net 和 LEAStereo 略有差距但速度快 15-90x。

### Table 4: Middlebury V3 对比

| Method | RMS | AvgErr | Bad 0.5 | Bad 1.0 | Bad 2.0 | Bad 4.0 | A50 | Run-time |
|--------|-----|--------|---------|---------|---------|---------|-----|----------|
| **HITNet** | 9.97 | 1.71 | **34.2** | **13.3** | 6.46 | 3.81 | 0.40 | **0.14s** |
| LEAStereo | 8.11 | 1.43 | 49.5 | 20.8 | 2.75 | **2.75** | 0.53 | 2.9s |
| NOSS-ROB | 12.2 | 2.08 | 38.2 | 13.2 | 5.01 | 3.46 | 0.42 | 662s (CPU) |
| LocalExp | 13.4 | 2.24 | 38.7 | 13.9 | 5.43 | 3.69 | 0.43 | 881s (CPU) |
| MC-CNN | 21.3 | 3.82 | 40.7 | 17.1 | 8.08 | 4.91 | 0.45 | 150s |
| EdgeStereo | 9.84 | 2.67 | 55.6 | 32.4 | 18.7 | 10.8 | 0.72 | 0.35s |

**说明**: Middlebury V3 上 HITNet 在 Bad 0.5 和 Bad 1.0 上超越所有端到端学习方法。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[SceneFlow]] | ~35K | 合成，finalpass 版本 | 预训练 |
| [[KITTI-2012]] | 194 对 | 真实驾驶 | 评测 |
| [[KITTI-2015]] | 200 对 | 真实驾驶 | 评测 |
| [[ETH3D]] | — | 灰度高分辨率 | 评测 |
| [[Middlebury]] V3 | 23 对 | 高分辨率室内 | 评测 |

### 实现细节

- **Feature Extractor**: 小型 U-Net（strided conv + transposed conv + leaky ReLU）
- **$M$**: 默认 4（5 级分辨率）
- **$d_{max}$ ($D$)**: 192
- **Tile Update CNN**: 残差块 + [[Dilated Convolution|空洞卷积]]（无 BN）
- **损失函数**: $L^{\text{init}} + L^{\text{prop}} + L^{\text{slant}} + L^w$，在所有尺度和像素上求和
- **损失参数**: $\beta=1$, $A=B=C_1=1$, $C_2=1.5$
- **版本**: HITNet XL（0.114s）和 HITNet L（0.054s）

### 可视化结果

- HITNet 生成具有锐利边缘和精细细节的视差图
- 倾斜平面参数可视化展示了网络对局部几何的理解
- 初始化结果已相当准确，传播阶段主要修复边缘和无纹理区域

---

## 批判性思考

### 优点
1. **速度极快**: KITTI 分辨率仅 0.02s（50 FPS），是同期精度最高的实时方法
2. **几何先验融入设计**: 倾斜平面表示和空间传播将传统方法的有效策略融入可学习框架
3. **多 benchmark 全面领先**: 在 KITTI 2012/2015、ETH3D、Middlebury 的 <100ms 方法中全面第一
4. **不需要 3D cost volume**: 完全基于 2D 操作，内存效率高

### 局限性
1. **需要真值深度训练**: 无法利用无监督或自监督方法，限制了训练数据规模
2. **各 benchmark 分别训练**: 不同数据集使用略有不同的模型配置，缺少统一实验
3. **对光照变化敏感**: 论文承认在 Middlebury DjembeL 场景（强光照变化）上误差较大
4. **倾斜平面假设的局限**: 在高度非平面区域（如树叶、毛发）建模能力有限

### 潜在改进方向
1. 自蒸馏/无监督预训练减少对真值深度的依赖
2. 统一训练策略和模型配置
3. 结合语义信息处理光照变化和非平面区域

### 可复现性评估
- [x] 代码开源（推理和评测脚本）
- [x] 预训练模型
- [ ] 训练细节完整（部分在补充材料中）
- [x] 数据集可获取

---

## 关联笔记

### 基于
- [[PWC-Net]]: 光流中的 warping + coarse-to-fine 思路启发
- [[SOS Stereo]]: slanted support windows 的前身工作（同一团队）

### 对比
- [[RAFT-Stereo]]: 同为迭代优化方法但基于 correlation volume
- [[LEAStereo]]: NAS 搜索的高精度方法但速度慢 15x
- [[GANet]]: guided aggregation 方法，速度慢 90x
- [[PSMNet]]: 经典 3D 卷积方法

### 方法相关
- [[Slanted Plane]]: 核心几何表示——倾斜平面假设
- [[U-Net]]: 特征提取器架构
- [[Dilated Convolution]]: tile update CNN 中使用
- [[Contrastive Loss]]: 初始化阶段的损失函数

### 硬件/数据相关
- [[SceneFlow]]: 预训练数据集
- [[KITTI-2012]]: 经典自动驾驶立体评测
- [[KITTI-2015]]: 自动驾驶立体评测
- [[ETH3D]]: 高精度立体评测
- [[Middlebury]]: 高分辨率室内立体评测

---

## 速查卡片

> [!summary] HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching
> - **核心**: 用倾斜平面 tile 假设替代 3D cost volume，多分辨率初始化 + 2D 空间传播实现实时推理
> - **方法**: Tile Hypothesis $(d, d_x, d_y, \mathbf{p})$ + Multi-scale Initialization + Iterative Propagation
> - **结果**: 0.02s/帧，KITTI 2012/2015 实时方法第一，ETH3D Bad 2 最优 (0.80%)
> - **代码**: https://github.com/google-research/google-research/tree/master/hitnet

---

*笔记创建时间: 2026-05-12*
