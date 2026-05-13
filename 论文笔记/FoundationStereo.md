---
title: "FoundationStereo: Zero-Shot Stereo Matching"
method_name: "FoundationStereo"
authors: [Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield]
year: 2025
venue: CVPR 2025 (Best Paper Nomination)
tags: [stereo-matching, foundation-model, zero-shot-generalization, side-tuning, cost-volume-filtering, depth-estimation, transformer]
image_source: online
arxiv_html: https://arxiv.org/html/2501.09898
created: 2026-05-12
---

# 论文笔记：FoundationStereo: Zero-Shot Stereo Matching

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA |
| 日期 | January 2025 (CVPR 2025) |
| 项目主页 | https://nvlabs.github.io/FoundationStereo/ |
| 对比基线 | [[CREStereo]], [[RAFT-Stereo]], [[IGEV-Stereo]], [[Selective-IGEV]], [[CroCo v2]], [[MoCha-Stereo]], [[NMRF]] |
| 链接 | [arXiv](https://arxiv.org/abs/2501.09898) / [Code](https://github.com/NVlabs/FoundationStereo) |

---

## 一句话总结

> 构建百万级合成数据集 + 自筛选流水线，设计 Side-Tuning Adapter 适配 DepthAnythingV2 单目先验 + Attentive Hybrid Cost Filtering（APC + Disparity Transformer），首次在零样本设定下全面超越微调方法，Middlebury 和 ETH3D 排行榜双冠。

---

## 核心贡献

1. **FoundationStereo Dataset (FSD)**: 用 NVIDIA Omniverse 渲染 1M 高质量合成立体对，覆盖 47 种场景 / 12 种布局 / 多相机参数，配合迭代自筛选流水线剔除模糊样本
2. **Side-Tuning Adapter (STA)**: 用轻量 CNN（EdgeNeXt-S）侧接冻结的 [[DepthAnythingV2]]，以 4x4 conv 下采样后与 CNN 特征拼接，将互联网级单目先验引入立体匹配
3. **Attentive Hybrid Cost Filtering (AHCF)**: 在沙漏网络中用 3D Axial-Planar Convolution (APC) 解耦空间与视差维度 + Disparity Transformer (DT) 在视差维全局自注意力，增强 cost volume 的长程推理
4. **零样本 SOTA**: 无需逐域微调即在 Middlebury、ETH3D、KITTI 12/15 四大 benchmark 上全面超越现有微调方法

---

## 问题背景

### 要解决的问题
现有立体匹配方法依赖逐域微调才能取得竞争性能，缺乏像单目深度估计那样的"基础模型"级零样本泛化能力。

### 现有方法的局限
- Cost volume 聚合方法（[[PSMNet]], [[GwcNet]]）依赖 3D CNN，内存消耗大、难以处理高分辨率
- 迭代精化方法（[[RAFT-Stereo]], [[CREStereo]]）虽更灵活，但递归更新缺乏长程上下文推理，泛化到新域时性能下降
- 先前泛化工作（DSMNet, Former-RAFT-DAM）仅在 SceneFlow (~40K) 上训练，数据规模和多样性不足
- Transformer 用于单目特征提取（STTR, CroCo v2）但缺少 cost volume 的结构化归纳偏置

### 本文的动机
视觉基础模型（[[DepthAnythingV2]], [[DINOv2]], [[SAM]]）已在单目深度、分割等任务中展现强泛化。本文探索如何将这些互联网级先验有效注入立体匹配，同时通过大规模高质量合成数据和改进的 cost volume 滤波实现真正的零样本基础模型。

---

## 方法详解

### 模型架构

FoundationStereo 整体流程：

- **输入**: 校正后的左右立体图像对 $I_l, I_r \in \mathbb{R}^{H \times W \times 3}$
- **Side-Tuning Adapter (STA)**: 冻结 [[DepthAnythingV2]] 提取多级 ViT 特征，EdgeNeXt-S CNN 提取多级金字塔特征，4x4 conv 下采样 ViT 特征后与 CNN 特征拼接 → 混合特征 $f_l, f_r$
- **Context Network**: 类似 STA 的 CNN 提取上下文特征 $f_c$，初始化 GRU 隐状态
- **Hybrid Cost Volume**: Group-wise Correlation $\mathbf{V}_{\text{gwc}}$ + Concatenation $\mathbf{V}_{\text{cat}}$ 构建 4D cost volume $\mathbf{V}_C$
- **Attentive Hybrid Cost Filtering (AHCF)**: 沙漏网络中 3D APC 替代标准 3D conv + Disparity Transformer 全局推理 → 滤波后 $\mathbf{V}'_C$
- **Initial Disparity**: Soft-argmin 从 $\mathbf{V}'_C$ 回归初始视差 $d_0$
- **Iterative Refinement**: 多级 ConvGRU 迭代精化，每步从 $\mathbf{V}'_C$ 和 correlation volume 查询特征
- **输出**: 全分辨率视差图，通过 [[Convex Upsampling]] 上采样

### 核心模块

#### 模块1: Side-Tuning Adapter (STA)

**设计动机**: DepthAnythingV2 包含丰富的互联网级单目先验（语义 + 几何），但 ViT 特征缺乏立体匹配所需的高频细节。通过侧接 CNN 网络融合 ViT 先验和 CNN 高频特征。

**具体实现**:
- 冻结 DepthAnythingV2-S（ViT-S）提取 4 级特征金字塔 $f^{(i)}_l, f^{(i)}_r$，$i \in \{4, 8, 16, 32\}$
- [[EdgeNeXt]]-S 作为侧网络提取多级 CNN 特征
- 在 1/4 尺度，用 $4 \times 4$ stride-4 卷积下采样 DepthAnythingV2 最终输出头前的特征
- 下采样后与同级 CNN 特征拼接形成混合特征
- 设计选择 (c) 优于直接使用 DPT head (a) 和 ViT-Adapter 交叉注意力 (b)
- 图像尺寸 resize 为 14 的倍数以匹配 ViT patch size
- STA 权重左右图共享

#### 模块2: Attentive Hybrid Cost Filtering (AHCF)

**设计动机**: 标准 $3 \times 3 \times 3$ 卷积的感受野在视差维度上增长缓慢，无法建模大视差范围内的全局关系。

**具体实现**:
- **Hybrid Cost Volume**: $\mathbf{V}_C = [\mathbf{V}_{\text{gwc}}, \mathbf{V}_{\text{cat}}]$，其中 $G=8$ 组相关 + 拼接
- **3D Axial-Planar Convolution (APC)**: 将 $3 \times 3 \times 3$ 卷积解耦为空间卷积 $K_s \times K_s \times 1$ + 视差卷积 $1 \times 1 \times K_d$，类似 3D 版 [[Separable Convolution]]，但仅分离空间和视差维度不分组
  - 视差维度特殊处理：因其编码独特的特征比较信息
  - 在沙漏网络的下/上采样块中均使用 APC
  - 最优 kernel: 空间 $(3,3,1)$，视差 $(1,1,17)$
- **Disparity Transformer (DT)**: 对 $\mathbf{V}_C$ 先用 $4 \times 4 \times 4$ stride-4 的 3D conv 下采样，reshape 为 token 序列（每个视差层一批 token），加 cosine 位置编码，送入 4 层 [[FlashAttention]] transformer encoder，再上采样回原尺寸与沙漏输出相加
  - Cosine 位置编码优于 RoPE（因 cost volume 视差维度大小恒定）
  - 仅在视差维度做自注意力比全 volume 注意力更有效
  - DT 放在沙漏后（parallel）效果最优

#### 模块3: FoundationStereo Dataset (FSD) + 迭代自筛选

**设计动机**: SceneFlow 仅 40K 对、场景单一；现有合成数据集缺乏多样性或真实感。

**具体实现**:
- **NVIDIA Omniverse** 渲染引擎，path-tracing 高真实感渲染
- 47 种场景（室内 + 室外 + 飞行物体）、12 种布局、高质量 3D 资产
- **Domain randomization**: 随机基线、焦距、视角、光照、物体配置
- 分辨率 $1280 \times 720$，含反射、低纹理、严重遮挡
- **Varying baseline + intrinsics**: 不同于其他数据集的固定相机参数
- **迭代自筛选**: 训练初版模型 → 在 FSD 上评测 → BP-2 > 60% 的样本标记为模糊 → 重新生成替换 → 迭代 2 次更新 FSD 和模型
- 最终 1M 对，1000K 规模

---

## 关键公式

### 公式1: [[Correlation Volume|Hybrid Cost Volume 构建]]

$$
\mathbf{V}_{\text{gwc}}(g, d, h, w) = \langle \hat{f}^{(4)}_{l,g}(h, w), \hat{f}^{(4)}_{r,g}(h, w-d) \rangle
$$
$$
\mathbf{V}_{\text{cat}}(d, h, w) = [\text{Conv}(f^{(4)}_l)(h, w), \text{Conv}(f^{(4)}_r)(h, w-d)]
$$
$$
\mathbf{V}_C(d, h, w) = [\mathbf{V}_{\text{gwc}}(d, h, w), \mathbf{V}_{\text{cat}}(d, h, w)]
$$

**含义**: 混合 cost volume 结合组级相关（保留匹配代价结构）和拼接（保留单目先验信息），$G=8$ 组

**符号说明**:
- $\hat{f}$: $L_2$ 归一化后的特征
- $g \in \{1,...,G\}$: 特征组索引
- $d \in \{1,...,D/4\}$: 视差候选
- $[\cdot, \cdot]$: 通道维拼接

### 公式2: [[Transformer|Disparity Transformer]]

$$
\mathbf{Q}_0 = \text{PE}(\text{R}(\text{Conv}_{4 \times 4 \times 4}(\mathbf{V}_C))) \in \mathbb{R}^{(\frac{H}{4} \times \frac{W}{4}) \times C \times \frac{D}{16}}
$$

$$
\mathbf{Q}_1 = \text{Norm}(\text{MultiHead}(\mathbf{Q}_0, \mathbf{Q}_0, \mathbf{Q}_0) + \mathbf{Q}_0)
$$

$$
\mathbf{Q}_2 = \text{Norm}(\text{FFN}(\mathbf{Q}_1) + \mathbf{Q}_1)
$$

**含义**: 对 cost volume 下采样后 reshape 为 token，在视差维度上做全局自注意力，捕获长程视差依赖

**符号说明**:
- $\text{R}(\cdot)$: reshape 操作
- $\text{PE}(\cdot)$: cosine 位置编码
- $h = 4$: 注意力头数
- 输出上采样回原尺寸与沙漏输出相加

### 公式3: [[Soft-Argmin|初始视差预测]]

$$
d_0 = \sum_{d=0}^{\frac{D}{4}-1} d \cdot \text{Softmax}(\mathbf{V}'_C)(d)
$$

**含义**: 对滤波后 cost volume 沿视差维 soft-argmin 回归亚像素精度的初始视差

### 公式4: [[GRU|ConvGRU 迭代精化]]

$$
\mathbf{V}_{\text{corr}}(w', h, w) = \langle f^{(4)}_l(h, w), f^{(4)}_r(h, w') \rangle
$$
$$
\mathbf{F}_V(h, w) = [\mathbf{V}'_C(d_k, h, w), \mathbf{V}_{\text{corr}}(w - d_k, h, w)]
$$
$$
x_k = [\text{Conv}_v(\mathbf{F}_V), \text{Conv}_d(d_k), d_k, c]
$$
$$
z_k = \sigma(\text{Conv}_z([h_{k-1}, x_k]))
$$
$$
r_k = \sigma(\text{Conv}_r([h_{k-1}, x_k]))
$$
$$
\hat{h}_k = \tanh(\text{Conv}_h([r_k \odot h_{k-1}, x_k]))
$$
$$
h_k = (1 - z_k) \odot h_{k-1} + z_k \odot \hat{h}_k
$$
$$
d_{k+1} = d_k + \text{Conv}_\Delta(h_k)
$$

**含义**: 每步从滤波 cost volume 和 correlation volume 查询特征，送入 ConvGRU 更新隐状态并预测视差残差

**符号说明**:
- $\odot$: 逐元素乘积
- $\sigma$: sigmoid
- $c = \text{ReLU}(f_c)$: 上下文特征（含 STA 先验）
- $\mathbf{F}_V$: 查询的 volume 特征
- 使用 3 级 GRU（1/4, 1/8, 1/16）+ attention-based selection

### 公式5: [[L1 Loss|训练损失]]

$$
\mathcal{L} = |d_0 - \overline{d}|_{\text{smooth}} + \sum_{k=1}^{K} \gamma^{K-k} ||d_k - \overline{d}||_1
$$

**含义**: 初始视差用 smooth L1 监督 + 迭代精化用指数加权 L1 监督（$\gamma = 0.9$）

---

## 关键图表

### Figure 1: 零样本预测效果

![[FoundationStereo_fig1.png|600]]

**说明**: FoundationStereo 在 in-the-wild 图像上的零样本预测。方法泛化到室内/室外多种场景、无纹理/反射/半透明/细结构等挑战性物体、复杂光照和不同传感器。

### Figure 2: Overview / 整体架构

![[FoundationStereo_fig2.png|600]]

**说明**: FoundationStereo 架构。Side-Tuning Adapter (STA) 用冻结的 DepthAnythingV2 提供单目先验，CNN 提供高频细节。Attentive Hybrid Cost Filtering (AHCF) 包含 Axial-Planar Convolution (APC) 沙漏和 Disparity Transformer (DT)。初始视差经 soft-argmin 预测后，由多级 ConvGRU 迭代精化。

### Figure 3: STA 设计选择与模块效果

![[FoundationStereo_fig3.png|600]]

**说明**: 左：STA 模块三种设计选择——(a) 直接使用 DPT head，(b) ViT-Adapter 风格交叉注意力，(c) 4x4 conv 下采样后拼接（最优）。右：STA 和 AHCF 各自的效果可视化。STA 帮助处理光照不一致的模糊区域，AHCF 有效恢复细结构。

### Figure 4: FSD 数据集样例与自筛选流水线

![[FoundationStereo_fig4.png|600]]

**说明**: 左：FSD 数据集样本，含室内/室外结构化场景和更多样化的飞行物体。右：迭代自筛选流程——训练初版模型，评测标记模糊样本（BP-2 > 60%），重新生成替换后迭代更新。

### Figure 5: In-the-Wild 定性对比

![[FoundationStereo_fig5.png|600]]

**说明**: 与 CroCo v2、CREStereo、IGEV、Selective-IGEV 的零样本定性对比。场景包括反射表面、半透明物体、重复纹理、细结构。FoundationStereo 在所有场景中视觉效果最优。

### Table 1: 合成训练数据集对比

| 属性 | Sintel | SceneFlow | CREStereo | IRS | TartanAir | FallingThings | UnrealStereo4K | Spring | **FSD (Ours)** |
|------|--------|-----------|-----------|-----|-----------|---------------|----------------|--------|---------------|
| 渲染器 | Blender | Blender | Blender | Unreal | Unreal | Unreal | Unreal | Blender | **NVIDIA Omniverse** |
| 渲染质量 | High | Low | High | High | High | High | High | High | **High** |
| 场景数 | 10 | 9 | 0 | 4 | 18 | 3 | 8 | 47 | **47** |
| 立体对数 | 1K | 40K | 200K | 103K | 306K | 62K | 7.7K | 6K | **1000K** |
| 分辨率 | 1024x436 | 960x540 | 1920x1080 | 960x540 | 640x480 | 960x540 | 3840x2160 | 1920x1080 | **1280x720** |
| 反射 | x | x | v | v | v | v | v | x | **v** |
| 变化基线 | x | x | x | x | x | x | x | 固定 | **变化基线+内参** |

**说明**: FSD 在规模（1M）、场景多样性（47 种）、渲染质量（Omniverse path-tracing）和相机参数多样性上全面领先。

### Table 2: 零样本泛化对比

| Method | Middlebury BP-2 | ETH3D BP-1 | KITTI-12 D1 | KITTI-15 D1 |
|--------|----------------|------------|-------------|-------------|
| CREStereo++ | 14.8 | 4.4 | 4.7 | 5.2 |
| DSMNet | 13.8 | 6.2 | 6.2 | 6.5 |
| HVT-RAFT | 10.4 | 3.0 | 3.7 | 5.2 |
| RAFT-Stereo | 12.6 | 3.3 | 4.7 | 5.5 |
| IGEV | 8.8 | 4.0 | 5.2 | 5.7 |
| IGEV++ | 7.8 | 4.1 | 5.1 | 5.9 |
| NMRF | 7.5 | 3.8 | 4.2 | 5.1 |
| Ours (Scene Flow only) | 5.5 | 1.8 | 3.2 | 4.9 |
| Selective-IGEV* | 7.5 | 3.4 | 3.2 | 4.5 |
| **Ours** | **1.1** | **0.5** | **2.3** | **2.8** |

**关键发现**: 即使仅在 SceneFlow 上训练，FoundationStereo 也全面超越对比方法。完整训练后 Middlebury BP-2 从 7.5 降至 1.1，ETH3D BP-1 从 3.4 降至 0.5，提升巨大。

### Table 3: In-the-Wild 定量对比

| Method | LEAStereo | GANet | ACVNet | IGEV-Stereo | NMRF | MoCha-Stereo | Selective-IGEV | **Ours** |
|--------|-----------|-------|--------|-------------|------|-------------|----------------|---------|
| EPE | 0.78 | 0.84 | 0.48 | 0.47 | 0.45 | 0.41 | 0.44 | **0.34** |

**说明**: 在真实场景数据上 EPE 0.34，超越次优方法 MoCha-Stereo 的 0.41，提升 17%。

### Table 4: ETH3D 排行榜

| Method | Zero-Shot | BP-0.5 | BP-1.0 | EPE |
|--------|-----------|--------|--------|-----|
| GMStereo | x | 5.94 | 1.83 | 0.19 |
| HITNet | x | 7.83 | 2.79 | 0.20 |
| RAFT-Stereo | x | 7.04 | 2.44 | 0.18 |
| CREStereo | x | 3.58 | 0.98 | 0.13 |
| IGEV-Stereo | x | 3.52 | 1.12 | 0.14 |
| CroCo-Stereo | x | 3.27 | 0.99 | 0.14 |
| Selective-IGEV | x | 3.06 | 1.23 | 0.12 |
| **Ours (finetuned)** | **x** | **1.26** | **0.26** | **0.09** |
| **Ours** | **v** | **2.31** | **1.52** | **0.13** |

**关键发现**: 微调后 ETH3D BP-0.5 从 3.06 降至 1.26，超越次优方法一半以上误差率。零样本版本 EPE 0.13 已与微调的 CREStereo 持平。

### Table 5: STA 消融

| Row | Variations | BP-2 |
|-----|-----------|------|
| 1 | DINOv2-L | 2.46 |
| 2 | DepthAnythingV2-S | 2.22 |
| 3 | DepthAnythingV2-B | 2.11 |
| 4 | **DepthAnythingV2-L** | **1.97** |
| 5 | STA (a) — DPT head | 6.48 |
| 6 | STA (b) — ViT-Adapter | 2.22 |
| 7 | **STA (c) — 4x4 conv** | **1.97** |
| 8 | Unfreeze ViT | 3.94 |
| 9 | **Freeze ViT** | **1.97** |

**关键发现**: DepthAnythingV2 优于 DINOv2（任务相关性更强）；设计 (c) 大幅优于 (a)(b)；冻结 ViT 至关重要（解冻会破坏预训练先验）。

### Table 6: AHCF 消融

| Row | Variations (DT) | BP-2 | Row | Variations (APC) | BP-2 |
|-----|-----------------|------|-----|-----------------|------|
| 1 | RoPE | 2.19 | 10 | (3,3,1), (1,1,5) | 2.10 |
| 2 | **Cosine** | **1.97** | 11 | (3,3,1), (1,1,9) | 2.06 |
| 3 | 1/32 | 2.06 | 12 | (3,3,1), (1,1,13) | 2.01 |
| 4 | **1/16** | **1.97** | 13 | **(3,3,1), (1,1,17)** | **1.97** |
| 5 | Full volume | 2.25 | 14 | (3,3,1), (1,1,21) | 1.98 |
| 6 | **Disparity only** | **1.97** | 15 | (7,7,1), (1,1,17) | 1.99 |
| 7 | Pre-hourglass | 2.06 | | | |
| 8 | Post-hourglass | 2.20 | | | |
| 9 | **Parallel** | **1.97** | | | |

**关键发现**: Cosine 优于 RoPE（视差维度大小恒定）；仅在视差维度做注意力优于全 volume（降低复杂度且更有效）；APC 视差核大小 17 达到饱和。

### Table 7: 模块效果与 FSD 效果消融

| Row | STA | APC | DT | BP-2 |
|-----|-----|-----|----|------|
| 1 | | | | 2.48 |
| 2 | | v | | 2.21 |
| 3 | | v | v | 2.16 |
| 4 | v | | v | 2.05 |
| 5 | **v** | **v** | **v** | **1.97** |

| Row | FSD | BP-2 |
|-----|-----|------|
| 1 | x | 2.34 |
| 2 | **v** | **1.15** |

**关键发现**: STA、APC、DT 各自都有贡献；FSD 数据集贡献最大，从 BP-2 2.34 降至 1.15，降幅 50%+。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| **FoundationStereo Dataset (FSD)** | 1M 对 | NVIDIA Omniverse，47 场景，变化基线/内参 | 主训练 |
| [[SceneFlow]] | 35K | 合成多场景 | 混合训练 |
| Sintel | 1K | 电影级合成 | 混合训练 |
| [[CREStereo]] 数据 | 200K | Blender 合成 | 混合训练 |
| FallingThings | 62K | 散落物体 | 混合训练 |
| InStereo2K | 2K | 室内结构光真值 | 混合训练 |
| Virtual KITTI 2 | — | 合成驾驶 | 混合训练 |
| [[Middlebury]] | 23 对 | 高分辨率室内 | 评测（第一名） |
| [[ETH3D]] | 47 对 | 灰度高分辨率 | 评测（第一名） |
| [[KITTI-2012]] / [[KITTI-2015]] | 200 对 | 驾驶场景 | 评测 |

### 实现细节

- **框架**: PyTorch
- **优化器**: [[AdamW]]，学习率 1e-4，在 0.8 处衰减 0.1
- **Batch Size**: 128（分布于 32 x NVIDIA A100 GPU）
- **训练步数**: 200K iterations
- **训练分辨率**: 320x736 随机裁剪
- **GRU 迭代次数**: 训练 22 次 / 推理 32 次
- **最大视差**: 416
- **数据增强**: 类似 RAFT-Stereo
- **Feature Backbone**: EdgeNeXt-S（STA 中的 CNN）
- **DepthAnythingV2**: ViT-S（冻结），图像 resize 为 14 的倍数
- **推理时间**: 375x1242 分辨率约 0.7s（A100）

### 可视化结果

- 在 in-the-wild 场景中对反射、半透明、重复纹理、细结构均鲁棒
- STA 有效处理光照不一致区域（暗色吉他音孔、灯罩反射）
- AHCF 恢复细结构（栏杆、薄物体边界）
- 零样本推理已超越或匹配现有微调方法

---

## 批判性思考

### 优点
1. **真正的零样本基础模型**: 首次在立体匹配领域实现无需微调即超越微调方法的性能
2. **系统性设计**: 数据（FSD + 自筛选）+ 架构（STA + AHCF）双管齐下，每个组件都有清晰的消融验证
3. **STA 设计精巧**: 冻结 ViT + 侧接 CNN 的简单方案大幅优于更复杂的 adapter 方法
4. **APC 和 DT 解耦空间/视差推理**: 在不增加过多计算的情况下显著扩展视差维度的感受野
5. **排行榜双冠 + 零样本超越微调**: ETH3D 和 Middlebury 均第一，且零样本已超越先前微调方法

### 局限性
1. **推理速度慢**: 375x1242 分辨率需 0.7s（A100），无法实时，也远慢于 [[HITNet]] (0.02s) 和 [[RAFT-Stereo]] (0.13s)
2. **计算资源需求高**: 训练需 32 x A100，数据集渲染需 NVIDIA Omniverse 商业工具，可复现性受限
3. **FSD 数据集未公开**: 论文发表时数据集渲染工具和具体配置未完全开源
4. **透明物体覆盖有限**: 论文承认 FSD 中透明物体多样性不足

### 潜在改进方向
1. 模型蒸馏/剪枝实现轻量化实时推理
2. 扩展 FSD 覆盖更多透明/镜面物体
3. 结合时序信息实现视频立体匹配
4. 探索更大 ViT backbone（DepthAnythingV2-L）在资源允许时的效果

### 可复现性评估
- [x] 代码开源
- [x] 预训练模型
- [x] 训练细节完整
- [ ] FSD 数据集未完全公开（依赖 Omniverse）

---

## 关联笔记

### 基于
- [[RAFT-Stereo]]: 迭代精化范式的直接前身
- [[DepthAnythingV2]]: STA 中冻结的单目深度基础模型
- [[IGEV-Stereo]]: AHCF 改进的 cost volume 滤波基础
- [[GwcNet]]: Group-wise correlation 的来源

### 对比
- [[CREStereo]]: 先前 Middlebury/ETH3D 冠军，被 FoundationStereo 大幅超越
- [[RAFT-Stereo]]: 迭代精化方法，FoundationStereo 在零样本下超越其微调结果
- [[Selective-IGEV]]: 先前零样本 SOTA，被全面超越
- [[CroCo v2]]: 基于 cross-view completion 预训练的方法
- [[NMRF]]: Neural Markov Random Field 方法

### 方法相关
- [[Correlation Volume]]: Hybrid Cost Volume 的基础
- [[GRU]]: 迭代精化的核心组件
- [[Convex Upsampling]]: 最终上采样方法
- [[FlashAttention]]: Disparity Transformer 中使用
- [[Separable Convolution]]: APC 的设计灵感
- [[Soft-Argmin]]: 初始视差预测
- [[EdgeNeXt]]: STA 中的 CNN backbone
- [[Side-Tuning]]: 适配基础模型的策略

### 硬件/数据相关
- [[SceneFlow]]: 混合训练数据集
- [[Middlebury]]: 主要评测 benchmark（第一名）
- [[ETH3D]]: 主要评测 benchmark（第一名）
- [[KITTI-2015]]: 评测 benchmark
- [[NVIDIA Omniverse]]: FSD 渲染工具

---

## 速查卡片

> [!summary] FoundationStereo: Zero-Shot Stereo Matching
> - **核心**: 首个立体匹配基础模型，零样本超越微调方法
> - **方法**: Side-Tuning Adapter (DepthAnythingV2 + EdgeNeXt-S) + AHCF (APC + Disparity Transformer) + 1M FSD 数据集
> - **结果**: Middlebury 第一 (BP-2: 1.26 微调 / 1.1 零样本)，ETH3D 第一 (BP-0.5: 1.26 / BP-1: 0.26)
> - **代码**: https://github.com/NVlabs/FoundationStereo

---

*笔记创建时间: 2026-05-12*
