---
title: "Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization"
method_name: "Reloc3r"
authors: [Siyan Dong, Shuzhe Wang, Shaohui Liu, Lulu Cai, Qingnan Fan, Juho Kannala, Yanchao Yang]
year: 2025
venue: CVPR 2025
tags: [visual-localization, camera-pose-regression, relative-pose-estimation, motion-averaging, visual-place-recognition]
image_source: online
arxiv_html: https://arxiv.org/html/2412.08376v1
created: 2026-05-13
---

# 论文笔记：Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The University of Hong Kong; Aalto University; ETH Zurich; VIVO; University of Oulu |
| 日期 | December 2024 |
| 项目主页 | - |
| 对比基线 | [[DUSt3R]], [[MASt3R]], [[RoMa]], [[Map-free Relocalization]], [[ExReNet]] |
| 链接 | [arXiv](https://arxiv.org/abs/2412.08376) / [Code](https://github.com/ffrivera0/reloc3r) |

---

## 一句话总结

> 基于 DUSt3R backbone 的全对称相对位姿回归网络，配合简约 motion averaging 模块，在约 800 万图像对上大规模训练实现实时、可泛化、高精度的视觉定位。

---

## 核心贡献

1. **全对称 RPR 网络**: 基于 [[DUSt3R]] 的 [[Vision Transformer|ViT]] backbone 设计全对称架构（共享 decoder 和回归头），参数减少 28% 的同时提升精度
2. **无度量尺度训练策略**: 仅学习平移方向而非度量平移，消除旋转/平移损失加权的问题，通过 [[Motion Averaging]] 恢复度量尺度
3. **大规模训练带来泛化**: 在 7 个数据集约 800 万图像对上训练，首次在未见场景上超越场景专用的 APR 方法，实时推理 24 FPS

---

## 问题背景

### 要解决的问题
视觉定位 (Visual Localization)：给定查询图像和已知位姿的数据库图像集，估计查询图像的 6-DoF 相机位姿。

### 现有方法的局限
- **基于结构的方法** (如 [[HLoc]]): 需要构建 3D 模型，推理时匹配耗时，大规模场景效率低
- **绝对位姿回归 (APR)**: 场景专用、需要密集视角覆盖、泛化差，本质上更像图像检索的位姿近似
- **相对位姿回归 (RPR)**: 具有跨场景泛化潜力，但现有方法精度远落后于 APR，且未充分利用大规模训练
- 已有 RPR 方法重视技术设计而忽视数据规模的扩展

### 本文的动机
受基础模型（如 [[DINOv2]], [[DUSt3R]]）通过可扩展架构和大规模训练取得突破的启发，作者认为 RPR 也可以通过"简约设计 + 大规模训练"实现质的飞跃。

---

## 方法详解

### 模型架构

Reloc3r 采用 **全对称 [[Vision Transformer|ViT]]** 架构：
- **输入**: 图像对 $(I_1, I_2)$，通过 [[NetVLAD]] 检索 top-K 数据库图像
- **Backbone**: [[DUSt3R]] 的 ViT-Large encoder（冻结）+ ViT-Base decoder（可训练）
- **核心模块**: 共享权重的全对称 encoder-decoder + 位姿回归头
- **输出**: 双向相对位姿 $\hat{P}_{I_1,I_2}$ 和 $\hat{P}_{I_2,I_1}$
- **总参数**: ~0.42B

### 核心模块

#### 模块1: ViT Encoder-Decoder

**设计动机**: 利用 [[DUSt3R]] 预训练的强大几何理解能力

**具体实现**:
- **Encoder**: m=24 个 ViT 块，使用 [[RoPE]] 位置编码
  - 每张图像 $I_i$ 分割为 $T$ 个 token（维度 $d$）
  - $F_i^{T \times d} = \text{Encoder}(\text{Patchify}(I_i^{H \times W \times 3}))$
- **Decoder**: n=12 个 ViT 块，含 [[Cross-Attention]] 层实现信息交换
  - $G_1 = \text{Decoder}(F_1, F_2)$, $G_2 = \text{Decoder}(F_2, F_1)$
- **全对称设计**: 两个分支共享 decoder 和回归头权重
- Encoder 冻结，仅更新 decoder 权重
- Memory-efficient attention 提升 ~14% 速度，节省 25% 显存

#### 模块2: 位姿回归头 (Pose Regression Head)

**设计动机**: 从 decoder 输出特征回归 6-DoF 相对位姿

**具体实现**:
- h=2 个卷积层 + average pooling
- 旋转用 [[9D Rotation Representation|9D 表示]]，通过 SVD 正交化映射到 3×3 旋转矩阵
- 平移仅回归方向（单位向量），不学习度量尺度
- 双分支输出: $\hat{P}_{I_1,I_2} = \text{Head}(G_1)$, $\hat{P}_{I_2,I_1} = \text{Head}(G_2)$

#### 模块3: Motion Averaging

**设计动机**: 从多个噪声相对位姿估计恢复绝对位姿和度量尺度

**具体实现**:
- **旋转平均**: 绝对旋转 $\hat{R}_q = R_{d_i} \cdot \hat{R}_{q,d_i}$，对多个估计取中位数（四元数表示）
- **相机中心三角化**: 从两个数据库-查询对的平移方向通过最小二乘法三角化绝对相机中心
  - 最小化相机中心到各平移方向线段的距离平方和
  - 通过 SVD 求解
- 无参数设计，不引入可训练参数

---

## 关键公式

### 公式1: [[Relative Pose Regression|相对位姿回归]]

$$
\hat{P}_{I_1,I_2}^{3 \times 4} = \text{Head}\bigl(\text{Decoder}(\text{Encoder}(I_1), \text{Encoder}(I_2))\bigr)
$$

**含义**: 从图像对端到端回归相对位姿

### 公式2: [[Rotation Loss|旋转损失]]

$$
\ell_R = \arccos\frac{\text{tr}(\hat{R}^{-1} R) - 1}{2}
$$

**含义**: 预测旋转与 GT 旋转之间的测地距离（角度差）

**符号说明**:
- $\hat{R}$: 预测旋转矩阵
- $R$: GT 旋转矩阵
- $\text{tr}(\cdot)$: 矩阵的迹

### 公式3: [[Translation Loss|平移损失]]

$$
\ell_t = \arccos\frac{\hat{t} \cdot t}{\|\hat{t}\| \|t\|}
$$

**含义**: 预测平移方向与 GT 平移方向的夹角。仅约束方向，不约束度量尺度。

**符号说明**:
- $\hat{t}$: 预测平移向量
- $t$: GT 平移向量

### 公式4: [[Pose Regression|总训练损失]]

$$
\mathcal{L} = \ell_R + \ell_t
$$

**含义**: 旋转和平移损失直接相加，无需权重系数。因为两者都以角度为单位，天然平衡。

### 公式5: [[Rotation Averaging|绝对旋转恢复]]

$$
\hat{R}_q = R_{d_i} \cdot \hat{R}_{q,d_i}
$$

**含义**: 从已知数据库图像旋转和估计的相对旋转计算查询图像的绝对旋转，多个估计取中位数。

### 公式6: [[Camera Center Triangulation|相机中心三角化]]

通过最小化到多条平移方向射线的距离平方和，用 SVD 求解查询相机中心 $c_q$:

$$
c_q = \arg\min_c \sum_{k=1}^K \bigl\|(I - \hat{d}_k \hat{d}_k^\top)(c - c_{d_k})\bigr\|^2
$$

**含义**: 从多个数据库-查询相对平移方向三角化查询相机的绝对位置

**符号说明**:
- $c_{d_k}$: 第 $k$ 个数据库相机中心
- $\hat{d}_k$: 从 $c_{d_k}$ 到查询相机的方向向量
- $I$: 单位矩阵

---

## 关键图表

### Figure 1: Accuracy-Efficiency Comparison / 精度-效率对比

![Figure 1](https://arxiv.org/html/2412.08376v1/x1.png)

**说明**: ScanNet1500 数据集上的位姿精度 (AUC@5) vs 运行时效率 (FPS) 对比。Reloc3r-512 在达到最高 AUC@5 的同时保持 24 FPS。Reloc3r-224 与 [[RoMa]] 精度相当但快 20 倍。

### Figure 2: Architecture Overview / 架构概览

![Figure 2](https://arxiv.org/html/2412.08376v1/extracted/6061956/fig/overview.png)

**说明**: Reloc3r 架构概览。图像对经 patchify 后输入共享权重的 ViT encoder，decoder 通过 [[Cross-Attention]] 交换信息，回归头预测双向相对位姿。[[Motion Averaging]] 模块从多个相对估计计算绝对度量位姿。

### Figure 3: Pose Visualization / 位姿可视化

![Figure 3](https://arxiv.org/html/2412.08376v1/extracted/6061956/fig/visual.png)

**说明**: 7 Scenes (Chess) 和 Cambridge Landmarks (KingsCollege) 上的位姿估计可视化。与 [[ExReNet]] 和 [[Map-free Relocalization]] 对比，Reloc3r 估计值更接近 GT。

### Figure 4: Cross-Attention Visualization / 交叉注意力可视化

![Figure 4](https://arxiv.org/html/2412.08376v1/extracted/6061956/fig/match.png)

**说明**: Reloc3r decoder 中 [[Cross-Attention]] 层学到的对应关系。上行为 [[Efficient LoFTR]] 匹配结果，下行为 Reloc3r 的 top-3 cross-attention 响应。尽管仅用位姿监督训练，Reloc3r 的相关区域质量优于专用匹配方法。

### Table 1: 训练数据

| Dataset | Scene Type | # Image Pairs |
|---------|-----------|---------------|
| CO3Dv2 | Object-centric | ~1M |
| ScanNet++ | Indoor | ~850K |
| ARKitScenes | Indoor | ~2.1M |
| BlendedMVS | Outdoor | ~1M |
| MegaDepth | Outdoor | ~1.8M |
| DL3DV | Indoor & outdoor | ~1.1M |
| RealEstate10K | Indoor & outdoor | ~100K |
| **Total** | **Mixed** | **~8M** |

**说明**: 首个在物体中心、室内和室外数据集混合训练的位姿回归方法。

### Table 2: CO3Dv2 多视图相对位姿

| Method | Category | RRA@15 | RTA@15 | mAA@30 |
|--------|----------|--------|--------|--------|
| DUSt3R (w/ PnP) | Non-PR | 94.3 | 88.4 | 77.2 |
| MASt3R | Non-PR | 94.6 | 91.9 | 81.1 |
| **Reloc3r-512** | **PR** | **95.8** | **93.7** | **82.9** |

**说明**: 作为位姿回归方法，Reloc3r-512 超越了非回归方法 [[MASt3R]] 和 [[DUSt3R]]。

### Table 3: ScanNet1500, RealEstate10K, ACID 成对相对位姿

| Method | Cat. | SN AUC@5/10/20 | RE10K AUC@5/10/20 | ACID AUC@5/10/20 |
|--------|------|-----------------|---------------------|---------------------|
| RoMa | Non-PR | 28.9/50.4/68.3 | 54.6/69.8/79.7 | 46.3/58.8/68.9 |
| MASt3R | Non-PR | 28.0/50.2/68.8 | 63.5/76.4/84.5 | 52.1/64.5/73.6 |
| **Reloc3r-512** | **PR** | **34.8/58.4/75.6** | **66.7/80.2/88.4** | **38.2/56.4/70.3** |

**说明**: 在 ScanNet1500 上超越所有方法（含非回归）。ACID 为训练中未见数据集，AUC@20=70.3 展示强泛化能力。推理仅需 42ms。

### Table 4: 7 Scenes 视觉定位（中位误差 m/deg）

| Category | Method | Average |
|----------|--------|---------|
| APR | PMNet | 0.06/1.93 |
| APR | DFNet+NeFeS | 0.02/0.79 |
| RPR (Unseen) | ExReNet | 0.08/2.52 |
| RPR (Unseen) | Map-free | 0.13/3.72 |
| **RPR (Unseen)** | **Reloc3r-224** | **0.05/1.26** |
| **RPR (Unseen)** | **Reloc3r-512** | **0.04/1.02** |

**说明**: Reloc3r-512 在未见场景上平均 0.04m/1.02deg，超越场景专用 APR 方法 PMNet (0.06/1.93)，接近需要新视角合成数据增强的 DFNet+NeFeS。

### Table 5: Cambridge Landmarks 视觉定位

| Category | Method | Avg(4) | Avg |
|----------|--------|--------|-----|
| APR | PMNet | 0.31/0.79 | - |
| RPR (Unseen) | ExReNet | 2.22/3.03 | 3.74/3.31 |
| **RPR (Unseen)** | **Reloc3r-512** | **0.38/0.52** | **0.55/0.56** |

**说明**: 较前一代最佳 RPR 方法误差减半，旋转精度超越所有 APR 方法。

### Table 6: 消融实验 (ScanNet1500)

| Method | AUC@5 | AUC@10 | AUC@20 |
|--------|-------|--------|--------|
| Reloc3r-512 asymmetric | 32.71 | 56.84 | 74.63 |
| Reloc3r-512 metric pose | 25.70 | 50.20 | 70.07 |
| **Reloc3r-512 (default)** | **34.79** | **58.37** | **75.56** |

**关键发现**: (1) 全对称设计优于非对称设计，同时减少 28% 参数；(2) 仅学习方向而非度量平移效果更好，验证了 motion averaging 恢复度量尺度的策略。

### Table 7: 权重初始化消融

| Initialization | AUC@5 | AUC@10 | AUC@20 |
|---------------|-------|--------|--------|
| No init. (512) | 3.98 | 15.58 | 37.02 |
| CroCo v2 (full) | 22.44 | 47.62 | 68.65 |
| MASt3R (full) | 32.62 | 56.28 | 74.32 |
| **DUSt3R-512 (full)** | **34.79** | **58.37** | **75.56** |

**关键发现**: DUSt3R 预训练初始化最优，随机初始化性能骤降至 ~4 AUC@5。

### Table 8: Cambridge Landmarks 详细结果（含结构化方法）

| Method | Avg | Time |
|--------|-----|------|
| HLoc (SP+SG) | 0.07/0.14 | 737ms |
| DUSt3R-512 (top-20) | 0.16/0.24 | >3000ms |
| Reloc3r-512 top-10 | 0.55/0.56 | 235ms |

**说明**: 在大规模室外场景中，Reloc3r 仍不及基于特征匹配+几何优化的 HLoc，但速度快 3 倍以上。

### Table 9: MegaDepth1500 相对位姿

| Method | Category | AUC@5/10/20 |
|--------|----------|-------------|
| RoMa | Non-PR | 62.6/76.7/86.3 |
| MASt3R | Non-PR | 42.4/61.5/76.9 |
| **Reloc3r-512** | **PR** | **49.6/67.9/81.2** |

**说明**: MegaDepth 含显著内参变化，是位姿回归的重大挑战。Reloc3r 大幅超越 MASt3R 但不及基于匹配的 RoMa。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| CO3Dv2, ScanNet++, ARKitScenes 等 | ~8M pairs | 混合场景 | 训练 |
| ScanNet1500 | 真实室内 | 标准基准 | 相对位姿评估 |
| RealEstate10K | 真实室内外 | 视频帧对 | 相对位姿评估 |
| ACID | 真实 | 未见场景 | 泛化测试 |
| 7 Scenes | 真实室内 | 小场景 | 视觉定位评估 |
| Cambridge Landmarks | 真实室外 | 大场景 | 视觉定位评估 |
| MegaDepth1500 | 真实室外 | 大内参变化 | 相对位姿评估 |

### 实现细节

- **Backbone**: ViT-Large encoder (m=24 blocks) + ViT-Base decoder (n=12 blocks)
- **回归头**: h=2 卷积层 + average pooling
- **优化器**: 学习率 1e-5 → 1e-7（渐降）
- **Batch Size**: 8
- **训练**: 8× AMD MI250x-40G GPU
- **推理**: NVIDIA RTX 4090，Reloc3r-512 @24 FPS，fp16 @40 FPS
- **图像检索**: [[NetVLAD]]，top-10
- **Encoder 冻结**: 仅更新 decoder 和回归头

### 可视化结果

- Cross-attention 响应图显示网络在仅有位姿监督的情况下自动学到了语义对应关系
- 在绘画、草图、不同人脸之间也能合理估计相对位姿，展示强泛化能力

---

## 批判性思考

### 优点
1. **设计简约高效**: 全对称架构减少 28% 参数，无参数 motion averaging 无额外学习开销
2. **大规模训练策略**: 首个在多领域混合数据上大规模训练的 RPR 方法，泛化能力质变
3. **速度优势明显**: 42ms (512 res) / 15ms (224 res)，比 RoMa (300ms) 快 7-20 倍
4. **巧妙的损失设计**: 仅学习平移方向，天然平衡旋转/平移损失，无需权重调优

### 局限性
1. 所有查询图像和数据库图像共线时，度量尺度无法求解（motion averaging 退化）
2. 大内参变化（如大倍率变焦）时平移方向估计不准
3. 在大规模室外场景 (Cambridge GreatCourt) 上仍不及基于结构的方法
4. 依赖图像检索 ([[NetVLAD]]) 的质量

### 潜在改进方向
1. 扩大训练数据规模和多样性
2. 探索替代 motion averaging 的更鲁棒绝对位姿恢复策略
3. 与稀疏特征匹配方法融合提升大场景精度

### 可复现性评估
- [x] 代码开源（github.com/ffrivera0/reloc3r）
- [x] 预训练模型
- [x] 训练细节完整
- [x] 数据集可获取（均为公开数据集）

---

## 关联笔记

### 基于
- [[DUSt3R]]: Transformer backbone 和预训练权重来源
- [[CroCo]]: 自监督预训练（DUSt3R 的基础）
- [[NetVLAD]]: 图像检索模块

### 对比
- [[RoMa]]: 非回归匹配方法，精度相当但 Reloc3r 快 7-20 倍
- [[MASt3R]]: 非回归方法，Reloc3r 在多数基准上超越
- [[ExReNet]]: 此前最佳 RPR 方法
- [[Map-free Relocalization]]: 零样本 RPR 方法
- [[HLoc]]: 基于结构的定位方法，大场景精度更高但慢得多

### 方法相关
- [[Motion Averaging]]: 从相对位姿恢复绝对位姿的核心模块
- [[9D Rotation Representation]]: 旋转矩阵的连续表示
- [[Cross-Attention]]: decoder 中实现双图信息交换
- [[RoPE]]: 旋转位置编码
- [[PnP]]: DUSt3R 的位姿恢复方式（Reloc3r 用回归替代）
- [[Visual Place Recognition]]: 图像检索阶段

### 硬件/数据相关
- [[ScanNet]]: 主要评估数据集
- [[MegaDepth]]: 训练和评估数据集

---

## 速查卡片

> [!summary] Reloc3r: Large-Scale Training of Relative Camera Pose Regression
> - **核心**: 全对称 ViT 回归网络 + 无参数 motion averaging，大规模训练实现泛化
> - **方法**: DUSt3R backbone + 仅学平移方向 + 旋转平均/相机中心三角化
> - **结果**: CVPR 2025；7 Scenes 0.04m/1.02deg；ScanNet AUC@5=34.79；24 FPS
> - **代码**: https://github.com/ffrivera0/reloc3r

---

*笔记创建时间: 2026-05-13*
