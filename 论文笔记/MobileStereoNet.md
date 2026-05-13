---
title: "MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching"
method_name: "MobileStereoNet"
authors: [Faranak Shamsafar, Samuel Woerz, Rafia Rahim, Andreas Zell]
year: 2022
venue: WACV 2022
tags: [stereo-matching, lightweight-network, mobilenet, cost-volume, edge-deployment, depth-estimation]
image_source: mixed
arxiv_html: https://arxiv.org/html/2108.09770
created: 2026-05-12
---

# 论文笔记：MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Tuebingen, WSI Institute for Computer Science |
| 日期 | August 2021 |
| 项目主页 | — |
| 对比基线 | [[GCNet]], [[PSMNet]], [[GANet]], [[GwcNet]], [[DeepPruner]] |
| 链接 | [arXiv](https://arxiv.org/abs/2108.09770) / [Code](https://github.com/cogsys-tuebingen/mobilestereonet) |

---

## 一句话总结

> 将 MobileNet 的轻量化模块扩展到 3D 立体匹配，设计 2D/3D 两种轻量模型，并引入可学习的 Interlacing Cost Volume 提升 2D 模型精度。

---

## 核心贡献

1. **MobileNet Blocks 3D 扩展**: 将 [[MobileNetV1]] 的 [[Depthwise Separable Convolution|深度可分离卷积]] 和 [[MobileNetV2]] 的 [[Inverted Residual Block]] 从 2D 扩展到 3D，用于立体匹配 cost volume 处理
2. **Interlacing Cost Volume**: 提出新的可学习 3D cost volume 构建方法，通过交错左右特征并用 3D 卷积蒸馏，使 2D encoder-decoder 达到接近 3D 网络的精度
3. **2D/3D 双模型设计**: 分别设计 MSNet2D（27%/95% 更少参数/运算）和 MSNet3D（72%/38% 更少参数/运算），面向不同部署需求

---

## 问题背景

### 要解决的问题
深度学习立体匹配模型精度持续提升但计算量急剧增加，导致模型无法在中等 GPU 上运行，更无法部署到嵌入式设备（FPGA、移动端）。

### 现有方法的局限
- 3D 模型（[[GCNet]], [[PSMNet]], [[GANet]]）使用 3D 卷积处理 4D cost volume，计算量大、内存需求高
- 部分方法在中等 GPU 上甚至会 OOM
- [[DeepPruner]] 虽减少运算但参数量仍过大，在小数据集微调时容易过拟合

### 本文的动机
[[MobileNetV1]] 和 [[MobileNetV2]] 在图像分类中成功用轻量化模块替代标准卷积，显著降低计算量。将这些模块扩展到 3D 并应用于立体匹配的 encoder-decoder 部分，可以在保持精度的同时大幅降低计算和参数开销。

---

## 方法详解

### 模型架构

MobileStereoNet 提供两种变体：

**MSNet2D（2D 模型）**:
- **Feature Extraction**: 共享 ResNet-like backbone → 320 通道 → 4 次 pointwise 降通道至 32
- **Cost Volume**: **Interlacing Cost Volume** (3D, 可学习) → $d_{max}/4 \times H/4 \times W/4$
- **Encoder-Decoder**: Pre-hourglass (2D $v_2$ blocks, $t=3$) + 3 个 Hourglass (2D $v_2$ blocks, $t=2$, 通道宽度 48)
- **输出**: 上采样后的视差图，[[Smooth L1 Loss]] 监督

**MSNet3D（3D 模型）**:
- **Feature Extraction**: 同上
- **Cost Volume**: Group-wise Correlation (4D) → $40 \times d_{max}/4 \times H/4 \times W/4$
- **Encoder-Decoder**: Pre-hourglass (3D $v_2$ blocks, $t=3$) + 3 个 Hourglass (3D $v_2$ blocks, $t=2$)
- **输出**: 同上

### 核心模块

#### 模块1: 2D/3D MobileNet Blocks

**设计动机**: 标准 3D 卷积计算量为 $k^3 \times k \times k \times C_{in} \times C_{out}$，用 [[Depthwise Separable Convolution|深度可分离卷积]] 可降低一个数量级。

**具体实现**:
- **$v_1$ Block (MobileNetV1 风格)**: depthwise conv + pointwise conv，2D 版降低 7.9x/18.9x（2D/3D），但无残差连接
- **$v_2$ Block (MobileNetV2 风格)**: pointwise expansion ($t$) → depthwise conv → pointwise projection + 残差连接，2D 版降低 2.7x/7.0x（2D/3D）
- 将 2D 的 $C \times H \times W$ 扩展为 3D 的 $C \times D \times H \times W$，depthwise 和 pointwise 卷积核相应从 $k \times k$ 变为 $k \times k \times k$

#### 模块2: Interlacing Cost Volume

**设计动机**: 传统 cost volume（相关/拼接）是无参数的，无法学习最优的左右特征聚合方式。通过交错 (interlace) 左右特征对并用可学习的 3D 卷积处理，可提升 2D 模型精度至接近 3D 模型。

**具体实现**:
- 给定左特征 $f_L(.,x,y)$ 和平移后右特征 $f_R(.,x-d,y)$，各 $C \times H \times W$
- 沿通道维交错拼接为 $2C \times H \times W$
- Unsqueeze 为 4D: $1 \times 2C \times H \times W$
- 第一层 3D 卷积核 $2i \times 3 \times 3$（stride $i \times 1 \times 1$），每次覆盖 $i$ 个左右特征对（$i=4$ 最优，即 group-wise interlacing）
- 后续两层 3D 卷积进一步蒸馏 → 最终输出 3D cost volume
- 最后一层用 2D 卷积转为单通道

---

## 关键公式

### 公式1: [[Correlation Volume|传统 3D Cost Volume]]

$$
C_{3D}(d, x, y) = G(f_L(x, y), f_R(x - d, y))
$$

**含义**: 对每个像素 $(x,y)$ 和视差 $d$，通过相似度函数 $G$ 计算左右特征的匹配代价

**符号说明**:
- $(x, y)$: 空间位置
- $d \in (0, d_{max})$: 视差候选值
- $f_R(x-d, y)$: 右图在对应视差位置的特征
- $G(\cdot, \cdot)$: 相似度度量（相关、Hamming 距离等）

### 公式2: [[Correlation Volume|Interlacing Cost Volume]]

$$
C_{3D}(d, x, y) = \text{Interlace}\{f_L(., x, y), f_R(., x-d, y)\}
$$

**含义**: 通过交错左右特征并用可学习的 3D 卷积蒸馏，生成参数化的 3D cost volume

**符号说明**:
- $\text{Interlace}\{\cdot\}$: 交错操作 + 3D 卷积蒸馏
- $f_L(., x, y)$: 左图在 $(x,y)$ 的全通道特征向量
- $f_R(., x-d, y)$: 右图在视差 $d$ 位置的全通道特征向量

---

## 关键图表

### Figure 1: Performance vs. Computation / 性能-计算量权衡

![[MobileStereoNet_fig1.png|600]]

**说明**: SceneFlow（左）和 KITTI 2015（右）上 EPE vs 参数量/MACs 的散点图。2D-MobileStereoNet 在最少 MACs 下达到接近 3D 模型的精度；3D-MobileStereoNet 在最少参数量下取得竞争性精度。

### Figure 2: MobileNet Blocks / MobileNet 模块 2D→3D 扩展

![[MobileStereoNet_fig2.png|600]]

**说明**: 左：MobileNetV1 ($v_1$) block 及其 3D 扩展。右：MobileNetV2 ($v_2$, Inverted Residual) block 及其 3D 扩展。核心是将 depthwise 和 pointwise 卷积核从 2D 升维到 3D。

### Figure 3: Reduction Factor / v2 Block 的计算量降低因子

![[MobileStereoNet_fig3.png|600]]

**说明**: 不同扩展因子 $t$ 下 2D/3D $v_2$ block 相对标准卷积的计算量降低倍数。$t$ 增大时降低因子减小，2D block 对 $t$ 更敏感。通道数 $C_{in}=C_{out}=\{32,64,128\}$, $k=3$。

### Figure 4: Network Architectures / 2D 和 3D 基线网络架构

![[MobileStereoNet_fig4.png|600]]

**说明**: 上：2D 基线，使用 Interlacing Cost Volume + 2D encoder-decoder。下：3D 基线（GwcNet-g），使用 Group-wise Correlation + 3D encoder-decoder。MobileStereoNet 通过替换基线中的标准卷积为 MobileNet blocks 得到。

### Figure 5: Interlacing Cost Volume / 交错 Cost Volume 构建

![[MobileStereoNet_fig5.png|600]]

**说明**: 左：在特定视差 $d$ 下交错左（红）右（蓝）特征的过程。第一层 3D 卷积核覆盖 $i$ 个通道对（$Interlaced_4$）。右：后续多层 3D 卷积逐步聚合信息，最终输出该视差层的聚合特征。

### Figure 6: SceneFlow 定性结果

![[MobileStereoNet_fig6.png|600]]

**说明**: SceneFlow 测试集定性结果。左列为输入图像/视差真值，中间两列分别为 2D-MobileStereoNet 和 3D-MobileStereoNet 的预测及误差图。

### Figure 7: KITTI 2015 定性对比

![[MobileStereoNet_fig7.png|600]]

**说明**: KITTI 2015 上与 GCNet、PSMNet、GwcNet-g 的定性对比。3D-MobileStereoNet 因 3D 卷积在 encoder-decoder 中更好保留边缘细节；2D-MobileStereoNet 运算量更少但视觉效果接近。

### Table 1: MobileNet Blocks 计算量分析

| Operator | Operations (MACs) | Example Reduction Factor |
|----------|------------------|-------------------------|
| 2D Std. Conv. | $k \times k \times C_{in} \times C_{out}$ | — |
| 2D $v_1$ Block | $(k \times k \times C_{in}) + (C_{in} \times C_{out})$ | 7.9x |
| 2D $v_2$ Block | $(C_{in} \times t \times C_{in}) + (k \times k \times t \times C_{in}) + (t \times C_{in} \times C_{out})$ | 2.7x |
| 3D Std. Conv. | $k \times k \times k \times C_{in} \times C_{out}$ | — |
| 3D $v_1$ Block | $(k \times k \times k \times C_{in}) + (C_{in} \times C_{out})$ | 18.9x |
| 3D $v_2$ Block | $(C_{in} \times t \times C_{in}) + (k \times k \times k \times t \times C_{in}) + (t \times C_{in} \times C_{out})$ | 7.0x |

**说明**: 对 $k=3, C_{in}=32, C_{out}=64, t=2$ 的计算量分析。3D $v_1$ 降低最多 (18.9x)，但无残差连接；$v_2$ 综合精度和效率最优。

### Table 2: Interlacing Cost Volume 消融

| Method | EPE (px) | D1 (%) | px-3 (%) |
|--------|----------|--------|----------|
| Concatenation | 1.86 | 7.46 | 8.48 |
| Correlation | 1.71 | 6.80 | 7.84 |
| $Interlaced_1$ | 1.70 | 6.20 | **7.06** |
| $Interlaced_2$ | 1.61 | 6.39 | 7.31 |
| **$Interlaced_4$** | **1.55** | **6.15** | **7.06** |
| $Interlaced_8$ | 1.64 | 6.41 | 7.35 |
| $Interlaced_{16}$ | 1.73 | 6.65 | 7.58 |

**说明**: SceneFlow 测试集上不同 cost volume 方法的对比。$Interlaced_4$（每次覆盖 4 个通道对）取得最低 EPE。

### Table 3: 模块替换消融 (a) 2D 模型

| FE | HG | EPE (px) | MACs (G) | Params (M) |
|----|-----|----------|----------|------------|
| conv. | conv. | 1.55 | 74.42 | 4.07 |
| conv. | $v_1$ | 1.62 | 73.97 | 3.52 |
| $v_1$ | conv. | 1.66 | 30.43 | 1.52 |
| $v_1$ | $v_1$ | 1.59 | 29.98 | 0.98 |
| conv. | $v_2$ | 1.63 | 74.32 | 3.75 |
| $v_2$ | conv. | 1.57 | 35.54 | 1.81 |
| $v_2$ | $v_2$ | 1.53 | 35.44 | 1.49 |
| **$v_1$ ; $v_2$** | — | **1.50** | **30.33** | **1.21** |
| $v_2$ ; $v_1$ | — | 1.60 | 35.10 | 1.26 |

### Table 3b: 模块替换消融 (b) 3D 模型

| FE | HG | EPE (px) | MACs (G) | Params (M) |
|----|-----|----------|----------|------------|
| conv. | conv. | 0.97 | 155.2 | 4.21 |
| conv. | $v_1$ | 0.99 | 143.66 | 3.42 |
| $v_1$ | conv. | 0.98 | 111.2 | 1.66 |
| $v_1$ | $v_1$ | 1.03 | 99.67 | 0.87 |
| conv. | $v_2$ | 0.97 | 149.01 | 3.53 |
| $v_2$ | conv. | 0.96 | 116.32 | 1.94 |
| $v_2$ | $v_2$ | 0.97 | 110.13 | 1.27 |
| $v_1$ ; $v_2$ | — | 0.99 | 105.01 | 0.98 |
| **$v_2$ ; $v_1$** | — | **1.02** | **104.78** | **1.15** |

**关键发现**: Feature extraction 占大部分计算量；用 $v_1$ 替换 feature extraction + $v_2$ 替换 hourglass 是 2D 模型最优组合（EPE 1.50）；3D 模型中 $v_2;v_2$ 组合精度最佳（EPE 0.97）。

### Table 4: SceneFlow 测试集最终对比

**(a) 2D 模型:**

| Method | EPE (px) | Params (M) | Reduction |
|--------|----------|------------|-----------|
| DispNet-C | 1.67 | 38 | 17.0x |
| CRL | 1.60 | 78.77 | 35.3x |
| AutoDispNet-C | 1.51 | 37 | 16.6x |
| iResNet | 1.40 | 43.11 | 19.3x |
| **2D-MobileStereoNet** | **1.14** | **2.23** | — |

**(b) 3D 模型:**

| Method | EPE (px) | MACs (G) | Params (M) | Red. MACs | Red. Params |
|--------|----------|----------|------------|-----------|-------------|
| GCNet | 1.84 | 718.01 | 3.18 | 4.7x | 1.8x |
| PSMNet | 0.88 | 256.66 | 5.22 | 1.7x | 2.9x |
| GA-Net-deep | 0.84 | 670.25 | 6.58 | 4.4x | 3.7x |
| GA-Net-11 | 0.93 | 383.42 | 4.48 | 2.5x | 2.5x |
| Gwc40-Cat24Base | 1.12 | 169.42 | 4.60 | 1.1x | 2.6x |
| GwcNet-gc | **0.76** | 260.49 | 6.82 | 1.7x | 3.9x |
| GwcNet-g | 0.79 | 246.27 | 6.43 | 1.6x | 3.6x |
| DeepPruner-Best | 0.86 | 129.23 | 7.39 | 0.8x | 4.2x |
| DeepPruner-Fast | 0.97 | **51.83** | 7.47 | 0.3x | 4.2x |
| **3D-MobileStereoNet** | **0.80** | **153.14** | **1.77** | — | — |

**说明**: 2D-MobileStereoNet 以 2.23M 参数达到 1.14 EPE，远超其他 2D 方法。3D-MobileStereoNet 以 1.77M 参数/153G MACs 达到 0.80 EPE，接近最优 GwcNet-gc 但参数少 3.9x。

### Table 5: KITTI 2015 验证集

| Method | EPE (px) | D1 (%) | px-3 (%) | MACs (G) | Params (M) |
|--------|----------|--------|----------|----------|------------|
| PSMNet | 0.88 | 2.00 | 2.10 | 256.66 | 5.22 |
| GA-Net-deep | 0.63 | 1.61 | 1.67 | 670.25 | 6.58 |
| GA-Net-11 | 0.67 | 1.92 | 2.01 | 383.42 | 4.48 |
| GwcNet-gc | 0.63 | 1.55 | 1.60 | 260.49 | 6.82 |
| GwcNet-g | **0.62** | **1.49** | **1.53** | 246.27 | 6.43 |
| 2D-MobileStereoNet | 0.79 | 2.35 | 2.67 | **32.2** | **2.12** |
| 3D-MobileStereoNet | 0.66 | 1.59 | 1.69 | 153.14 | 1.77 |

**说明**: KITTI 2015 验证集。2D-MobileStereoNet 精度与 PSMNet 可比但计算量仅 1/8；3D-MobileStereoNet 超越 PSMNet/GA-Net-11。

### Table 6: KITTI 2015 排行榜

| Methods | All D1_bg (%) | All D1_fg (%) | All D1_all (%) | Noc D1_bg (%) | Noc D1_fg (%) | Noc D1_all (%) |
|---------|--------------|--------------|---------------|--------------|--------------|---------------|
| MC-CNN | 2.89 | 8.88 | 3.89 | 2.48 | 7.64 | 3.33 |
| Fast DS-CS | 2.83 | 4.31 | 3.08 | 2.53 | 3.74 | 2.73 |
| GCNet | 2.21 | 6.16 | 2.87 | 2.02 | 5.58 | 2.61 |
| DeepPruner-Fast | 2.32 | 3.91 | 2.59 | 2.13 | 3.43 | 2.35 |
| PSMNet | 1.86 | 4.62 | 2.32 | 1.71 | 4.31 | 2.14 |
| AutoDispNet-CSS | 1.94 | 3.37 | 2.18 | 1.80 | 2.98 | 2.00 |
| DeepPruner-Best | 1.87 | 3.56 | 2.15 | 1.71 | 3.18 | 1.95 |
| GwcNet-g | **1.74** | 3.93 | 2.11 | **1.61** | 3.49 | **1.92** |
| **2D-MobileStereoNet** | 2.49 | 4.53 | 2.83 | 2.29 | 3.81 | 2.54 |
| **3D-MobileStereoNet** | 1.75 | 3.87 | **2.10** | 1.61 | **3.50** | 1.92 |

**说明**: 3D-MobileStereoNet 在 D1_all 上超越 GCNet 和 PSMNet，与 GwcNet-g 持平，但参数少 72%、运算少 38%。

### Table 7: 模型内存占用

| Method | GA-Net-deep | GA-Net-11 | GwcNet-gc | GwcNet-g | PSMNet | 2D-MSNet | 3D-MSNet |
|--------|------------|----------|----------|---------|--------|---------|---------|
| Memory (MB) | 26.4 | 18.0 | 27.9 | 26.3 | 21.1 | **10.03** | **7.99** |

**说明**: 两种 MobileStereoNet 的内存占用远小于其他方法，3D-MSNet 仅 7.99 MB，有利于内存受限设备部署。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[SceneFlow]] | 35,454 训练 / 4,370 测试 | 合成场景，540x960 | 预训练 + 评测 |
| [[KITTI-2015]] | 200 对（160/40 split） | 真实驾驶，376x1236 | 微调 + 评测 |

### 实现细节

- **Backbone**: ResNet-like feature extraction（共享左右图）
- **优化器**: 未明确提及（推测 Adam）
- **训练轮数**: 20 epochs（SceneFlow 预训练）
- **$d_{max}$**: 192
- **MACs 计算分辨率**: 256x512
- **损失函数**: [[Smooth L1 Loss]]

### 可视化结果

- 2D-MobileStereoNet 在 SceneFlow 和 KITTI 上视觉效果接近 3D 模型
- 3D-MobileStereoNet 因 3D 卷积在 encoder-decoder 中更好地保留边缘锐利度
- 两者都显著优于 GCNet 的视觉效果

---

## 批判性思考

### 优点
1. **系统性的轻量化研究**: 完整分析了 $v_1$/$v_2$ block 在 2D/3D 各模块中的效果，提供了清晰的设计指导
2. **Interlacing Cost Volume 创新**: 通过可学习的 cost volume 构建弥补 2D 模型精度差距，思路新颖
3. **极小的内存和参数开销**: 3D-MSNet 仅 1.77M 参数 / 7.99 MB 内存，适合嵌入式部署
4. **全面的消融实验**: 逐模块替换分析，对实际工程应用有参考价值

### 局限性
1. **未报告推理速度/FPS**: 论文仅报告 MACs 和参数量，未给出实际推理时间，难以评估实际部署效果
2. **仅在 SceneFlow + KITTI 上评测**: 缺少 ETH3D、Middlebury 等多样性评测，泛化能力未验证
3. **2D 模型与 SOTA 3D 模型差距仍大**: MSNet2D 的 KITTI EPE 0.79 vs GwcNet-g 0.62，轻量化的代价明显

### 潜在改进方向
1. 结合 RAFT-Stereo 的迭代优化思路替代固定的 hourglass 结构
2. 引入 [[Knowledge Distillation|知识蒸馏]] 从大模型迁移到轻量模型
3. 在更多 benchmark 上验证泛化能力

### 可复现性评估
- [x] 代码开源
- [x] 预训练模型
- [ ] 训练细节完整（缺少优化器细节）
- [x] 数据集可获取

---

## 关联笔记

### 基于
- [[MobileNetV1]]: $v_1$ block（depthwise separable convolution）的来源
- [[MobileNetV2]]: $v_2$ block（inverted residual）的来源
- [[GwcNet]]: 3D 基线网络架构

### 对比
- [[PSMNet]]: 经典 3D 立体匹配，参数量 5.22M vs 1.77M
- [[GANet]]: 高精度但高计算量的方法
- [[DeepPruner]]: 另一种高效立体匹配方案
- [[GCNet]]: 最早的端到端 3D 立体匹配

### 方法相关
- [[Depthwise Separable Convolution]]: 核心轻量化技术
- [[Inverted Residual Block]]: MobileNetV2 的核心组件
- [[Correlation Volume]]: Interlacing Cost Volume 的改进基础
- [[Smooth L1 Loss]]: 训练损失函数

### 硬件/数据相关
- [[SceneFlow]]: 主要训练数据集
- [[KITTI-2015]]: 评测数据集

---

## 速查卡片

> [!summary] MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching
> - **核心**: 将 MobileNet 轻量化模块扩展到 3D，设计参数极少的 2D/3D 立体匹配网络
> - **方法**: MobileNet 2D/3D blocks + Interlacing Cost Volume + Stacked Hourglass
> - **结果**: MSNet2D: 2.23M params, EPE 1.14; MSNet3D: 1.77M params, EPE 0.80 (SceneFlow)
> - **代码**: https://github.com/cogsys-tuebingen/mobilestereonet

---

*笔记创建时间: 2026-05-12*
