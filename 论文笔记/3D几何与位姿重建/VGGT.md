---
title: "VGGT: Visual Geometry Grounded Transformer"
method_name: "VGGT"
authors: [Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny]
year: 2025
venue: CVPR 2025 (Best Paper Award)
tags: [3d-reconstruction, multi-view-geometry, feed-forward-transformer, depth-estimation, point-tracking, camera-pose-estimation]
image_source: online
arxiv_html: https://arxiv.org/html/2503.11651v1
created: 2026-05-13
---

# 论文笔记：VGGT: Visual Geometry Grounded Transformer

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Visual Geometry Group, University of Oxford; Meta AI |
| 日期 | March 2025 |
| 项目主页 | https://vgg-t.github.io/ |
| 对比基线 | [[DUSt3R]], [[MASt3R]], [[VGGSfM]], [[COLMAP]] |
| 链接 | [arXiv](https://arxiv.org/abs/2503.11651) / [Code](https://github.com/facebookresearch/vggt) |

---

## 一句话总结

> 基于大规模 Transformer 的前馈式多视图 3D 重建网络，无需几何后处理即可在一秒内从任意数量图像预测相机参数、深度图、点云和 3D 点轨迹。

---

## 核心贡献

1. **统一多任务前馈架构**: 单一大型 [[Transformer]] 同时预测相机参数、深度图、点云图和点轨迹，无需 [[Bundle Adjustment]] 等几何后处理
2. **Alternating-Attention 机制**: 交替使用帧内自注意力和全局自注意力，在不引入 [[Cross-Attention]] 的情况下实现多视图信息融合
3. **全面 SOTA 性能**: 在相机位姿估计、多视图深度估计、稠密点云重建和 3D 点追踪等任务上全面超越现有方法，前馈仅需 0.2 秒

---

## 问题背景

### 要解决的问题
从多视图图像恢复场景的完整 3D 属性（相机参数、深度、点云、点轨迹），传统方法需要复杂的几何优化管线。

### 现有方法的局限
- 传统 [[Structure from Motion|SfM]] 管线（如 [[COLMAP]]）依赖迭代优化（特征匹配 → 三角化 → [[Bundle Adjustment]]），计算开销大
- [[DUSt3R]] 和 [[MASt3R]] 只能处理两张图像对，需要后处理融合成对重建结果
- 端到端可微 SfM（如 [[VGGSfM]]）仍依赖几何优化，复杂度高

### 本文的动机
大型 Transformer 在 NLP 和 CV 中已展现强大的学习能力（如 GPT、CLIP、[[DINOv2]]、Stable Diffusion），作者认为 3D 重建也可以不依赖显式几何推理，直接通过大规模前馈网络学习完成。

---

## 方法详解

### 模型架构

VGGT 采用**大型 [[Transformer]]** 架构：
- **输入**: $N$ 张 RGB 图像 $I_i \in \mathbb{R}^{3 \times H \times W}$，$N$ 可从 1 到数百
- **Backbone**: [[DINOv2]] (ViT-L) 进行 patch tokenization，14×14 patch
- **核心模块**: Alternating-Attention（交替帧内/全局[[Self-Attention]]），L=24 层
- **输出**: 相机参数 $g_i$、深度图 $D_i$、点云图 $P_i$、追踪特征 $T_i$
- **总参数**: ~1.2B

### 核心模块

#### 模块1: Alternating-Attention Backbone

**设计动机**: 在多帧信息融合与帧内特征归一化之间取得平衡

**具体实现**:
- 每层包含帧内 [[Self-Attention]]（各帧 token 独立 attend）和全局 Self-Attention（所有帧 token 联合 attend）
- 共 L=24 层，每层含帧内 + 全局两个自注意力子层
- 不使用 [[Cross-Attention]]，仅使用 Self-Attention
- 使用 QKNorm 和 LayerScale（初始化为 0.01）保证训练稳定性
- Feature dimension 1024，16 heads（遵循 ViT-L 设定）

#### 模块2: 预测头 (Prediction Heads)

**Camera Head**:
- 为每帧附加 1 个 camera token $t_i^g$ 和 4 个 register token $t_i^R$
- 第一帧使用特殊的可学习 token（$\bar{t}^g, \bar{t}^R$），使模型识别参考帧
- 4 层额外自注意力层 + 线性层输出相机参数 $g = [q, t, f]$（四元数旋转 + 平移 + FOV）
- 第一帧相机设为恒等变换：$q_1 = [0,0,0,1]$，$t_1 = [0,0,0]$

**Dense Prediction Head (DPT)**:
- 输出 image token 通过 [[DPT]] 层转换为稠密特征图 $F_i \in \mathbb{R}^{C'' \times H \times W}$
- 取第 4、11、17、23 层的 token 输入 DPT 进行上采样
- 3×3 卷积分别映射到深度图、点云图和追踪特征
- 同时输出 aleatoric 不确定性图 $\Sigma_i^D$ 和 $\Sigma_i^P$

#### 模块3: 追踪模块 (Tracking Module)

**设计动机**: 利用 [[CoTracker|CoTracker2]] 架构在学习到的追踪特征上进行点追踪

**具体实现**:
- 给定查询点 $y_q$，双线性采样 $T_q$ 获取特征
- 与所有帧的特征图 $T_i$ 计算 [[Correlation Map|相关性图]]
- 通过自注意力层处理相关性信息
- 不假设时序顺序，适用于任意图像集合（不仅限于视频）
- 与主网络 $f$ 端到端联合训练

### Over-complete Predictions

并非所有预测量独立——相机参数可从点云图通过 [[PnP]] 推出，深度可从点云和相机推出。但训练时显式预测所有量带来显著性能增益。推理时，独立估计的深度图与相机参数结合得到的 3D 点比专用点云分支更准确。

---

## 关键公式

### 公式1: [[Feed-Forward Network|前馈推理]]

$$
f\bigl((I_i)_{i=1}^N\bigr) = (g_i, D_i, P_i, T_i)_{i=1}^N
$$

**含义**: 单次前馈即可从 $N$ 张图像预测所有 3D 属性

**符号说明**:
- $g_i \in \mathbb{R}^9$: 相机参数（四元数旋转 $q \in \mathbb{R}^4$ + 平移 $t \in \mathbb{R}^3$ + FOV $f \in \mathbb{R}^2$）
- $D_i \in \mathbb{R}^{H \times W}$: 深度图
- $P_i \in \mathbb{R}^{3 \times H \times W}$: 点云图（在第一帧坐标系下）
- $T_i \in \mathbb{R}^{C \times H \times W}$: 追踪特征

### 公式2: [[Multi-Task Learning|多任务训练损失]]

$$
\mathcal{L} = \mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pmap}} + \lambda \mathcal{L}_{\text{track}}
$$

**含义**: 四项损失联合训练，$\lambda = 0.05$ 平衡追踪损失

**符号说明**:
- $\mathcal{L}_{\text{camera}} = \sum_{i=1}^N \|\hat{g}_i - g_i\|_\epsilon$: Huber 损失
- $\mathcal{L}_{\text{depth}}$: 深度损失（含 aleatoric 不确定性加权）
- $\mathcal{L}_{\text{pmap}}$: 点云图损失
- $\mathcal{L}_{\text{track}}$: 追踪损失

### 公式3: [[Aleatoric Uncertainty|深度损失（含不确定性）]]

$$
\mathcal{L}_{\text{depth}} = \sum_{i=1}^N \|\Sigma_i^D \odot (\hat{D}_i - D_i)\| + \|\Sigma_i^D \odot (\nabla \hat{D}_i - \nabla D_i)\| - \alpha \log \Sigma_i^D
$$

**含义**: 使用 aleatoric 不确定性加权的深度损失，附加梯度项约束深度变化的平滑性

**符号说明**:
- $\Sigma_i^D$: 预测的不确定性图
- $\odot$: 逐元素乘法
- $\nabla$: 空间梯度算子
- $\alpha$: 不确定性正则化系数

### 公式4: [[Point Map|点云图损失]]

$$
\mathcal{L}_{\text{pmap}} = \sum_{i=1}^N \|\Sigma_i^P \odot (\hat{P}_i - P_i)\| + \|\Sigma_i^P \odot (\nabla \hat{P}_i - \nabla P_i)\| - \alpha \log \Sigma_i^P
$$

**含义**: 与深度损失结构相同，用于 3D 点云图回归

### 公式5: [[Point Tracking|追踪损失]]

$$
\mathcal{L}_{\text{track}} = \sum_{j=1}^M \sum_{i=1}^N \|y_{j,i} - \hat{y}_{j,i}\|
$$

**含义**: M 个查询点在 N 帧上的 2D 坐标回归损失，另附可见性二元交叉熵损失

**符号说明**:
- $y_{j,i}$: 第 $j$ 个查询点在第 $i$ 帧的 GT 坐标
- $\hat{y}_{j,i}$: 预测坐标

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2503.11651v1/x1.png)

**说明**: VGGT 接受 1 到数百张图像输入，单次前馈在一秒内预测相机参数、点云图、深度图和点轨迹。获得 CVPR 2025 Best Paper Award。

### Figure 2: Architecture / 模型架构

![Figure 2](https://arxiv.org/html/2503.11651v1/x2.png)

**说明**: 架构概览。输入图像通过 [[DINOv2]] patchify 为 token，附加 camera token 后交替经过帧内和全局 [[Self-Attention]] 层。Camera Head 预测相机参数，[[DPT]] Head 输出深度图、点云图和追踪特征。

### Figure 3: Qualitative Comparison / 定性对比

![Figure 3](https://arxiv.org/html/2503.11651v1/x3.png)

**说明**: 与 [[DUSt3R]] 的 3D 点云预测定性对比。展示了油画场景几何恢复、无重叠区域重建和重复纹理处理的能力。

### Figure 4: Additional Visualizations / 额外可视化

![Figure 4](https://arxiv.org/html/2503.11651v1/x4.png)

**说明**: 点云图估计的额外可视化。相机棱锥展示估计的相机位姿，展现对室内外复杂场景的重建能力。

### Figure 5: Point Tracking / 点追踪可视化

![Figure 5](https://arxiv.org/html/2503.11651v1/x5.png)

**说明**: 刚性和动态点追踪可视化。上方：VGGT 追踪模块在无序静态图像集上输出关键点轨迹。下方：将 VGGT backbone 微调增强 [[CoTracker]] 的动态点追踪能力。

### Figure 6: Novel View Synthesis / 新视角合成

![Figure 6](https://arxiv.org/html/2503.11651v1/x6.png)

**说明**: 新视角合成定性结果。上行为输入图像，中行为目标视角 GT，下行为 VGGT 合成结果。

### Table 1: 相机位姿估计

| Method | Re10K AUC@30 ↑ | CO3Dv2 AUC@30 ↑ | Time |
|--------|----------------|------------------|------|
| COLMAP+SPSG | 45.2 | 25.3 | ~15s |
| PixSfM | 49.4 | 30.1 | >20s |
| DUSt3R | 67.7 | 76.7 | ~7s |
| MASt3R | 76.4 | 81.8 | ~9s |
| VGGSfM v2 | 78.9 | 83.4 | ~10s |
| Fast3R | 72.7 | 82.5 | ~0.2s |
| **VGGT (FF)** | **85.3** | **88.2** | **~0.2s** |
| **VGGT (with BA)** | **93.5** | **91.8** | **~1.8s** |

**说明**: VGGT 前馈模式在 0.2 秒内全面超越所有方法。加 BA 后进一步提升至 93.5/91.8。

### Table 2: DTU 数据集稠密 MVS 估计

| Known GT Camera | Method | Acc.↓ | Comp.↓ | Overall↓ |
|-----------------|--------|-------|--------|----------|
| Yes | GeoMVSNet | 0.331 | 0.259 | 0.295 |
| Yes | MASt3R | 0.403 | 0.344 | 0.374 |
| No | DUSt3R | 2.677 | 0.805 | 1.741 |
| No | **VGGT** | **0.389** | **0.374** | **0.382** |

**说明**: 在不使用 GT 相机参数的条件下，VGGT 将 DUSt3R 的 Overall 从 1.741 降至 0.382，接近使用 GT 相机的方法。

### Table 3: ETH3D 点云图估计

| Method | Acc.↓ | Comp.↓ | Overall↓ | Time |
|--------|-------|--------|----------|------|
| DUSt3R | 1.167 | 0.842 | 1.005 | ~7s |
| MASt3R | 0.968 | 0.684 | 0.826 | ~9s |
| **VGGT (Depth+Cam)** | **0.873** | **0.482** | **0.677** | **~0.2s** |

**说明**: VGGT 在 0.2 秒内超越需要约 10 秒全局对齐的 DUSt3R 和 MASt3R。

### Table 4: ScanNet-1500 两视图匹配

| Method | AUC@5 ↑ | AUC@10 ↑ | AUC@20 ↑ |
|--------|---------|----------|----------|
| SuperGlue | 16.2 | 33.8 | 51.8 |
| LoFTR | 22.1 | 40.8 | 57.6 |
| Roma | 31.8 | 53.4 | 70.9 |
| **VGGT** | **33.9** | **55.2** | **73.4** |

**说明**: VGGT 的追踪头虽非专为两视图设计，仍超越 SOTA 两视图匹配方法 [[RoMa|Roma]]。

### Table 5: Alternating-Attention 消融实验 (ETH3D)

| Architecture | Acc.↓ | Comp.↓ | Overall↓ |
|-------------|-------|--------|----------|
| Cross-Attention | 1.287 | 0.835 | 1.061 |
| Global Self-Attention Only | 1.032 | 0.621 | 0.827 |
| **Alternating-Attention** | **0.901** | **0.518** | **0.709** |

**关键发现**: Alternating-Attention 显著优于 Cross-Attention 和纯全局 Self-Attention，验证了帧内/全局交替策略的有效性。

### Table 6: 多任务学习消融 (ETH3D)

| w. $\mathcal{L}_{\text{cam}}$ | w. $\mathcal{L}_{\text{depth}}$ | w. $\mathcal{L}_{\text{track}}$ | Acc.↓ | Comp.↓ | Overall↓ |
|---|---|---|-------|--------|----------|
| No | Yes | Yes | 1.042 | 0.627 | 0.834 |
| Yes | No | Yes | 0.920 | 0.534 | 0.727 |
| Yes | Yes | No | 0.976 | 0.603 | 0.790 |
| **Yes** | **Yes** | **Yes** | **0.901** | **0.518** | **0.709** |

**关键发现**: 四项任务联合训练效果最佳。相机参数估计对点云质量贡献最大，追踪损失也有显著正向影响。

### Table 7: GSO 数据集新视角合成

| Method | Known Input Cam | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|----------------|--------|--------|---------|
| GS-LRM | Yes | 29.59 | 0.944 | 0.051 |
| LVSM | Yes | 31.71 | 0.957 | 0.027 |
| **VGGT-NVS** | **No** | **30.41** | **0.949** | **0.033** |

**说明**: 不需要输入相机参数且仅使用 20% 训练数据，VGGT 即达到竞争力结果。

### Table 8: TAP-Vid 动态点追踪

| Method | Kinetics AJ | RGB-S AJ | DAVIS AJ |
|--------|------------|---------|---------|
| BootsTAPIR | 54.6 | 70.8 | 61.4 |
| CoTracker | 49.6 | 67.4 | 61.8 |
| **CoTracker + VGGT** | **57.2** | **72.1** | **64.7** |

**说明**: 使用 VGGT 预训练特征替换 [[CoTracker]] backbone 后，各数据集 AJ 均有显著提升。

### Table 9: 运行时间和 GPU 内存

| Input Frames | 1 | 2 | 4 | 10 | 50 | 100 | 200 |
|-------------|-----|------|------|------|------|-------|-------|
| Time (s) | 0.04 | 0.05 | 0.07 | 0.14 | 1.04 | 3.12 | 8.75 |
| Mem. (GB) | 1.88 | 2.07 | 2.45 | 3.63 | 11.41 | 21.15 | 40.63 |

**说明**: 单张 NVIDIA H100 GPU (flash attention v3)，336×518 分辨率。200 帧仅需 8.75 秒和 40.63 GB 显存。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Co3Dv2, BlendMVS, DL3DV, MegaDepth | 大规模 | 室内外真实场景 | 训练 |
| Kubric, PointOdyssey | 合成 | 含动态物体 | 追踪训练 |
| ScanNet, HyperSim, Habitat, Replica | 室内 | 含深度标注 | 训练 |
| RealEstate10K, CO3Dv2 | 真实 | 多视图视频 | 位姿评估 |
| DTU, ETH3D | 真实 | MVS 基准 | 深度/点云评估 |
| ScanNet-1500 | 真实 | 室内 | 匹配评估 |
| TAP-Vid (Kinetics, RGB-S, DAVIS) | 真实/合成 | 视频 | 追踪评估 |

### 实现细节

- **Backbone**: [[DINOv2]] ViT-L，14×14 patch
- **优化器**: [[AdamW]]，峰值学习率 0.0002，8K warmup + cosine schedule
- **Batch Size**: 每 batch 48 帧（2-24 帧/场景）
- **训练轮数**: 160K iterations
- **硬件**: 64× NVIDIA A100 GPU，训练 9 天
- **精度**: bfloat16 + gradient checkpointing
- **梯度裁剪**: 阈值 1.0

### 可视化结果

- 在油画、无重叠视角、重复纹理等极端场景下 VGGT 仍能产生合理 3D 重建
- 结合 BA 后在 phototourism 场景上表现尤为突出
- 单视图输入时也能展示合理的深度估计能力

---

## 批判性思考

### 优点
1. **真正端到端**: 从原始图像到完整 3D 属性的前馈推理，无需复杂管线
2. **极高效率**: 10 帧仅需 0.2 秒，比 DUSt3R (7s)、MASt3R (9s) 快 30-45 倍
3. **多任务互补**: 多任务联合训练带来各任务的相互增益，over-complete prediction 策略新颖
4. **强大泛化**: 在从未训练过的 RealEstate10K 上仍取得 SOTA
5. **下游适配灵活**: 作为 feature backbone 增强追踪和新视角合成效果

### 局限性
1. 不支持鱼眼或全景图像
2. 极端旋转角度下性能下降
3. 对严重非刚性变形场景处理能力有限
4. 训练成本高昂（64 A100 × 9 天）

### 潜在改进方向
1. 可微 [[Bundle Adjustment]] 集成（论文提及可使训练变为无监督，但计算代价高 4×）
2. 支持非刚性场景的扩展
3. 更高分辨率输入的高效处理

### 可复现性评估
- [x] 代码开源（github.com/facebookresearch/vggt）
- [x] 预训练模型（含商用许可版本 VGGT-1B-Commercial）
- [x] 训练细节完整
- [x] 数据集可获取（均为公开数据集）

---

## 关联笔记

### 基于
- [[DUSt3R]]: 点云图表示的开创性工作
- [[MASt3R]]: DUSt3R 的改进版，加入匹配能力
- [[DINOv2]]: 视觉基础模型用作 tokenizer
- [[CoTracker]]: 追踪模块架构来源

### 对比
- [[DUSt3R]]: 双图配对 → 全局优化 vs VGGT 一次前馈
- [[MASt3R]]: 仍需后处理，VGGT 前馈即超越
- [[VGGSfM]]: 依赖可微 BA，VGGT 无需几何优化
- [[COLMAP]]: 传统 SfM 管线，慢且精度低

### 方法相关
- [[Alternating Attention]]: 核心注意力机制
- [[DPT]]: 稠密预测头架构
- [[Bundle Adjustment]]: 可选后处理提升精度
- [[Aleatoric Uncertainty]]: 损失函数中的不确定性建模
- [[Point Map]]: 稠密 3D 点云表示

### 硬件/数据相关
- [[NVIDIA A100]]: 训练硬件
- [[NVIDIA H100]]: 推理评估硬件

---

## 速查卡片

> [!summary] VGGT: Visual Geometry Grounded Transformer
> - **核心**: 大型前馈 Transformer 直接预测场景完整 3D 属性，无需几何后处理
> - **方法**: Alternating-Attention (帧内+全局自注意力)，DINOv2 tokenizer，多任务联合训练
> - **结果**: CVPR 2025 Best Paper；位姿 AUC@30 达 85.3 (FF) / 93.5 (BA)；0.2 秒/10 帧
> - **代码**: https://github.com/facebookresearch/vggt

---

*笔记创建时间: 2026-05-13*
