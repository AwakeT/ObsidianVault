---
title: "VGGT-Ω"
method_name: "VGGT-Omega"
authors: [Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, Christian Rupprecht]
year: 2026
venue: CVPR 2026 (Oral)
tags: [3d-reconstruction, feed-forward-transformer, dynamic-reconstruction, depth-estimation, camera-pose-estimation, scaling-law, self-supervised-learning, dino-v3, register-attention, vla]
image_source: online
arxiv: https://arxiv.org/abs/2605.15195
arxiv_pdf: https://arxiv.org/pdf/2605.15195
project: https://vggt-omega.github.io/
code: https://github.com/facebookresearch/vggt-omega
created: 2026-05-20
---

# 论文笔记：VGGT-Ω

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Visual Geometry Group, University of Oxford; Meta AI |
| 日期 | 2026-05-14 |
| 会议 | CVPR 2026 Oral |
| 项目主页 | https://vggt-omega.github.io/ |
| 代码 | https://github.com/facebookresearch/vggt-omega |
| 对比基线 | [[VGGT]], [[MegaSaM]], [[Depth Anything 3]], [[PI3]], [[MonST3R]], MapAnything |
| 链接 | [arXiv](https://arxiv.org/abs/2605.15195) / [PDF](https://arxiv.org/pdf/2605.15195) / [Project](https://vggt-omega.github.io/) / [Code](https://github.com/facebookresearch/vggt-omega) |

---

## 一句话总结

> VGGT-Ω 是 [[VGGT]] 的规模化升级版：用 [[DINOv3]] 初始化、Register Attention、单一轻量 dense head、多任务训练损失、大规模动态视频标注管线和 teacher-student 自监督训练，将前馈式 3D/4D 重建扩展到更大模型和更大数据规模，并证明重建模型性能会随模型/数据规模呈现可预测提升。

---

## 核心贡献

1. **可规模化 VGGT 架构**：用 16 个 scene registers 聚合跨帧全局信息，引入 Register Attention 替代部分昂贵 global attention；同时把 VGGT 的多个 dense prediction heads 简化为单个 depth head + 多任务训练损失，训练显存约为 VGGT 的 30%。
2. **动态场景训练数据管线**：构建面向互联网视频的大规模标注流程，结合 VLM 预筛选、Grounding-DINO 动态区域屏蔽、特征匹配/跟踪、VGGT 初始化、COLMAP BA、多视图一致性和监督式几何过滤，从约 4000 万内部互联网风格视频中筛出约 80 万高质量序列，其中约三分之一含动态内容。
3. **自监督训练协议**：从监督训练 checkpoint 出发，用 teacher-student EMA 对 unlabeled videos 做特征、相机和深度蒸馏，利用 1800 万未标注视频增强泛化。
4. **规模律验证**：系统展示模型从 0.2B 扩到 10B、数据从 2K 扩到 2M sequences 时，3D point error 单调下降，呈现类似 power-law 的 scaling 行为。
5. **静态 + 动态重建 SOTA**：在 7-Scenes、NRGBD、ETH3D、DyCheck、Sintel、TUM-Dynamic 上同时提升相机位姿和深度估计，Sintel AUC@3 从 22.5 提升到 40.0，相对提升 77%。
6. **Scene registers 的通用表征价值**：scene tokens 可直接增强 OpenVLA-OFT，在 LIBERO 平均成功率从 97.1 提升到 98.5；也能通过 InfoNCE 与 VLM/LLM 文本 embedding 对齐，说明几何重建 token 具备语义和空间理解能力。

---

## 问题背景

### 要解决的问题

[[VGGT]] 已证明大规模前馈 Transformer 可以直接从多视图图像预测相机、深度、点云和跟踪特征，并在若干场景中超过传统 SfM / BA 管线。但 VGGT 仍有三个关键瓶颈：

1. **训练成本高**：global attention 和多个高分辨率 dense heads 带来显存瓶颈；
2. **动态场景能力有限**：传统多视图重建数据和监督主要面向刚性静态场景；
3. **规模化规律不清楚**：3D 前馈重建模型是否像 LLM / 2D foundation model 一样随模型和数据规模稳定提升，尚未被系统验证。

### 现有方法的局限

- [[COLMAP]]、MegaSaM 等优化式方法在良态场景下精度高，但依赖匹配、三角化、BA 或非刚性优化，速度慢且对宽基线、弱纹理、动态内容敏感。
- [[DUSt3R]] / [[MASt3R]] 一类 pairwise 方法需要后处理融合多视图，难以直接处理长序列。
- [[VGGT]] 虽然前馈高效，但原始结构包含多 dense heads 和高分辨率卷积层，训练显存成本高，不利于进一步扩大数据和模型。
- [[Depth Anything 3]]、PI3 等动态场景前馈模型表现强，但在严格位姿、重复纹理、大 roll 运动、宽基线等场景仍存在失败模式。

### 本文动机

作者提出的问题是：

> Feed-forward reconstruction 能否像语言模型和 2D 视觉基础模型一样通过模型规模、数据规模和自监督训练继续扩展？

VGGT-Ω 的答案是肯定的。论文通过架构简化、数据管线、动态视频监督和 teacher-student 自监督，证明前馈重建不仅能提升几何性能，还能学习对 VLA 和语言对齐有用的空间 token。

---

## 方法详解

### 模型架构

VGGT-Ω 是一个前馈式多视图 / 视频重建 Transformer：

- **输入**：$N$ 张 RGB 图像 $I_1, ..., I_N \in \mathbb{R}^{3 \times H \times W}$
- **Backbone**：[[DINOv3]] 初始化的 ViT，patch size = 16
- **每帧 token**：image tokens + 1 个 camera token + 16 个 scene registers
- **时序聚合**：Alternating Attention，即 frame-wise attention 与 global attention / register attention 交替
- **输出**：每帧相机参数 $g_i=(q_i,t_i,f_i)$ 和深度图 $D_i$
- **训练监督**：虽然只预测 camera + depth，但仍通过 point loss 和 matching loss 施加多任务几何监督

形式化输出：

$$
((g_1, D_1), ..., (g_N, D_N)) = f(I_1, ..., I_N)
$$

其中：

- $D_i \in \mathbb{R}^{H \times W}$：第 $i$ 帧深度；
- $g_i \in \mathbb{R}^{9}$：四元数旋转 $q_i \in \mathbb{R}^{4}$、平移 $t_i \in \mathbb{R}^{3}$、视场角 $f_i \in \mathbb{R}^{2}$；
- 主点默认位于图像中心。

与 [[VGGT]] 的关键差异是：VGGT-Ω **不直接预测 point maps 和 tracking features**，但保留对应训练损失。

---

## 核心模块

### 模块1：DINOv3 Tokenization + Scene Registers

每帧输入图像经过 DINOv3 初始化的 ViT 得到 patch tokens：

$$
z_i^F = \text{DINO}(I_i) \in \mathbb{R}^{H'W' \times C}
$$

随后每帧拼接：

```text
[camera token] + [16 scene registers] + [image patch tokens]
```

其中：

- **camera token**：用于预测相机参数；
- **scene registers**：用于聚合跨帧场景级信息；
- camera/register token 具有 reference frame 与 normal frame 两种可学习参数，继承 VGGT 的参考帧机制。

论文强调这些 registers 不是辅助 token，而是可以作为下游任务的场景级空间表征。

### 模块2：Register Attention

VGGT 的 global attention 允许所有帧的所有 tokens 互相注意，计算成本随总 token 数平方增长。作者观察到 VGGT global attention map 很稀疏，因此提出 Register Attention：

```text
Full Global Attention:
    all image tokens + camera/register tokens across frames attend globally

Register Attention:
    only scene registers across frames attend globally
    image tokens do not directly attend to other frames in this block
```

经过 Register Attention 更新后的 registers，会在后续 frame-wise attention 中与本帧 image tokens 交互，将聚合的全局信息重新分发给局部 patch tokens。

关键结果：

- 替换 25% global attention 为 register attention，point error 从 0.071 仅变为 0.073，几乎无损；
- 训练时 backbone FLOPs 降低约 23%，显存降低约 16%；
- 如果所有 global attention 都替换为 register attention，FLOPs 可降到原来的 6%，1000 帧推理从 240.2s 降到 11.7s，但精度下降到接近原始 VGGT 水平。

这说明 registers 可以作为跨帧信息交换瓶颈，但不能完全替代所有 dense cross-frame interaction。

### 模块3：轻量 Dense Decoder

VGGT 使用多个 DPT dense heads，且高分辨率卷积层训练显存开销很大。VGGT-Ω 改为：

1. 保留低分辨率 DPT convolution layers；
2. 移除 1/4 分辨率以上的高分辨率卷积块；
3. 用单个 MLP + pixel shuffle 输出 depth 和 confidence。

MLP 输出 $2u^2$ 通道，$u=4$，随后 pixel shuffle 将 token map 上采样到更高分辨率，两个输出通道分别对应：

```text
depth
confidence / aleatoric uncertainty
```

作者也尝试过完全 MLP-only decoder。它在 benchmark 上效果不错且更快、更省显存，但容易产生 blocky artifacts，尤其在室外远距离结构、天空、山脉等无界深度区域，因此最终保留少量低分辨率卷积。

### 模块4：单次 Camera Head

VGGT 的 camera head 使用迭代精炼。VGGT-Ω 为简化架构，使用轻量 transformer 处理所有 camera tokens 与 registers，然后每个更新后的 camera token 经过 MLP 输出相机参数。

关键变化：

- 不使用 VGGT 的 iterative refinement；
- 直接单 pass 输出 camera；
- 论文在 Discussion 中承认，如果恢复 VGGT 式迭代精炼，位姿性能还能进一步提升约 4%-6% AUC@3，但作者选择保留更简洁的 base model。

### 模块5：动态场景输出设计

VGGT-Ω 支持动态场景，但没有显式预测 motion mask、dynamic point map 或 ray map。

作者的判断是：

- point map 在静态场景中有用，但动态场景会把相机运动和物体运动耦合；
- ray map 会增加昂贵 dense output，并可能把像素外观变化和相机参数纠缠；
- 因此最终只预测 **depth + camera**，动态场景能力交给数据驱动先验学习。

这是一种“隐式动态建模”：不显式分解动态物体，而是让模型通过大规模动态数据学习合理先验。

---

## 关键公式

### 公式1：前馈重建

$$
((g_1, D_1), ..., (g_N, D_N)) = f(I_1, ..., I_N)
$$

**含义**：单次前馈从多帧图像得到所有相机和深度。

### 公式2：总损失

$$
\mathcal{L}
= \lambda_{\text{cam}}\mathcal{L}_{\text{cam}}
+ \lambda_{\text{depth}}\mathcal{L}_{\text{depth}}
+ \lambda_{\text{point}}\mathcal{L}_{\text{point}}
+ \lambda_{\text{match}}\mathcal{L}_{\text{match}}
$$

补充材料给出的权重：

```text
λ_cam = 5.0
λ_depth = 1.0
λ_point = 0.5
λ_match = 0.1
```

### 公式3：Camera Loss

$$
\mathcal{L}_{\text{cam}} = \sum_i |\hat{g}_i - g_i|
$$

与 VGGT 的 Huber loss 不同，VGGT-Ω 使用 L1，作者认为更稳定。

### 公式4：Depth Loss

Depth loss 沿用 VGGT 的 aleatoric uncertainty 和梯度一致性：

$$
\mathcal{L}_{\text{depth}}
= \sum_i
\left[
c_i^D \odot (1 + D_i^{-1}) \odot e_i
+ c_i^D \odot \nabla e_i
\right]
- \alpha \sum_i \log c_i^D
$$

其中：

$$
e_i = \hat{D}_i - D_i
$$

**含义**：用不确定性加权深度残差和深度梯度残差，同时用 $\log c_i^D$ 避免模型无意义地提高不确定性。

### 公式5：Point Loss

模型不直接输出 point map，但从预测深度和相机反投影得到点云后做 point supervision：

$$
e_i = \pi^{-1}(\hat{D}_i, \hat{g}_i) - P_i
$$

Point loss 与 depth loss 形式相同，只是把残差替换为 3D point residual。

**关键意义**：保留 VGGT over-complete prediction 的训练收益，但不保留昂贵 dense point head。

### 公式6：Matching Loss

最后一层 tokens 上构造正负 patch 对：

$$
\mathcal{L}_{\text{match}}
= \mathbb{E}_{pos}[-\log \sigma(s)]
+ \mathbb{E}_{neg}[-\log (1-\sigma(s))]
$$

其中 $s$ 是 L2-normalized token 的 cosine similarity。

正样本来自 3D 投影重叠的 patch；负样本需要同时满足几何约束和外观差异，避免动态区域或未匹配区域被误认为负样本。

---

## 数据与训练

### 训练数据

VGGT-Ω 的训练数据由两部分组成：

#### 公开与内部监督数据

公开数据包括 Aria、BEDLAM、BEHAVIOR-1K、Co3Dv2、uCo3D、DL3DV、Dynamic Replica、EDEN、EFM3D、HOT3D、Habitat、Hypersim、Mapfree、Mapillary Metropolis、MPSD、MegaDepth、MegaSynth、Mid-Air、Mvssynth、ParallelDomain-4D、Replica、SAIL-VOS、ScanNet 系列、TartanAirV2、TartanGround、Taskonomy、UnrealStereo4K、Virtual KITTI、Waymo、WildRGBD 等。

作者排除了 VGGT 使用过的 Kubric 和 PointOdyssey，因为其背景几何可能是 proxy geometry，会造成无效深度监督。

总规模：

```text
约 3M supervised sequences
每个 sequence 含 10 到 20,000 张图像
```

#### 内部互联网视频数据

从约 4000 万互联网风格视频中构建新标注管线：

```text
40M videos
  → VLM pre-filter
  → dynamic region masking
  → feature matching / tracking
  → VGGT initialization
  → COLMAP bundle adjustment
  → dense MVS depth
  → multi-view consistency
  → supervised geometric filtering
  → 600K static + 200K dynamic high-quality sequences
```

最终监督数据比 VGGT 多 15 倍以上。

### VLM 预筛选

VLM 用于判断视频是否适合多视图几何重建，过滤：

- 多镜头剪辑、转场、montage；
- 屏幕录制、幻灯片、动画；
- 严重运动模糊、rolling shutter、损坏视频；
- 大面积水印、遮挡；
- 360° / 鱼眼等非 pinhole 投影；
- 纯旋转或 zoom、缺乏平移视差；
- 弱纹理、强重复纹理；
- 反射、透明、焦外虚化；
- 动态主体占比极高的片段。

VLM 约将 50% 视频判为 hard reject，40% 判为 technically reconstructible but low quality，仅约 10% 进入后续高质量标注流程。

### 动态区域处理

使用 [[Grounding-DINO]] 检测人、车等可移动类别 bounding boxes，并将这些区域从匹配、跟踪和验证中排除。

注意：这并不意味着模型不学习动态物体深度。作者假设动态像素深度主要从合成数据中学习，而真实互联网视频监督只保留可靠的刚性区域。

### 自监督训练

VGGT-Ω 的自监督采用 teacher-student EMA：

```text
supervised checkpoint
    ├── student: gradient descent
    └── teacher: EMA(student)
```

同一组视频帧经过不同增强输入 student 和 teacher：

- color jitter；
- blur；
- 90° rotation；
- random patch masking；
- random frame reordering。

对齐回相同帧顺序后，student 匹配 teacher：

1. 多层 token feature L2 matching；
2. camera regression；
3. depth regression。

为防 collapse，自监督阶段冻结 camera 和 depth heads，只训练表征部分。teacher 以 EMA 更新：

$$
\theta_T \leftarrow m \theta_T + (1-m)\theta_S
$$

补充材料中 EMA decay 为 0.999。

### 实现细节

| 项目 | 配置 |
|---|---|
| 模型规模 | 200M / 500M / 1B / 10B |
| alternating-attention blocks | 12 / 12 / 24 / 16 |
| hidden size | 384 / 768 / 1024 / 4096 |
| Backbone 初始化 | [[DINOv3]] |
| patch size | 16 |
| 训练迭代 | 240K |
| 训练阶段 | 160K supervised + 50K self-supervised + 30K supervised |
| 优化器 | [[AdamW]] |
| 学习率 | supervised 2e-4, self-supervised 1e-4 |
| schedule | 5% linear warmup + 95% cosine decay |
| 输入帧数 | uniform sample from [1, 24] |
| 图像面积 | 约 512×512，aspect ratio ∈ [0.33, 1.33] |
| 训练硬件 | 128 × 96GB H100 |
| 精度与并行 | bfloat16, gradient checkpointing, FSDP |
| 梯度裁剪 | norm 1.0 |
| 注意力稳定 | QKNorm |

---

## 关键图表

### Figure 1：模型规模与数据规模 Scaling

| Scaling 维度 | 配置变化 | Point Error ↓ |
|---|---|---|
| Model size | 0.2B → 1B → 5B → 10B | 0.107 → 0.073 → 0.057 → 0.046 |
| Data size | 2K → 20K → 200K → 2M sequences | 0.275 → 0.210 → 0.160 → 0.129 → 0.073 |

**说明**：模型和数据规模同时呈现单调收益，说明前馈 3D 重建具备可预测的 scaling 潜力。

### Figure 2：架构概览

VGGT-Ω 将 camera token 和 16 个 scene registers 拼接到 image tokens 上，在 attention blocks 中交替使用：

```text
global attention 或 register attention
frame attention
```

并用训练期 point / matching losses 替代 VGGT 中冗余 dense heads。

### Figure 3：VGGT Global Attention 稀疏性

作者可视化 VGGT 第 13 层 global attention，发现跨帧 attention map 非常稀疏。这是 Register Attention 的动机：全量 token-to-token 跨帧交互并非每层都必要。

### Figure 4：静态与动态场景定性结果

VGGT-Ω 能处理交通流、网球运动员、自然景观、水下珊瑚等场景。论文特别强调模型能在没有显式 motion output 的情况下处理动态内容。

### Figure 5：与 MegaSaM 定性对比

MegaSaM 在航拍、稀疏动态场景和大 roll 室内场景中容易出现几何漂移、纹理涂抹、重复模式和结构断裂。VGGT-Ω 的重建更全局一致。

### Figure 6：与 Depth Anything 3 定性对比

DA3 在雪地缆车序列中被重复纹理误导，几乎估不出相机运动；在无人机强 roll 飞向塔的序列中产生严重 ghosting 和重复塔结构。VGGT-Ω 能恢复更合理轨迹和结构。

### Figure 7：推理显存与速度

公平比较使用单张 80GB A100、flash attention v2：

| 方法 | 最大输入帧数 | 速度特点 |
|---|---:|---|
| DA3-Giant | 约 750 帧 OOM | 更早触达显存上限 |
| VGGT | 约 1250 帧 | corrected implementation 后显存可大幅优化 |
| VGGT-Ω | 约 1250 帧 | 比 VGGT 快约 20-25% |
| VGGT-Ω Register Attention Only | 约 1250 帧 | 1000 帧 runtime 240.2s → 11.7s，但精度下降 |

重要结论：

- 推理阶段去掉 global attention 主要提升速度，不显著降低峰值显存；
- flash attention 不显式 materialize full attention matrix；
- 峰值显存主要由 frame attention activations 和 FFN intermediates 等线性随帧数增长的张量决定。

### Figure 8：Scene Registers 与语言对齐

VGGT-Ω 从 registers 读取场景级 embedding，VLM 为同一图像序列生成场景描述 embedding，二者用 symmetric InfoNCE 对齐。由于 language token 只能读取 registers，不能读取 image patch tokens，因此成功对齐说明 registers 本身携带了场景级语义信息。

### Figure 9：Motion-aware Representation

对中间 image tokens 做 PCA + k-means，无需 motion labels / optical flow / learned probe，即可把移动舞者从静态人群和背景中分离出来。说明动态感知能力从重建训练中自发涌现。

### Figure 10：训练数据常见问题

论文补充材料总结了大规模 3D 数据的典型污染源：

- sensor depth foreground-background leakage；
- synthetic fake background；
- COLMAP / BA doming effect；
- humans in walls；
- thin structure depth 错误；
- transparent/window 深度定义不一致。

这些数据质量问题会被模型记忆，并在相似测试场景中表现为具体失败模式。

---

## 实验结果

### Table 1：Camera Pose Estimation

指标为 AUC，越高越好。

| Method | 7Scenes AUC@3 | NRGBD AUC@3 | ETH3D AUC@3 | DyCheck AUC@3 | Sintel AUC@3 | TUM-Dyn AUC@3 |
|---|---:|---:|---:|---:|---:|---:|
| MonST3R | 9.0 | 13.9 | 1.7 | 11.5 | 4.3 | 7.7 |
| MapAnything | 5.8 | 35.2 | 13.2 | 6.1 | 2.9 | 4.3 |
| MegaSaM | 10.6 | 17.2 | 5.9 | 26.8 | 22.5 | 15.4 |
| VGGT | 10.9 | 81.7 | 18.8 | 21.0 | 15.0 | 16.6 |
| PI3 | 13.3 | 83.8 | 35.3 | 23.3 | 14.8 | 16.1 |
| DA3 | 18.7 | 86.4 | 46.1 | 32.1 | 16.2 | 20.8 |
| **Ours-1B** | **29.6** | **89.7** | **49.8** | **38.4** | **35.3** | **30.2** |
| **Ours-10B** | **36.4** | **92.5** | **56.3** | **43.7** | **40.0** | **36.4** |

| Method | 7Scenes AUC@30 | NRGBD AUC@30 | ETH3D AUC@30 | DyCheck AUC@30 | Sintel AUC@30 | TUM-Dyn AUC@30 |
|---|---:|---:|---:|---:|---:|---:|
| MegaSaM | 71.8 | 83.1 | 38.1 | 53.1 | 58.3 | 59.0 |
| VGGT | 74.4 | 97.7 | 62.1 | 78.7 | 50.0 | 61.2 |
| PI3 | 77.0 | 98.2 | 79.6 | 81.0 | 53.5 | 59.2 |
| DA3 | 78.2 | 98.4 | 87.0 | 83.9 | 52.7 | 62.7 |
| **Ours-1B** | **83.1** | **98.8** | **88.5** | **87.3** | **73.0** | **82.3** |
| **Ours-10B** | **88.2** | **99.1** | **90.4** | **90.9** | **79.1** | **87.5** |

**关键发现**：

- MegaSaM 在 Sintel 的严格动态位姿上强，但在 ETH3D 宽基线场景退化严重；
- feed-forward 方法在静态场景整体鲁棒，但旧模型对动态场景提升有限；
- VGGT-Ω 同时提升静态和动态场景，在 Sintel AUC@3 从 22.5 提到 40.0，相对提升 77%。

### Table 2：Depth Estimation

| Method | 7Scenes δ1.25 / AbsRel | NRGBD δ1.25 / AbsRel | ETH3D δ1.25 / AbsRel | DyCheck δ1.25 / AbsRel | Sintel δ1.25 / AbsRel | TUM-Dyn δ1.25 / AbsRel |
|---|---|---|---|---|---|---|
| MegaSaM | 93.8 / 0.065 | 96.2 / 0.057 | 94.8 / 0.083 | 97.4 / 0.042 | 74.1 / 0.207 | 92.9 / 0.083 |
| VGGT | 91.9 / 0.073 | 99.1 / 0.019 | 97.4 / 0.036 | 95.2 / 0.055 | 79.2 / 0.189 | 92.2 / 0.064 |
| PI3 | 92.8 / 0.068 | 99.2 / 0.011 | 99.6 / 0.016 | 97.4 / 0.041 | 82.5 / 0.144 | 95.5 / 0.046 |
| DA3 | 93.0 / 0.063 | 99.5 / 0.010 | 99.6 / 0.015 | 97.7 / 0.039 | 86.1 / 0.118 | 94.3 / 0.049 |
| **Ours-1B** | **94.6 / 0.058** | **99.6 / 0.010** | **99.8 / 0.012** | **98.4 / 0.038** | **89.5 / 0.097** | **97.4 / 0.041** |
| **Ours-10B** | **96.3 / 0.050** | **99.7 / 0.007** | **99.8 / 0.009** | **98.7 / 0.030** | **93.5 / 0.081** | **98.3 / 0.035** |

**关键发现**：

- VGGT-Ω 在静态 benchmark 上继续压低 AbsRel；
- 动态数据提升更明显，Sintel δ1.25 从 DA3 的 86.1 提到 93.5，AbsRel 从 0.118 降到 0.081；
- 10B 始终优于 1B，说明 scaling 对 camera 和 depth 都有效。

### Table 3：LIBERO VLA Benchmark

VGGT-Ω scene registers 作为冻结几何编码器，拼接进 OpenVLA-OFT 输入。

| Method | Spatial SR | Object SR | Goal SR | Long SR | Average SR |
|---|---:|---:|---:|---:|---:|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| TraceVLA | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **OpenVLA-OFT + Frozen Scene Tokens** | **99.3** | **99.2** | **99.0** | **96.7** | **98.5** |

**说明**：scene registers 可作为 plug-and-play 空间 token，提高 VLA 策略的空间理解和任务成功率。

### 消融实验

| 消融项 | 设置 | Point Error ↓ | 结论 |
|---|---|---:|---|
| Register Attention | 全 global attention | 0.071 | 精度略优但成本更高 |
| Register Attention | 25% global 替换为 register attention | 0.073 | 基本无损，FLOPs/显存更优 |
| Multi-task losses | 去掉 point + matching losses | 0.078 | 训练期多任务监督仍重要 |
| Multi-head setup | VGGT 原始多 dense heads | 0.070 | 精度略好但不利于 scaling |
| Self-supervision | 10% steps 换成 self-supervised | 0.070 vs 0.073 | 小幅提升，主要改善 OOD 泛化 |
| Annotation pipeline | 与 MegaSaM pseudo GT 对比 | 96.4 AUC@30 / 99.3 δ1.25 | 保守过滤得到更高质量 pseudo labels |

---

## 重要设计观察

### 1. Registers 是空间理解 token

论文最值得关注的点不只是几何性能，而是 scene registers 的表征价值：

- registers 能增强 VLA；
- registers 能与语言 embedding 对齐；
- registers 能作为 sequence-level 或 frame-level prediction 的接口；
- camera token 也可视为一种带直接监督的 register-like token。

对 VLN / VLA 的意义：重建模型不只是外部几何工具，还可以提供结构化空间 token，作为导航/操作模型的输入。

### 2. 多任务学习可以从“多输出头”转向“多训练损失”

VGGT 的经验是 point map、tracking、depth、camera 多任务预测互相增益。VGGT-Ω 进一步说明：

> 多任务监督收益不一定要求保留多 dense heads；可以只保留轻量输出头，在训练期通过几何一致性 loss 注入监督。

这对工程化非常关键，因为 dense heads 的 activation memory 比参数量更容易成为训练瓶颈。

### 3. 动态场景不一定需要显式动态输出

VGGT-Ω 不预测 motion mask / dynamic point map，但可处理动态场景。其前提是：

- 训练数据覆盖足够动态内容；
- 标注管线剔除不可靠动态区域；
- 合成数据补足动态像素深度；
- 模型容量足够学习动态先验。

这对我们做流式语义地图有启发：动态识别可以是后续 head / map layer 的任务，但基础几何模型不一定必须显式输出所有动态变量。

### 4. 数据质量比数据数量更关键

作者强调 pseudo-label generation 目标不是最大化 yield，而是保留高置信标注。对 3D 重建来说，错误监督会导致具体可复现的失败模式，例如：

- 人被吸到墙里；
- 透明玻璃/窗户深度不一致；
- thin structures 被抹掉；
- doming effect 造成全局弯曲；
- fake background 让模型学到错误远景。

这比 2D 分类/检测更敏感，因为几何标注错误会直接污染模型的空间先验。

### 5. DINOv3 初始化主要提升训练效率

作者发现不使用 DINO 初始化也能收敛到类似性能，但需要 4-8 倍更多训练迭代。DINOv3 的主要价值是降低优化难度、提高训练效率。

---

## 批判性思考

### 优点

1. **系统性 scaling 证据**：不是单点提升，而是展示模型规模、数据规模均可预测提升，说明前馈重建具备 foundation model 潜力。
2. **架构更适合大规模训练**：Register Attention 和单 dense head 把显存瓶颈从 head 层移走，使 15× 数据规模成为可能。
3. **动态场景能力强**：相比 VGGT、DA3、MegaSaM，在 Sintel、DyCheck、TUM-Dynamic 上相机和深度都有明显提升。
4. **scene registers 有下游价值**：对 VLA 和语言对齐的实验说明，几何重建 token 可以成为具身智能模型的空间表征。
5. **数据管线经验非常有价值**：论文对 3D 数据污染源、过滤策略、伪标签质量的讨论对后续工程实践指导意义很强。
6. **保持基础模型简洁**：作者没有为了榜单极致性能堆复杂 head，而是保留更干净的 base representation，便于社区扩展。

### 局限性

1. **训练资源极高**：核心实验使用 128 张 96GB H100，普通实验室难以复现完整训练。
2. **依赖内部大规模视频数据**：4000 万内部视频和内部合成/真实数据不可完全复现。
3. **仍非在线流式模型**：VGGT-Ω 仍是多帧 feed-forward reconstruction，不是 LingBot-Map 式 causal streaming memory；长视频需要另行设计窗口或缓存策略。
4. **动态建模是隐式的**：不输出 motion mask / dynamic object state，对下游需要显式动态地图的 VLN/VLA 任务仍需额外模块。
5. **极端运动和成像仍失败**：强 motion blur、FOV 剧变、强畸变、部分办公室多显示器场景会导致性能下降。
6. **敏感内容 mask 可能引入局部伪影**：训练数据中人脸、商标等被模糊或遮挡，可能导致局部深度不稳定。

### 潜在改进方向

1. **与 LingBot-Map 结合**：把 VGGT-Ω 的 DINOv3 + scene registers + register attention 迁移到 causal GCA 架构，形成真正流式版本。
2. **恢复轻量 camera iterative refinement**：在保持简洁的前提下，为导航场景提升位姿稳定性。
3. **显式动态地图 head**：增加 motion/object-change head，为 VLN 的动态场景更新提供结构化输出。
4. **开放词汇语义对齐**：利用 registers-language alignment，将 scene tokens 与导航指令、目标物体词表对齐。
5. **高分辨率与 thin structure 优化**：改进 thin objects、透明/反射区域和远景深度的 dense head。
6. **端侧 register-attention-only 变体**：虽然精度下降，但 FLOPs 仅 6%，适合探索轻量机器人部署。

### 可复现性评估

- [x] 论文和补充材料公开；
- [x] 项目主页公开；
- [x] 代码仓库公开；
- [x] 训练细节、loss 权重、数据过滤规则较完整；
- [ ] 完整训练数据不可完全复现，包含大量内部数据；
- [ ] 完整 10B 模型训练成本极高；
- [ ] 互联网视频标注管线依赖内部视频集合和 VLM 处理能力。

---

## 对当前 VLN / 语义地图方案的启发

### 1. DINOv3 + Registers 可作为地图 token 的依据

我们在 DINO-GeoSemMap 方案中设想了 `semantic token / object token / free-space token / change token`。VGGT-Ω 证明了 registers 能自然聚合场景级信息，并能被语言和 VLA 使用，因此这些 map tokens 不只是工程假设，而有论文证据支撑。

### 2. 几何预训练可服务语言导航

VGGT-Ω 的 VLA 实验说明，强几何模型输出的 scene tokens 可以直接提升具身策略。对 VLN 来说，可以考虑：

```text
RGB stream → VGGT-Ω / LingBot-style geometry encoder → scene/register tokens → VLN policy
```

而不是只把几何模型当成 depth/pose 工具。

### 3. 训练期多任务、推理期轻输出

对于检测、分割、深度、位姿一体模型，可以采用类似思想：

```text
训练期：camera + depth + point + matching + mask + semantic consistency
推理期：只保留 VLN 必需 heads
```

这样既能利用多任务监督，又不让部署模型过重。

### 4. 数据过滤必须成为方案核心

如果我们要用真实机器人 RGB 流训练语义地图模型，不能只强调数据规模。必须建立类似 VGGT-Ω 的数据质量标准：

- 动态区域 mask；
- 多视角一致性；
- 位姿平滑度；
- 视差充足性；
- 深度完整性；
- thin structure / transparent object 的特殊处理。

### 5. VGGT-Ω 不替代 LingBot-Map

VGGT-Ω 解决的是大规模 feed-forward reconstruction 和 representation scaling；LingBot-Map 解决的是在线 causal streaming 和长程 KV/memory 管理。对机器人在线 VLN，二者应结合：

```text
VGGT-Ω:
    更强 backbone、registers、dynamic prior、data scaling

LingBot-Map:
    流式 GCA、anchor/window/trajectory memory、实时长序列处理
```

---

## 关联笔记

### 基于

- [[VGGT]]：原始前馈式多视图重建 Transformer
- [[DINOv3]]：VGGT-Ω 的 ViT 初始化
- [[DINOv2]]：VGGT 的 tokenizer，对比 DINOv3 的变化
- [[DPT]]：dense depth decoder 基础
- [[Grounding-DINO]]：动态区域检测与过滤
- [[COLMAP]]：伪标签生成中的 BA 与重建过滤

### 对比

- [[MegaSaM]]：动态优化式重建，Sintel 强但宽基线/弱纹理退化
- [[Depth Anything 3]]：强 feed-forward 动态深度/相机模型，但重复纹理和强 roll 场景存在 ghosting
- [[PI3]]：动态场景 VGGT-style 方法
- [[MonST3R]]：动态 DUSt3R 扩展
- MapAnything：多场景几何基线

### 方法相关

- [[Alternating Attention]]：帧内与跨帧交替注意力
- [[Self-Attention]]：global attention / register attention 基础
- [[Multi-Task Learning]]：多任务训练损失
- [[Aleatoric Uncertainty]]：深度 confidence loss
- [[Knowledge Distillation]]：teacher-student 自监督训练
- [[Contrastive Loss]]：register-language InfoNCE 对齐
- [[Vision-Language-Action]]：scene registers 下游应用

### 数据/系统相关

- [[NVIDIA H100]]：训练硬件
- [[NVIDIA A100]]：推理速度与显存评估硬件
- [[ScanNet]]：数据质量问题案例
- [[MegaDepth]]：humans-in-walls 等伪标签问题来源之一

---

## 速查卡片

> [!summary] VGGT-Ω
> - **核心**：VGGT 的规模化版本，用 DINOv3、scene registers、Register Attention、单 dense head、多任务 losses 和大规模动态数据训练前馈 3D/4D 重建模型
> - **架构**：DINOv3 ViT + camera token + 16 scene registers + alternating attention + 25% register attention + depth head + camera head
> - **训练**：3M 公开/内部监督 sequences + 0.8M 高质量互联网视频标注 + 18M unlabeled videos self-supervised；240K iterations，128×H100
> - **结果**：Sintel camera AUC@3 从 22.5 提升到 40.0；depth δ1.25 从 86.1 提升到 93.5；OpenVLA-OFT + scene tokens 平均 SR 97.1 → 98.5
> - **关键启发**：scene registers 是可复用空间 token，几何重建可作为 VLN/VLA 的 scalable proxy task
> - **链接**：[arXiv](https://arxiv.org/abs/2605.15195) / [Project](https://vggt-omega.github.io/) / [Code](https://github.com/facebookresearch/vggt-omega)

---

*笔记创建时间: 2026-05-20*

