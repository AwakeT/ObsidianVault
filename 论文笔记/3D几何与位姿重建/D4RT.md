---
title: "Efficiently Reconstructing Dynamic Scenes One D4RT at a Time"
method_name: "D4RT"
authors: [Chuhan Zhang, Guillaume Le Moing, Skanda Koppula, Ignacio Rocco, Liliane Momeni, Junyu Xie, Shuyang Sun, Rahul Sukthankar, Joelle K. Barral, Raia Hadsell, Zoubin Ghahramani, Andrew Zisserman, Junlin Zhang, Mehdi S. M. Sajjadi]
year: 2026
venue: CVPR
tags: [dynamic-4d-reconstruction, point-tracking, feedforward-3d-reconstruction, query-based-decoder, camera-pose-estimation, video-depth-estimation]
zotero_collection: ""
image_source: online
arxiv_html: https://ar5iv.org/abs/2512.08924
created: 2026-06-16
---

# 论文笔记：Efficiently Reconstructing Dynamic Scenes One D4RT at a Time

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Google DeepMind, University College London, University of Oxford |
| 日期 | December 2025 / CVPR 2026 |
| 项目主页 | [Project Page](https://d4rt-paper.github.io/) |
| 对比基线 | [[MegaSaM]], [[VGGT]], [[pi3|π3]], [[SpatialTrackerV2]], [[DUSt3R]], [[CoTracker]] |
| 链接 | [arXiv](https://arxiv.org/abs/2512.08924) / [ar5iv HTML](https://ar5iv.org/abs/2512.08924) / [Project](https://d4rt-paper.github.io/) |

---

## 一句话总结

> D4RT 把动态场景的[[4D Reconstruction|4D 重建]]、[[Point Tracking|点跟踪]]、深度、点云和相机参数估计统一成“视频编码一次 + 任意时空点独立查询”的[[Query-Based Decoder|查询式解码]]问题，从而在动态视频上同时获得更高精度和更高吞吐。

---

## 核心贡献

1. **统一的点级 4D 查询接口**：给定视频 $V$，先编码为全局场景表示 $F$，再用查询 $q=(u,v,t_{src},t_{tgt},t_{cam})$ 直接预测任意源点在任意目标时间、任意相机坐标系下的 3D 位置。
2. **一个 decoder 覆盖多任务**：通过改变查询集合，可得到[[Point Cloud|点云]]、[[Video Depth Estimation|视频深度]]、3D 点轨迹、相机外参和内参，不再为每个任务维护独立 head。
3. **动态场景完整重建**：提出全像素密集跟踪算法，用 occupancy grid 避免 $O(T^2HW)$ 的暴力查询，在动态区域也能建立稠密时空对应。
4. **速度和精度同时领先**：在 TAPVid-3D、Sintel、ScanNet、KITTI、Bonn、Re10K 等评测中取得 SOTA 或 top-tier 表现；3D tracking 吞吐比已有方法快 18-300 倍。
5. **局部 RGB patch 保细节**：查询中加入 $9\times9$ local appearance patch，显著改善边界、毛发等高频细节，并支持原分辨率解码。

---

## 问题背景

### 要解决的问题
动态视频的几何理解不是静态 3D 重建的简单扩展：模型既要估计每一帧几何，又要知道同一个物理点在时间中的运动，还要处理相机运动、遮挡、动态物体和非刚体变化。传统 pipeline 往往把深度、相机、运动分割、点跟踪分开做，误差会在模块之间累积。

### 现有方法的局限
- [[MegaSaM]] 依赖多个外部模型和测试时优化，管线复杂且推理慢。
- [[VGGT]]、[[pi3|π3]] 等前馈 3D 重建方法已经很强，但通常无法直接给动态区域建立对应关系，或仍使用多个任务专用 decoder。
- [[SpatialTrackerV2]] 能处理动态点跟踪，但依赖迭代 refinement，吞吐低，并且主要从单帧出发跟踪，会导致第一帧不可见区域留下空洞。
- 多数方法对参考帧、相机内参变化、稀疏/稠密解码的灵活性支持不足。

### 本文动机
作者的核心判断是：动态 4D 场景并不一定需要“每帧密集输出 + 多任务 head”。只要视频 encoder 捕获了全局时空上下文，decoder 可以按需查询某个源像素在任意时间和参考坐标系中的 3D 位置。这个接口足够低层、足够通用，也足够并行。

---

## 方法详解

### 模型架构

D4RT 采用 **[[Vision Transformer|ViT]] 视频 encoder + 轻量 cross-attention pointwise decoder**：

- **输入**：视频 $V \in \mathbb{R}^{T \times H \times W \times 3}$
- **Encoder**：带 interleaved local frame-wise / global [[Self-Attention|self-attention]] 的 ViT，将视频编码成全局场景表示 $F \in \mathbb{R}^{N \times C}$
- **查询 token**：由归一化 2D 坐标 $(u,v)$ 的[[Fourier Features|Fourier feature]]、离散时间嵌入 $t_{src},t_{tgt},t_{cam}$、局部 RGB patch embedding 相加得到
- **Decoder**：小型[[Cross-Attention|cross-attention]] Transformer，每个查询独立 cross-attend 到 $F$
- **输出**：3D 点位置 $P \in \mathbb{R}^3$，附加预测包括 2D UV、可见性、位移、法向、置信度

### 核心模块

#### 模块1：全局场景表示
**设计动机**：动态重建需要知道整段视频中的时间流、遮挡关系和跨帧对应，因此不能只看单帧或局部邻域。

**具体实现**：
- 输入视频 resize 到固定正方形分辨率后 tokenization
- 额外加入原始宽高比 token，缓解 resize 对相机几何的影响
- 交替使用局部帧内 attention 和全局 attention，让表示同时包含空间细节和长程时序上下文

#### 模块2：独立查询式解码器
**设计动机**：如果每个 query 独立，训练时只需采样少量点即可获得监督；推理时可以按任务自由选择稀疏或稠密查询，并天然并行。

**具体实现**：
- 查询 $q=(u,v,t_{src},t_{tgt},t_{cam})$
- $(u,v,t_{src})$ 定义源点
- $t_{tgt}$ 定义要预测该点处于哪个目标时间状态
- $t_{cam}$ 定义输出 3D 坐标所在的相机参考系
- 不同 query 之间不做 self-attention；作者早期实验发现 query 间交互会带来 OOD 性能下降

#### 模块3：从查询到多任务输出

| 任务 | 查询方式 | 输出 |
|------|----------|------|
| Point Track | 固定 $(u,v,t_{src})$，遍历 $t_{tgt}=t_{cam}=1...T$ | 某个物理点的 3D 轨迹 |
| Point Cloud | 遍历所有像素与帧，固定参考帧 $t_{cam}$ | 全视频点云 |
| Depth Map | $t_{src}=t_{tgt}=t_{cam}$，遍历像素 | 每帧深度图 |
| Extrinsics | 同一批 3D 点在两个参考相机系下的坐标 | 通过[[Umeyama Algorithm|Umeyama]] 求相对位姿 |
| Intrinsics | 查询同帧 3D 点，结合 pinhole 几何 | 鲁棒估计 $f_x,f_y$ |

#### 模块4：高效密集动态对应

暴力跟踪所有像素需要 $O(T^2HW)$ 查询。D4RT 用 occupancy grid $G \in \{0,1\}^{T \times H \times W}$：

1. 从未访问像素中采样一批源点
2. 对每个源点查询完整视频轨迹
3. 将轨迹可见经过的时空像素标记为 visited
4. 重复直到所有像素被覆盖

这个策略利用时空冗余，在实验中得到 5-15 倍自适应加速。

---

## 关键公式

### 公式1：[[Global Scene Representation|全局场景表示]]

$$
F = E(V) \in \mathbb{R}^{N \times C}
$$

**含义**：视频 encoder $E$ 将输入视频 $V$ 编码为固定的全局场景表示 $F$，后续所有任务共享该表示。

**符号说明**：
- $V \in \mathbb{R}^{T \times H \times W \times 3}$：输入视频
- $F$：编码后的时空场景 token
- $N,C$：token 数量与通道维度

### 公式2：[[Query-Based Decoder|点级查询解码]]

$$
q = (u,v,t_{src},t_{tgt},t_{cam}), \qquad P = D(q,F) \in \mathbb{R}^3
$$

**含义**：decoder $D$ 根据查询 $q$ 和全局表示 $F$，预测源点在目标时间、目标相机坐标系中的 3D 位置。

**符号说明**：
- $(u,v)\in[0,1]^2$：源帧中的归一化 2D 坐标
- $t_{src}$：源点所在时间
- $t_{tgt}$：预测该点所处的目标时间
- $t_{cam}$：输出坐标所在的相机参考时间
- $P$：预测 3D 点坐标

### 公式3：[[Camera Pose Estimation|相机外参估计]]

$$
q_{i,k}=(u_k,v_k,i,i,i), \qquad q_{j,k}=(u_k,v_k,i,i,j)
$$

$$
\{D(q_{i,k},F)\}_k \xleftrightarrow{\text{Umeyama}} \{D(q_{j,k},F)\}_k
$$

**含义**：同一批源点在相机 $i$ 和相机 $j$ 坐标系下的 3D 坐标应该只差一个刚体变换，因此可用 Umeyama 算法求相对位姿。

### 公式4：[[Pinhole Camera Model|相机内参估计]]

$$
f_x = \frac{p_z(u-0.5)}{p_x}, \qquad
f_y = \frac{p_z(v-0.5)}{p_y}
$$

**含义**：在主点假设为 $(0.5,0.5)$ 的针孔模型下，由每个查询点的 3D 坐标反推出焦距，并对多个点取 median 提升鲁棒性。

### 公式5：[[Robust Loss|3D 点监督归一化]]

$$
\tilde{x}=\operatorname{sign}(x)\log(1+|x|)
$$

**含义**：预测和目标点集先按各自 mean depth 归一化，再用 log transform 降低远处点对 L1 损失的支配。

### 公式6：训练总损失

$$
\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N}
\left(
c\lambda_{3D}\mathcal{L}_{3D}
-\lambda_{conf}\log c
+\lambda_{2D}\mathcal{L}_{2D}
+\lambda_{vis}\mathcal{L}_{vis}
+\lambda_{disp}\mathcal{L}_{disp}
+\lambda_{conf}\mathcal{L}_{conf}
+\lambda_{normal}\mathcal{L}_{normal}
\right)_i
$$

**含义**：主损失是置信度加权的 3D 点 L1，同时加入 2D 投影、可见性、位移、置信度和法向等辅助监督。

---

## 关键图表

### Figure 1: D4RT Overview / 统一动态 4D 重建接口

![D4RT teaser](https://d4rt-paper.github.io/static/images/teaser_thumbnail.jpg)

**说明**：D4RT 用单一 encoder-decoder 框架从视频中输出点云、点轨迹和相机参数。关键不是 dense per-frame decoding，而是对全局场景表示进行独立时空查询。

### Figure 2: Model Overview / 模型查询机制

![D4RT pipeline](https://d4rt-paper.github.io/static/images/d4rt_pipeline.jpg)

**说明**：encoder 先得到全局场景表示 $F$；decoder 接收 $(u,v,t_{src},t_{tgt},t_{cam})$ 与局部 patch embedding，输出该点在目标时刻和目标相机坐标系下的 3D 位置。

### Figure 3: Pose Accuracy vs. Speed

**说明**：D4RT 在相机位姿估计上达到 200+ FPS，相比 [[VGGT]] 快 9 倍，相比 [[MegaSaM]] 快 100 倍，同时精度更高。

### Figure 4: Reconstruction Results Across Methods

**说明**：MegaSaM 和 π3 在动态场景中会重复或漏掉运动物体；SpatialTrackerV2 能跟踪动态点但只从单帧出发，存在遮挡空洞；D4RT 能重建视频中所有像素的完整 4D 表示。

### Figure 5: In-the-Wild Visualizations

**说明**：D4RT 在静态和动态公开视频上都能输出合理点云；动态场景中还能给出鲁棒的 3D 点轨迹。

### Figure 6: Local Patch Preserves Low-Level Details

**说明**：加入 local RGB patch 后，深度边界更清晰，细小结构更稳定；这是 D4RT 在不引入 DPT-style skip connection 的情况下保持细节的关键。

### Figure 7-13: Appendix Figures

**说明**：附录补充了完整模型图、KITTI 长序列拼接、高分辨率子像素解码、RGB patch size 消融，以及更多定性结果。它们共同说明 D4RT 的连续 query 坐标和 local RGB patch 可以在不提高 encoder 分辨率的情况下恢复更多高频细节。

### Table 1: Unified Decoding

| Task | $u$ | $v$ | $t_{src}$ | $t_{tgt}$ | $t_{cam}$ |
|------|-----|-----|-----------|-----------|-----------|
| Point Track | Fixed | Fixed | Fixed | $1...T$ | $1...T$ |
| Point Cloud | $1...W$ | $1...H$ | $1...T$ | $1...T$ | Fixed |
| Depth Map | $1...W$ | $1...H$ | $1...T$ | $1...T$ | $1...T$ |
| Extrinsics | $1...h$ | $1...w$ | Fixed | Fixed | $1...T$ |
| Intrinsics | $1...h$ | $1...w$ | $1...T$ | $1...T$ | $1...T$ |

### Table 2: Model Capabilities

| Model | 3D Recon. | Dynamic Corr. | Flexible Ref. Frames | Sparse Decoding | Global Context | Single Decoder |
|-------|-----------|---------------|----------------------|-----------------|----------------|----------------|
| MegaSaM | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| DUSt3R | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| VGGT | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| π3 | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ |
| St4RTrack | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| STv2 | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| **D4RT** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

### Table 3: 3D Tracking Throughput

| Method | 60 FPS | 24 FPS | 10 FPS | 1 FPS |
|--------|--------|--------|--------|-------|
| DELTA | 0 | 5 | 408 | 5,770 |
| SpatialTrackerV2 | 29 | 84 | 219 | 2,290 |
| **D4RT** | **550** | **1,570** | **3,890** | **40,180** |

**说明**：D4RT 的 decoder query 可并行，最大 track 数显著高于跟踪型方法。

### Table 4: 4D Reconstruction and Tracking

| Setting | Representative Result |
|---------|------------------------|
| Camera-coordinate 3D tracking, no GT intrinsics | D4RT 在 DriveTrack / ADT / PStudio 的综合表现最强，DriveTrack AJ 从 STv2 的 0.064 提升到 0.257 |
| Camera-coordinate 3D tracking, with GT intrinsics | D4RT 在 PStudio 明显领先，AJ=0.372, APD3D=0.498 |
| World-coordinate 3D tracking | D4RT 在 DriveTrack APD3D=0.470, L1=0.017；ADT APD3D=0.398, L1=0.093 |

### Table 5: Video Depth and Point Map Estimation

| Method | Point Cloud Sintel L1 ↓ | Point Cloud ScanNet L1 ↓ | Sintel AbsRel(S) ↓ | ScanNet AbsRel(S) ↓ | KITTI AbsRel(S) ↓ | Bonn AbsRel(S) ↓ |
|--------|--------------------------|---------------------------|--------------------|---------------------|-------------------|------------------|
| MegaSaM | 1.531 | 0.072 | 0.342 | 0.050 | 0.109 | 0.056 |
| VGGT | 1.582 | 0.063 | 0.318 | 0.044 | 0.094 | 0.055 |
| MapAnything | 1.718 | 0.064 | 0.397 | 0.043 | 0.096 | 0.076 |
| SpatialTrackerV2 | 1.375 | 0.036 | 0.209 | 0.027 | 0.075 | 0.042 |
| π3 | 1.139 | 0.030 | 0.241 | 0.021 | **0.055** | **0.033** |
| **D4RT** | **0.768** | **0.028** | **0.171** | **0.020** | **0.055** | 0.036 |

### Table 6: Camera Pose Estimation

| Method | Sintel ATE ↓ | ScanNet ATE ↓ | Re10K Pose AUC ↑ |
|--------|--------------|---------------|------------------|
| MegaSaM | 0.074 | 0.029 | 71.0 |
| VGGT | 0.168 | 0.016 | 70.2 |
| MapAnything | 0.202 | 0.023 | 68.7 |
| SpatialTrackerV2 | 0.126 | 0.018 | 75.7 |
| π3 | 0.086 | 0.015 | 78.7 |
| **D4RT** | **0.065** | **0.014** | **83.5** |

### Table 7: Local RGB Patch Ablation

| Local Patch | AbsRel(S) ↓ | AbsRel(SS) ↓ | ATE ↓ | RPE-T ↓ | RPE-R ↓ |
|-------------|-------------|--------------|-------|---------|---------|
| ✗ | 0.366 | 0.306 | 0.173 | 0.031 | 0.262 |
| ✓ | **0.302** | **0.257** | **0.091** | **0.028** | **0.245** |

### Table 8: Auxiliary Loss Ablation

| Setting | Depth AbsRel(S) Δ | ATE Δ | 结论 |
|---------|-------------------|-------|------|
| w/o 2D position | +0.071 | +0.002 | 2D 投影监督明显帮助深度 |
| w/o normal | +0.043 | +0.003 | 法向监督提升几何边界 |
| w/o displacement | +0.011 | +0.011 | 对深度和 pose 都有小幅收益 |
| w/o visibility | -0.003 | +0.012 | 对深度略有 trade-off，但 pose 变差 |
| w/o confidence | +0.002 | +0.126 | 置信度对 camera pose 很关键 |

### Table 9: Backbone Size

| Backbone | AbsRel(S) ↓ | AbsRel(SS) ↓ | ATE ↓ | RPE-T ↓ | RPE-R ↓ |
|----------|-------------|--------------|-------|---------|---------|
| ViT-B | 0.319 | 0.232 | 0.145 | 0.034 | 0.266 |
| ViT-L | 0.256 | 0.214 | **0.073** | 0.027 | 0.191 |
| ViT-H | 0.226 | 0.173 | 0.070 | 0.028 | 0.186 |
| ViT-g | **0.191** | **0.168** | 0.078 | **0.026** | **0.160** |

### Table 10: High-Resolution Decoding

| Config | Encoder Res. | RGB Patch | Output Res. | RGB Patch Res. | AbsRel(S) ↓ | PDBE(S) ↓ | AbsRel(SS) ↓ | PDBE(SS) ↓ |
|--------|--------------|-----------|-------------|----------------|-------------|-----------|--------------|------------|
| 1 | 256×256 | ✗ | 256×256 | 256×256 | 0.254 | 3.323 | 0.219 | 3.307 |
| 2 | 256×256 | ✓ | 256×256 | 256×256 | 0.218 | 2.254 | 0.179 | 2.243 |
| 3 | 256×256 | ✓ | Original | 256×256 | 0.217 | 2.266 | 0.178 | 2.258 |
| 4 | 256×256 | ✓ | Original | Original | 0.220 | **2.193** | **0.176** | **2.185** |

### Table 11: Model Initialization

| Init | AbsRel(S) ↓ | AbsRel(SS) ↓ | ATE ↓ | RPE-T ↓ | RPE-R ↓ |
|------|-------------|--------------|-------|---------|---------|
| None | 0.738 | 0.520 | 0.334 | 0.139 | 1.126 |
| VideoMAE | **0.302** | **0.257** | **0.091** | **0.028** | **0.245** |

---

## 实验

### 数据集

| 数据集 | 用途 | 特点 |
|--------|------|------|
| BlendedMVS | 训练 | 多视角重建数据 |
| Co3Dv2 | 训练 | 真实物体多视角 |
| Dynamic Replica | 训练 | 动态双目/深度 |
| Kubric | 训练 | 合成动态场景 |
| MVS-Synth | 训练 | 合成多视角 |
| PointOdyssey | 训练 | 长程点轨迹 |
| ScanNet++ / [[ScanNet]] | 训练/评测 | 室内 RGB-D / 3D 重建 |
| TartanAir | 训练 | 仿真驾驶/机器人环境 |
| Virtual KITTI / [[KITTI-2015|KITTI]] | 训练/评测 | 自动驾驶场景 |
| Waymo Open | 训练 | 真实驾驶视频 |
| TAPVid-3D | 评测 | 真实动态视频 3D 点跟踪 |
| Sintel | 评测 | 强动态与复杂渲染效果 |
| Bonn | 评测 | RGB-D 动态场景 |
| Re10K | 评测 | 相机 pose AUC |

### 实现细节

- **Encoder**：ViT-g，40 层，spatio-temporal patch size $2\times16\times16$
- **Decoder**：8 层 cross-attention，约 144M 参数
- **总参数量**：encoder 约 1B，decoder 约 144M
- **训练视频**：48-frame clips，分辨率 256×256
- **训练查询**：每个样本 decode 2048 random queries；30% oversample 在深度/运动边界
- **优化器**：[[AdamW]]，weight decay 0.03
- **训练步数**：500k steps
- **硬件**：64 TPU chips，本地 batch size 1，总训练约 2 天多

### 关键观察

1. **动态场景是 D4RT 的主场**：Sintel、TAPVid-3D 这类动态评测中，统一时空查询比纯重建方法优势更明显。
2. **速度优势来自 decoder 接口**：encoder 只跑一次，decoder 是轻量独立查询；因此 track 数提升时能近似并行扩展。
3. **local patch 是小设计但收益很大**：它让 query 不只是一个抽象坐标，还携带局部外观上下文，帮助对应匹配和边界定位。
4. **辅助任务不是装饰**：2D position、normal、visibility、confidence、displacement 分别稳定不同方面，其中 confidence 对 camera pose 影响最大。

---

## 批判性思考

### 优点
1. **接口非常干净**：一个 query 几乎把 4D 几何任务抽象到最小公因式，方法解释性很强。
2. **动态区域能力突出**：相比只做静态几何的 feedforward 重建，D4RT 明确建模跨时间点对应。
3. **推理可扩展**：稀疏查询、密集查询、原分辨率查询都可以在同一 decoder 下完成。
4. **实验覆盖面广**：跟踪、点云、深度、pose、速度、消融、长序列、高分辨率都有评估。

### 局限
1. **训练数据和算力门槛高**：需要大规模混合数据、VideoMAE 初始化、1B 级 backbone 和 64 TPU 训练。
2. **不是真正在线模型**：输入是一段 clip，encoder 依赖整段视频；实时流式 4D perception 还需要额外设计。
3. **相机内参估计假设较简化**：主公式默认 principal point 为 $(0.5,0.5)$，复杂畸变相机需要后续 refinement。
4. **尺度和坐标一致性仍依赖后处理**：长视频实验仍要 chunk overlap + Umeyama 对齐，尚未完全解决超长序列全局一致性。
5. **代码/权重可用性待确认**：项目页有结果展示，但笔记生成时未看到完整开源训练代码和模型权重。

### 潜在改进方向
1. **流式 D4RT**：为在线机器人/AR 场景设计 memory cache 或 sliding-window global representation。
2. **多相机/多传感器扩展**：将 LiDAR、IMU、事件相机等作为额外 query 条件或 encoder token。
3. **不确定性与物理约束**：把 confidence 扩展成更完整的几何不确定性，并加入刚体/非刚体运动先验。
4. **更轻量部署版本**：蒸馏 ViT-g encoder，保留查询式 decoder 的接口优势。

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型开源
- [x] 训练细节较完整
- [x] 数据集来源较完整
- [x] 核心算法可由论文复现

---

## 关联笔记

### 基于
- [[Vision Transformer]]：视频 encoder 的主干
- [[Cross-Attention]]：pointwise decoder 读取全局场景表示
- [[Scene Representation Transformer]]：D4RT 的 encoder-decoder 思路受其启发
- [[VideoMAE]]：encoder 初始化显著提升性能

### 对比
- [[VGGT]]：强大的 feedforward 3D 重建基线，但动态对应和单 decoder 统一性不足
- [[MegaSaM]]：依赖多模型与测试时优化，速度远慢于 D4RT
- [[pi3|π3]]：feedforward point map 重建强基线，但动态对应能力不足
- [[SpatialTrackerV2]]：动态跟踪强基线，但迭代式、吞吐低，稠密完整重建不如 D4RT
- [[DUSt3R]]：pairwise 3D 重建范式的代表，D4RT 进一步推广到视频 4D 查询

### 方法相关
- [[4D Reconstruction]]：论文核心问题
- [[Query-Based Decoder]]：核心架构接口
- [[Point Tracking]]：D4RT 输出点轨迹的任务形式
- [[Point Cloud]]：D4RT 通过遍历像素输出统一参考系点云
- [[Camera Pose Estimation]]：由同一点在不同相机参考系下的 3D 坐标对齐得到
- [[Video Depth Estimation]]：通过 $t_{src}=t_{tgt}=t_{cam}$ 的 query 得到

### 数据相关
- [[ScanNet]]
- [[KITTI-2015|KITTI]]
- [[TAPVid-3D]]
- [[Sintel]]
- [[Bonn]]

---

## 速查卡片

> [!summary] D4RT
> - **核心**：视频编码一次，任意点按 $(u,v,t_{src},t_{tgt},t_{cam})$ 独立查询 3D 位置。
> - **方法**：ViT 全局场景表示 + 轻量 cross-attention query decoder + local RGB patch。
> - **结果**：动态 4D tracking、点云、深度、相机 pose 多任务 SOTA/top-tier；3D track 吞吐快 18-300 倍。
> - **风险**：训练成本高，长视频仍需 chunk 对齐，流式应用还需改造。
> - **链接**：[Project](https://d4rt-paper.github.io/) / [arXiv](https://arxiv.org/abs/2512.08924)

---

*笔记创建时间: 2026-06-16*
