---
title: "3D Reconstruction with Spatial Memory"
method_name: "Spann3R"
authors: [Hengyi Wang, Lourdes Agapito]
year: 2024
venue: 3DV 2025
tags: [3d-reconstruction, spatial-memory, online-reconstruction, pointmap, dust3r, incremental-slam]
zotero_collection: 6-3D视觉
image_source: online
arxiv: https://arxiv.org/abs/2408.16061
arxiv_html: https://arxiv.org/html/2408.16061v1
created: 2026-07-02
---

# 论文笔记：3D Reconstruction with Spatial Memory

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[Spann3R]] |
| 机构 | University College London (UCL) |
| 日期 | 2024-08-28 (arXiv v1)；被 3DV 2025 接收 |
| 会议/来源 | 3DV 2025 |
| 项目主页 | [spanner](https://hengyiwang.github.io/projects/spanner) |
| 链接 | [arXiv](https://arxiv.org/abs/2408.16061) / [HTML](https://arxiv.org/html/2408.16061v1) |
| 相关工作 | [[DUSt3R]]、[[Point Map]]、[[Point Cloud Alignment]]、[[Global Optimization]]、[[3D Reconstruction]] |

---

## 一句话总结

> Spann3R 在 [[DUSt3R]] 基础上引入一个外部**空间记忆**，把逐图像对的局部 [[Point Map|pointmap]] 预测改成逐帧的全局 pointmap 预测，从而**去掉基于优化的全局对齐**，实现约 65 fps 的在线增量式稠密 3D 重建。

---

## 核心贡献

1. **基于空间记忆的在线重建范式**：提出 [[Spann3R]]，一个面向有序/无序图像集合的稠密 3D 重建方法，直接从图像回归 pointmap，无需相机参数或场景先验。
2. **外部空间记忆（[[Spatial Memory]]）**：网络学习记录所有历史相关 3D 信息，通过查询记忆来预测下一帧的几何结构。
3. **全局坐标系直接预测**：每帧 pointmap 直接表达在全局坐标系（首帧坐标系），**消除 DUSt3R 需要的 [[Global Optimization|基于优化的全局对齐]]**。
4. **实时性能**：有序序列在线模式约 **65 fps**（单张 4090，约 11 GB 显存），无需 test-time optimization。
5. **复用预训练权重**：直接复用 DUSt3R 预训练权重，仅在数据集子集上微调，在多个未见（unseen）数据集上展现有竞争力的性能与泛化能力。

---

## 问题背景

### 要解决的问题

从有序或无序的图像集合中做**稠密 3D 重建**，且希望能像 SLAM 一样**在线、实时、增量式**地进行，而不需要相机内外参先验。

### 现有方法的局限

- **[[DUSt3R]]** 对每个图像**对（pair）**回归 pointmap，且各自表达在**局部坐标系**中。
- 要得到全局一致的重建，DUSt3R 必须构建 pairwise 图并做 [[Global Optimization|基于优化的全局对齐]]，速度极慢（表中 DUSt3R FPS 仅 0.48–1.42），难以在线或实时。
- pairwise 组合数随视图数二次增长，不利于扩展到长序列。

### 本文动机

把逐对局部预测替换为**逐帧全局预测**：用一个学习到的空间记忆持续累积历史 3D 信息，让网络自己把新帧对齐到统一全局坐标系，从而彻底去掉优化环节——像扳手（spanner）一样"即时对齐"点云。

---

## 方法详解

### 总体架构

Spann3R 相对 [[DUSt3R]] 只增加"一个轻量记忆编码器 + 两个 MLP head"，并**复用（repurpose）**了 DUSt3R 的两个交织解码器：

- **输入**：图像帧 $I_t$（有序视频流或无序集合）。
- **视觉编码**：[[Vision Transformer|ViT-Large]] 编码器把帧 $I_t$ 编码为视觉特征 $f_t^I$。
- **记忆读出**：上一步查询特征 $f_{t-1}^Q$ 从记忆的 [[Cross-Attention|key/value]] 中检索，得到融合特征 $f_{t-1}^G$。
- **两个交织解码器（ViT-Base，[[Cross-Attention]] 联合处理）**：
  - **target decoder**：产生下一步的 query 特征 $f_t^Q$（用于查询记忆），同时输出一个仅用于监督的 pointmap/置信度。
  - **reference decoder**：吃融合后的记忆特征，输出用于重建的 pointmap $X_{t-1}$ 和置信度 $C_{t-1}$。
- **记忆编码**：由 reference decoder 特征 + 预测 pointmap 编码出记忆 key 与 value；因其同时含**几何 + 外观**特征，记忆读出可同时基于外观与距离。
- **输出**：全局坐标系下的逐帧 pointmap $X_t$。

**记忆编码器规格**：轻量 ViT，6 个 self-attention block，embedding dim 1024。

### 空间记忆（[[Spatial Memory]]）

由**稠密工作记忆（working memory）+ 稀疏长期记忆（long-term memory）+ 记忆查询机制**组成：

- **记忆查询**：[[Cross-Attention]]；训练时用 0.15 的 attention dropout，迫使模型从记忆值的子集推理几何；推理时对小权重做**硬裁剪阈值 $5\times10^{-4}$** 并重新归一化（去除离群 patch 干扰）。
- **工作记忆**：最近 **5 帧**的稠密特征；新特征仅当最大相似度 < 0.95 时插入；满时最旧特征流入长期记忆。
- **长期记忆**：受 [[XMem]] 启发；追踪每个 token 的累积注意力权重，达阈值时做**稀疏化**（只保留 top-k token）。消融显示 **4000 个记忆 token 对多数场景已足够**。

### 训练与推理

- **损失**：置信度回归损失 + scale 损失；预测与 GT pointmap 均按到原点平均距离归一化。直接固定首两视预测的尺度因离群点和室外场景（Co3D）无界性而效果不佳。
- **课程训练（curriculum）**：每序列随机采 5 帧（训练时记忆最多 4 帧）；采样窗口逐渐增大，最后 25% epoch 再减小，以对齐 train/inference 的帧间隔。
- **推理**：视频序列天然适配增量处理；无序集合则像 DUSt3R 建稠密 pairwise 图，用最高置信度对初始化，再用最小生成树/逐次挑选 next-best，view selection 用 sigmoid 形式的置信度提升鲁棒性。

---

## 关键公式

### 公式1：[[Vision Transformer|视觉编码]]

$$
f_t^I = \mathrm{Encoder}^I(I_t)
$$

**含义**：ViT-Large 把输入帧 $I_t$ 编码为视觉特征 $f_t^I$。

### 公式2：[[Spatial Memory|记忆读出]]

$$
f_{t-1}^G = \mathrm{Memory\_read}(f_{t-1}^Q, f^K, f^V)
$$

**含义**：用上一帧的查询特征 $f_{t-1}^Q$ 从记忆 key $f^K$、value $f^V$ 中检索历史 3D 信息，得到融合特征 $f_{t-1}^G$。

### 公式3：[[Cross-Attention|交织解码]]

$$
f_t^{H'},\ f_{t-1}^H = \mathrm{Decoder}(f_t^I, f_{t-1}^G)
$$

**含义**：两个交织解码器联合处理当前视觉特征与融合记忆特征，分别输出 target 分支特征 $f_t^{H'}$ 与 reference 分支特征 $f_{t-1}^H$。

### 公式4：查询/预测头

$$
f_t^Q = \mathrm{head}_{\mathrm{query}}^{\mathrm{target}}(f_t^{H'}, f_t^I)
$$

$$
X_{t-1},\ C_{t-1} = \mathrm{head}_{\mathrm{out}}^{\mathrm{ref}}(f_{t-1}^H)
$$

**含义**：target 头输出下一步查询特征 $f_t^Q$；reference 头输出全局坐标系 pointmap $X_{t-1}$ 与置信度 $C_{t-1}$。

### 公式5：[[Spatial Memory|记忆编码]]

$$
f_{t-1}^K = \mathrm{head}_{\mathrm{key}}^{\mathrm{ref}}(f_{t-1}^H, f_{t-1}^I)
$$

$$
f_{t-1}^V = \mathrm{Encoder}^V(X_{t-1}) + f_{t-1}^K
$$

**含义**：由 reference 解码特征 + 视觉特征编码记忆 key；再由预测 pointmap 编码几何 value 并叠加 key，使记忆同时携带外观与几何。

### 公式6：[[Cross-Attention|注意力融合]]

$$
f_{t-1}^G = A_{t-1} f^V + f_{t-1}^Q
$$

$$
A_{t-1} = \mathrm{Softmax}\!\left(\frac{f_{t-1}^Q (f^K)^\top}{\sqrt{C}}\right)
$$

**含义**：注意力权重 $A_{t-1}$ 由查询与记忆 key 的缩放点积 softmax 得到，加权 value 后与 query 残差相加得到融合特征。

**符号说明**：

- $f^K, f^V \in \mathbb{R}^{Bs\times(T\cdot P)\times C}$：记忆中 $T$ 帧、每帧 $P$ 个 patch 的 key/value。
- $f_{t-1}^Q \in \mathbb{R}^{Bs\times P\times C}$：当前查询。
- $A_{t-1} \in \mathbb{R}^{Bs\times P\times(T\cdot P)}$：注意力图。
- $C$：通道维。

### 公式7：[[Point Map|置信度回归损失]]

$$
\mathcal{L} = \mathcal{L}_{\mathrm{conf}} + \mathcal{L}_{\mathrm{scale}}
$$

$$
\mathcal{L}_{\mathrm{conf}} = \sum_t \sum_{i\in\mathcal{V}} C_t^i \mathcal{L}_{\mathrm{reg}}(i) - \alpha \log C_t^i,
\qquad
C_t^i = 1 + \exp(\hat{C}_t^i)
$$

**含义**：延续 DUSt3R 的置信度加权回归损失；$\alpha$ 控制置信度分布，$\mathcal{V}$ 为有效像素集，$C_t^i \ge 1$。

### 公式8：[[Point Map|尺度损失]]

$$
\mathcal{L}_{\mathrm{scale}} = \max(0,\ \bar{X} - \bar{X}_{\mathrm{gt}})
$$

**含义**：$\bar{X}$ 为预测点到原点的平均距离，惩罚预测尺度大于真值尺度，抑制尺度漂移。

### 公式9：[[Spann3R|课程采样帧间隔]]

$$
T = T_{\mathrm{min}} + \eta_a(T_{\mathrm{max}} - T_{\mathrm{min}})
$$

$$
\eta_a = \begin{cases} \min(1, 2\eta) & \eta<0.75 \\ \max(0.5, 4-4\eta) & \text{otherwise} \end{cases}
$$

**含义**：训练进度比 $\eta$ 控制采样帧间隔 $T$：先增大窗口学习远距关联，后期缩小以对齐推理时的帧间隔。

### 公式10：[[Spann3R|view selection 置信度]]

$$
C = \frac{C_1-1}{C_1} + \frac{C_2-1}{C_2}
$$

**含义**：无序集合下用 sigmoid 形式的置信度衡量视图对质量，比指数形式更鲁棒地选择 next-best view。

---

## 关键图表

> 图表来源：论文 arXiv HTML v1。共 10 张 Figure、8 张 Table。图片 URL 前缀为 `https://arxiv.org/html/2408.16061v1/`。

### Figure 1: Overview / Teaser

![Figure 1](https://arxiv.org/html/2408.16061v1/extracted/5819331/Figures/teaser/fig_teaser_v6.png)

**说明**：在统一坐标系下逐帧回归 pointmap 做增量重建，无需优化对齐，实时在线。

### Figure 2: Motivation

![Figure 2](https://arxiv.org/html/2408.16061v1/x1.png)

**说明**：[[DUSt3R]] 逐对局部坐标预测 vs. Spann3R 经[[Spatial Memory|空间记忆]]直接预测全局 pointmap。

### Figure 3: Overview of Spann3R

![Figure 3](https://arxiv.org/html/2408.16061v1/extracted/5819331/Figures/method/pipeline7.png)

**说明**：ViT 编码器 + 两交织解码器（target=query，reference=预测）+ 轻量记忆编码器的整体流水线。

### Figure 4: Overview of Spatial Memory

![Figure 4](https://arxiv.org/html/2408.16061v1/extracted/5819331/Figures/method/fig_mem.png)

**说明**：稠密工作记忆 + 稀疏长期记忆；通过注意力权重累积直方图管理长期记忆的稀疏化。

### Figure 5: Qualitative Examples

![Figure 5](https://arxiv.org/html/2408.16061v1/extracted/5819331/Figures/qual/fire04_ours01.png)

**说明**：fire-04 / redkit-06 / office-09 场景，列为 frame / DUSt3R / Ours / GT 的定性对比。

### Figure 6: Online Reconstruction

![Figure 6](https://arxiv.org/html/2408.16061v1/extracted/5819331/Figures/map_v2/ma01.png)

**说明**：两个室内场景（MA / Kit.）的在线重建，含 Manhattan 世界假设与闭环误差可视化。

### Figure 7: Ablation on Memory Size

![Figure 7](https://arxiv.org/html/2408.16061v1/extracted/5819331/Figures/memory/longterm_v2.png)

**说明**：Chamfer distance 随长期记忆最大 token 数变化；约 4000 token 后收益饱和。

### Figure 8: Visualization of the Attention Map

![Figure 8](https://arxiv.org/html/2408.16061v1/extracted/5819331/Figures/affinity/q4.png)

**说明**：注意力对视觉相似的 patch（如右眼/双脚）具有鲁棒性，说明记忆同时利用外观与几何。

### Figure 9: Qualitative Examples on Real-world Datasets

![Figure 9](https://arxiv.org/html/2408.16061v1/extracted/5819331/Figures/qual2/qual_gen_v4.png)

**说明**：Map-free Reloc / ETH3D / MipNeRF-360 / NeRF / TUM-RGBD 上的泛化定性结果。

### Figure 10: Outlier Scene on NRGBD

![Figure 10](https://arxiv.org/html/2408.16061v1/extracted/5819331/Figures/supp_gwr/gwr_ours_key.png)

**说明**：含镜面的 GWR 场景产生离群点（Acc：DUSt3R† 0.1096 / Ours 0.2074），是本文的失败案例。

### Table 1: 7Scenes & NRGBD 重建 + FPS

指标 Acc↓ / Comp↓（Mean/Med）、NC↑（Mean/Med）；`†` = DUSt3R 224 权重，`FV` = 8 帧间隔的 pair，`Onl` = online。

| Method | Optim/Onl | 7S Acc M | 7S Comp M | 7S NC M | NRGBD Acc M | NRGBD Comp M | NRGBD NC M | FPS |
|---|---|---|---|---|---|---|---|---|
| F-Recon | Optim | 0.1243 | 0.0554 | 0.6189 | 0.2855 | 0.1505 | 0.6547 | <0.1 |
| DUSt3R† | Optim | 0.0286 | 0.0280 | 0.6681 | 0.0544 | 0.0315 | 0.8024 | 0.78 |
| **Ours** | Onl | 0.0342 | 0.0241 | 0.6635 | 0.0691 | 0.0291 | 0.7775 | **65.49** |
| DUSt3R (FV) | Optim | 0.0188 | 0.0234 | 0.7851 | 0.0392 | 0.0342 | 0.8765 | 0.48 |
| DUSt3R† (FV) | Optim | 0.0279 | 0.0276 | 0.7630 | 0.0591 | 0.0409 | 0.8305 | 1.42 |
| Ours⋆ (FV) | — | 0.0233 | 0.0246 | 0.7791 | 0.0587 | 0.0390 | 0.8384 | 5.83 |
| **Ours (FV)** | Onl | 0.0239 | 0.0247 | 0.7768 | 0.0611 | 0.0392 | 0.8330 | **72.04** |

**表格说明**：在线模式重建精度与需优化对齐的 DUSt3R† 相当，但 FPS 快约 80–90 倍（65 vs 0.78）。

### Table 2: DTU 重建

| Method | Opt/Onl | Acc M | Comp M | NC M |
|---|---|---|---|---|
| DUSt3R | Opt | 2.114 | 2.033 | 0.749 |
| DUSt3R† | Opt | 2.296 | 2.158 | 0.747 |
| Ours⋆ | — | 2.902 | 2.120 | 0.732 |
| Ours | Onl | 4.785 | 2.743 | 0.721 |
| Ours (FV) | Onl | 3.375 | 2.870 | 0.777 |

**表格说明**：DTU 为室外/物体级、更 OOD，online 模式差距明显，说明泛化仍有边界。

### Table 3: 空间记忆消融（7Scenes）

| 变体 | Acc M | Comp M | NC M |
|---|---|---|---|
| w/o lm（仅工作记忆） | 0.2554 | 0.1470 | 0.5964 |
| w/o clip（无注意力裁剪） | 0.0349 | 0.0249 | 0.6627 |
| Full | **0.0342** | **0.0241** | **0.6635** |

**表格说明**：去掉长期记忆性能急剧下降（Acc 0.2554 vs 0.0342），证明长期记忆是关键组件。

### Table 4: View Selection 消融（DTU）

| 变体 | Acc M | Comp M | NC M |
|---|---|---|---|
| Ours⋆ (exp) | 3.099 | 2.247 | 0.731 |
| Ours⋆ (sigmoid) | **2.902** | **2.120** | 0.732 |

**表格说明**：sigmoid 置信度形式的 view selection 优于指数形式。

### Table 5–6: DUSt3R_ours 微调消融

在室内（7Scenes/NRGBD）与 DTU 上比较 DUSt3R†、微调后的 DUSt3R_ours、Ours。结论：微调仅带来小幅变化，Spann3R 主要增益来自空间记忆结构而非单纯微调。

### Table 7–8: 逐场景结果（7Scenes / NRGBD）

逐场景显示多数场景 Ours 的 Comp 更优；镜面反射场景（office-09、NRGBD 的 GWR）显著变差，对应漂移与离群点。均值：

| Dataset | Method | Acc M | Comp M | NC M |
|---|---|---|---|---|
| 7Scenes | DUSt3R† | 0.0286 | 0.0280 | 0.6681 |
| 7Scenes | Ours | 0.0342 | 0.0241 | 0.6635 |
| NRGBD | DUSt3R† | 0.0544 | 0.0315 | 0.8024 |
| NRGBD | Ours | 0.0691 | 0.0291 | 0.7775 |

---

## 实验

### 数据集与设置

| 项目 | 设置 |
|------|------|
| 初始化 | [[DUSt3R]] 预训练权重（ViT-Large 编码器、ViT-Base 解码器、DPT head）|
| 训练数据 | Habitat 子集、ScanNet、ScanNet++、ARKitScenes、BlendedMVS、Co3D-v2 |
| 分辨率 | 224 × 224 |
| 优化器 | AdamW，lr 5e-5，β=(0.9, 0.95) |
| 训练轮数 | 120 epoch |
| 硬件 | 8× V100 (32GB)，约 10 天 |
| 评测（全 unseen） | 7Scenes、NRGBD、DTU（单张 4090 24GB）|
| 运行时 | 默认 online 约 65 fps，约 11 GB 显存 |

### 关键结果

- 在线模式下重建精度与需优化对齐的 DUSt3R **相当**，但速度快约两个数量级（65 fps vs <1 fps）。
- 长期记忆消融证明其对全局一致性至关重要（去掉后 Acc 从 0.0342 恶化到 0.2554）。
- DTU（室外/物体级 OOD）在线模式退化明显，泛化边界可见。

---

## 批判性思考

### 优点

1. **首个把 DUSt3R 做成在线增量式**：用空间记忆替代优化对齐，两个数量级的速度提升。
2. **改动极小**：复用 DUSt3R 权重与解码器，只加轻量记忆编码器，工程代价低。
3. **记忆分级设计合理**：工作记忆 + 长期记忆 + 稀疏化，兼顾容量与效率。
4. **记忆同时含外观与几何**：注意力对视觉相似 patch 鲁棒。

### 局限

1. **无 bundle adjustment → 漂移与闭环误差**：长序列、多房间、前向运动场景会失败。
2. **镜面/反射场景产生离群点**（Fig.10 GWR），几何一致性差。
3. **依赖有位姿的 RGB-D 训练数据**，训练成本高。
4. **泛化边界**：DTU 等 OOD 室外物体级场景在线模式退化明显。

### 潜在改进方向

1. 引入轻量在线 BA / 位姿图优化缓解漂移与闭环。
2. 更强的长期记忆管理（更全局的场景表示）以支持大尺度场景。
3. 显式处理镜面/反射区域的离群点。

---

## 关联笔记

### 基于

- [[DUSt3R]]：Spann3R 直接复用其权重与解码器，把成对局部预测改为逐帧全局预测。
- [[Point Map]]：pointmap 表示是本文的输出形式。

### 对比

- [[Global Optimization]]：DUSt3R 需要的全局对齐优化，正是本文要消除的对象。
- [[MASt3R]]：同谱系的成对匹配/重建方法。

### 方法相关

- [[Spatial Memory]]：本文核心组件，外部空间记忆。
- [[Cross-Attention]]：记忆读写机制。
- [[XMem]]：长期记忆稀疏化的灵感来源。
- [[3D Reconstruction]]：任务范畴。

### 谱系相关

- [[MUSt3R]]、[[CUT3R]]、[[TTT3R]]：DUSt3R 谱系的其他增量式/在线 3D 重建工作。

---

## 速查卡片

> [!summary] Spann3R
> - **核心**：DUSt3R + 外部空间记忆，逐帧全局 pointmap，去掉优化对齐
> - **记忆**：工作记忆（5 帧）+ 长期记忆（稀疏化，约 4000 token）
> - **模型**：ViT-L 编码器 + 两交织 ViT-B 解码器 + 轻量记忆编码器（6 block）
> - **关键结论**：在线精度≈DUSt3R，速度快 ~80×（65 fps）
> - **结果**：7Scenes Acc 0.0342 / NRGBD Acc 0.0691（online）
> - **局限**：无 BA → 漂移/闭环误差；镜面场景离群
> - **项目**：hengyiwang.github.io/projects/spanner

---

*笔记创建时间: 2026-07-02*
