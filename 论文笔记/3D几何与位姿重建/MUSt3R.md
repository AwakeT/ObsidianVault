---
title: "MUSt3R: Multi-view Network for Stereo 3D Reconstruction"
method_name: "MUSt3R"
authors: [Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, Vincent Leroy]
year: 2025
venue: CVPR 2025
tags: [3d-reconstruction, multi-view, pointmap, memory-mechanism, visual-odometry, dust3r]
zotero_collection: 6-3D视觉
image_source: online
arxiv: https://arxiv.org/abs/2503.01661
arxiv_html: https://arxiv.org/html/2503.01661v1
created: 2026-07-02
---

# 论文笔记：MUSt3R: Multi-view Network for Stereo 3D Reconstruction

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[MUSt3R]] |
| 机构 | Naver Labs Europe |
| 日期 | 2025-03-03 (arXiv v1) |
| 会议/来源 | CVPR 2025 |
| 代码 | [github.com/naver/must3r](https://github.com/naver/must3r) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.01661) / [HTML](https://arxiv.org/html/2503.01661v1) |
| 相关工作 | [[DUSt3R]]、[[MASt3R]]、[[Spann3R]]、[[Point Map]]、[[CroCo v2]]、[[Camera Pose Estimation]] |

---

## 一句话总结

> MUSt3R 把 [[DUSt3R]] 的**两视图**架构扩展为**对称的多视图（N-view）** 网络，配合**多层记忆机制**，直接为所有视图预测统一坐标系下的 pointmap，支持离线与在线重建，在无标定 [[Visual Odometry|VO]]、相对位姿、深度估计上达到 SOTA 且推理高速。

---

## 核心贡献

1. **对称架构（symmetric architecture）**：将 DUSt3R 的非对称成对结构改为**共享权重的对称 Siamese 解码器**，使网络能在**度量空间（metric space）** 中直接做 N 视图预测，所有视图统一坐标系。
2. **多层记忆机制（[[Memory Mechanism]]）**：类比 KV-cache 的因果记忆，降低计算复杂度，支持离线与在线重建，可高帧率推断数千张 pointmap。
3. **SOTA 且不牺牲速度**：在深度、相对位姿、3D 重建、无标定 VO 上均达先进水平，比 DUSt3R-512-GA 快约 40 倍。

---

## 问题背景

### 要解决的问题

在无相机标定、无视角先验的前提下，对**大规模图像集合**做稠密无约束的 3D 重建，且能**离线与在线**兼容，适用于 SfM 与 visual SLAM。

### 现有方法的局限

- **[[DUSt3R]]** 是二元（two-view）架构，只处理图像**对**。
- N 张图像需要 $O(N^2)$ 个图像对推理，再靠 [[Global Optimization|全局对齐 (GA)]] 把局部 pointmap 拼到全局坐标系——开销大、慢（DUSt3R-512-GA 仅 0.1 FPS）。
- [[Spann3R]] 虽用空间记忆做在线，但长序列漂移严重（TUM ATE 平均 47.9cm）。

### 本文动机

让单一网络一次性输出**所有视图在共同坐标系下的 pointmap**，并把类 KV-cache 的记忆思想引入解码器，做到可扩展、在线、高帧率。

---

## 方法详解

### 从 DUSt3R（二元基线）出发

- Siamese [[Vision Transformer|ViT]] 编码器 → latent $E_i$；投影 $D_i^0 = \mathrm{Lin}(E_i)$；$L$ 个交织解码块带 [[Cross-Attention]]；两个预测头。
- 把图像对映射到共享坐标系下的 pointmaps $\{X_{i,1}\}$，$i\in\{1,2\}$。

### MUSt3R 多视图架构

- **对称 Siamese 解码器（共享权重）**：两分支共享权重，解码器可训练参数减半。
- **可学习嵌入 $B$**：用来标识参考图像 $I_1$（否则对称结构会崩溃，见消融）。
- **移除 cross-attention 中的 [[RoPE]]**（无损）。
- 新增 **$X_{i,i}$**（局部 pointmap，在各自相机坐标系）输出，与 **$X_{i,1}$**（全局，在参考帧坐标系）一起用于恢复位姿。
- **相对位姿**通过 $X_{i,i}$ 与 $X_{i,1}$ 之间的 [[Procrustes Analysis|Procrustes 分析]]求解，比 [[PnP]] 更简单更快。

### 引入因果性（Causality）与记忆

- **迭代记忆（iterative memory）**：逐层保存已计算的 tokens $M_n^l$；新图像 cross-attend 已存 tokens。
- **KV-cache 类比**：类似因果 transformer 推理的 KV cache——新图像 attend 过去，过去不被更新。
- **Rendering（渲染）**：处理一张图像但**不把其 tokens 加入记忆**，可"打破因果性"，用未来帧重新计算 pointmap（视频结尾做）。
- **Global 3D Feedback**：用倒数第二层（$l=L-1$）信息增强记忆 tokens，$\mathrm{Inj}^{3D}$ = LayerNorm + 两层 MLP。

### 记忆管理

同一网络支持两种场景：

- **在线（online，视频顺序帧）**：运行时维护记忆 + 3D 场景；场景以 **KDTrees** 存储，每个 3D 点按视线方向分配到球面"八分体（octants）"。**Discovery rate（发现率）**：归一化距离的第 $p$ 百分位超过阈值 $\tau_d$ 则把该帧加入记忆（VO 超参 $\tau_d=5\%$、$p_d=85\%$）。
- **离线（offline，无序集合）**：用编码器特征 $E_i$ 做 **ASMK** 图像检索；**最远点采样**选关键帧；从连接度最高关键帧起贪心重排序，送入构建 latent 场景表示，再把所有图像从记忆中 render 出来。

---

## 关键公式

### 公式1：[[Point Map|DUSt3R 预测头]]

$$
X_{i,1},\ C_i = \mathrm{Head}_i^{3D}(E_i, D_i^L)
$$

**含义**：由编码器 latent $E_i$ 与末层解码输出 $D_i^L$ 回归参考帧坐标系下的 pointmap $X_{i,1}$ 与置信度 $C_i$。

### 公式2：[[Point Map|逐像素回归损失]]

$$
\ell_{regr}(i,j) = \sum_{p \in I_i} \left\| \frac{1}{z} X_{i,j}[p] - \frac{1}{\hat{z}} \hat{X}_{i,j}[p] \right\|
$$

**含义**：$j=1$ 为参考视图；$z,\hat{z}$ 为尺度归一化因子（有效点到原点平均距离），度量数据时令 $z:=\hat{z}$；外层包裹置信度损失 $\mathcal{L}_{conf}$。

### 公式3：[[MUSt3R|参考嵌入]]

$$
D_2^0 = \mathrm{Lin}(E_2) + B
$$

**含义**：可学习嵌入 $B$ 标识参考图 $I_1$，是对称解码器不崩溃的关键。

### 公式4：多视图 cross-attention

$$
D_i^l = \mathrm{Dec}^l(D_i^{l-1}, M_{n,-i}^{l-1})
$$

**含义**：其中 $M_n^l = \mathrm{Cat}_N(D_1^l,\ldots,D_n^l)$；$M_{n,-i}^l$ 为排除图像 $i$ 后的记忆。每张图像对其余所有图像的记忆做 cross-attention。

### 公式5：多视图输出（含局部 pointmap）

$$
(X_{i,1},\ X_{i,i},\ C_i) = \mathrm{Head}^{3D}(D_i^L),\quad i\in\{1\ldots n\}
$$

**含义**：同时输出全局坐标系 $X_{i,1}$ 与局部坐标系 $X_{i,i}$ 两个 pointmap，后者用于恢复位姿。

### 公式6：[[Memory Mechanism|记忆更新（因果推理）]]

$$
D_{n+1}^l = \mathrm{Dec}^l(D_{n+1}^{l-1}, M_n^{l-1})
$$

**含义**：新图 $n+1$ 只 cross-attend 已保存的记忆 $M_n^l$，过去 tokens 不被更新（类 KV-cache）。

### 公式7：[[MUSt3R|Global 3D Feedback]]

$$
\bar{D}_i^l = \begin{cases}
D_i^l + \mathrm{Inj}^{3D}(D_i^{L-1}), & \forall l < L-1 \text{ and } i \in \mathcal{P} \\
D_i^l, & l = L-1 \text{ or } i \in \mathcal{N}
\end{cases}
$$

**含义**：$\mathcal{P}$ = 先前图像，$\mathcal{N}$ = 新图像；用倒数第二层 3D 信息回注增强记忆。消融显示从 $L-1$ 层注入显著优于从最终层注入。

### 公式8–10：[[Point Map|log-space 损失]]

$$
f: x \to \frac{x}{\|x\|} \log(1 + \|x\|)
$$

$$
X'_{i,j}[p] = f\!\left(\tfrac{1}{z} X_{i,j}[p]\right),\quad \hat{X}'_{i,j}[p] = f\!\left(\tfrac{1}{\hat z} \hat{X}_{i,j}[p]\right)
$$

$$
\ell_{regr}(i,j) = \sum_{p \in I_i} \left\| X'_{i,j}[p] - \hat{X}'_{i,j}[p] \right\|
$$

**含义**：把归一化 pointmap 送入对数映射 $f$ 再算 L2，超过 10 视图时比 MASt3R 默认损失更稳。

### 公式11：[[MUSt3R|总训练损失]]

$$
\mathcal{L} = \sum_{i=1}^{n+N} \ell_{regr}(i,1) + \ell_{regr}(i,i)
$$

**含义**：同时监督全局 $(i,1)$ 与局部 $(i,i)$ pointmap；$n$ 为记忆图像数，$N$ 为额外 rendered 图像数。

---

## 关键图表

> 图表来源：论文 arXiv HTML v1。共 8 张主 Figure、12 张 Table。图片 URL 前缀 `https://arxiv.org/html/2503.01661v1/`。

### Figure 1: Qualitative Reconstruction

![Figure 1](https://arxiv.org/html/2503.01661v1/x1.jpg)

**说明**：Aachen Day-Night 序列（离线，上）与 TUM-RGBD Freiburg1-room 序列（在线，下）的重建定性示例。

### Figure 2: Framework Overview

![Figure 2](https://arxiv.org/html/2503.01661v1/x3.png)

**说明**：无标定重建框架总览。网络预测局部 $X_{i,i}$ 与全局 $X_{i,1}$ pointmap，据此高效恢复焦距、深度图、位姿与稠密 3D；记忆按启发式选择性更新。

### Figure 3: Architecture (Memory Usage)

![Figure 3](https://arxiv.org/html/2503.01661v1/x5.png)

**说明**：解码器深度 $L=3$、线性 Head、无 Inj³ᴰ 时的架构。左：两图初始化；右：给定新图/帧时记忆的使用与更新。

### Figure 4: 3D Feedback Module

![Figure 4](https://arxiv.org/html/2503.01661v1/x7.png)

**说明**：$L=3$ 时的 [[MUSt3R|Global 3D Feedback]] 模块结构。

### Figure 5: Cambridge Landmarks

![Figure 5](https://arxiv.org/html/2503.01661v1/x8.jpg)

**说明**：Cambridge Landmarks 数据集的重建定性示例。

### Figure 6: MIP-360

![Figure 6](https://arxiv.org/html/2503.01661v1/x13.jpg)

**说明**：MIP-360 数据集重建定性示例。

### Figure 7: ScanNet++ / RealEstate10K / TUM-RGBD

![Figure 7](https://arxiv.org/html/2503.01661v1/x22.jpg)

**说明**：ScanNet++（含变焦）、RealEstate10K、TUM RGB-D 三类数据的重建对比。

### Figure 8: Ablation Curves

![Figure 8](https://arxiv.org/html/2503.01661v1/x28.png)

**说明**：随记忆中视图数变化的中位绝对回归误差。左：不同 Inj³ᴰ 注入架构；右：log 损失 vs 默认 MASt3R 损失。

### Table 1: 无标定 VO（TUM-RGB SLAM，ATE RMSE [cm]）

| 类 | Method | Avg ATE | FPS |
|---|---|---|---|
| U | [[Spann3R]] | 47.9 | 4.8 |
| U | MUSt3R-C（causal） | 7.1 | 11.1 |
| U | **MUSt3R** | **5.5** | 8.4 |
| D | DROID-VO | 11.4 | — |
| D | GlORIE-VO* | 9.3 | — |
| S | DPVO | 8.4 | — |

**表格说明**：MUSt3R 平均 ATE 5.5cm，超越所有稠密无约束方法（U），且优于多数稠密（D）/稀疏（S）SLAM；比 Spann3R 好近 9 倍。

### Table 6: 3D 重建（7Scenes / NRGBD / DTU）+ 速度显存

| Method | 7S Acc | NRGBD Acc | DTU Acc | FPS | Mem |
|---|---|---|---|---|---|
| DUSt3R-224 | 0.029 | 0.054 | 2.296 | 0.74 | 38.1G |
| [[Spann3R]] | 0.034 | 0.069 | 4.785 | 27.4 | 5.0G |
| MUSt3R-224 | 0.028 | 0.062 | 3.256 | 40.4 | 4.1G |
| **MUSt3R-512** | 0.026 | **0.048** | 3.261 | 12.1 | 8.1G |

**表格说明**：MUSt3R 在重建精度上追平或超过 DUSt3R，同时速度快 40–86 倍、显存仅 4–8G（DUSt3R 需 38.1G）。

### Table 7: 多视图位姿回归（CO3Dv2 / RealEstate10K，10 帧）

| Method | Co3D mAA(30) | RE10K mAA(30) | FPS |
|---|---|---|---|
| DUSt3R-512-GA | 76.7 | 67.7 | 0.1 |
| [[Spann3R]] | 70.4 | 60.8 | 7.4 |
| MUSt3R-224 | 80.7 | 74.7 | 11.7 |
| **MUSt3R-512** | **84.1** | **75.1** | 4.1 |
| MUSt3R-512 (Pro) | 78.3 | 65.5 | 32.9 |

**表格说明**：MUSt3R-512 全面 SOTA，比需全局对齐的 DUSt3R-512-GA 快约 40 倍（4.1 vs 0.1 FPS）；(Pro) 用 [[Procrustes Analysis|Procrustes]] 换 PnP，速度进一步到 32.9 FPS。

### Table 9: DUSt3R 简化消融（成对设置，Regression Loss↓）

| 配置 | Regression Loss |
|---|---|
| Baseline | 0.193 |
| No RoPE in CA | 0.191 |
| Symmetric(none) | 0.560（崩溃） |
| Symmetric(embed) | 0.192 |

**表格说明**：cross-attention 去 [[RoPE]] 无损；对称解码器无可学习嵌入会崩溃（0.560），加嵌入 $B$ 恢复到基线水平。

### Table 10: 记忆管理消融（记忆上限 $n$、每次更新图像数 $s$）

| Method | $n$ | $s$ | 7S Acc | FPS | Mem |
|---|---|---|---|---|---|
| MUSt3R-224 | 10 | 1 | 0.037 | 86.31 | 2.5G |
| MUSt3R-224 | 20 | 1 | 0.030 | 59.51 | 2.9G |
| MUSt3R-224 | all | 1 | 0.028 | 40.41 | 4.1G |
| MUSt3R-512 | all | 1 | 0.026 | 12.10 | 8.1G |

**表格说明**：$s=1$（逐帧）质量最好；$n=10$ 时 224 版可达 86.31 FPS / 2.5G 显存；模型虽只在 ≤10 视图训练，却可利用多至 50 视图记忆。

---

## 实验

### 数据集与设置

| 项目 | 设置 |
|------|------|
| 初始化 | [[CroCo v2]]，解码器深度 $L=12$ |
| 分辨率 | 先 224 预训练，再 512 微调 |
| 成对预训练 | 14 个数据集（Habitat、ARKitScenes、BlendedMVS、MegaDepth、ScanNet++、CO3D-v2、TartanAir 等）|
| 多视图训练 | 12 个数据集，每场景 $N=10$ 张图像 |
| 其他 | 编码器冻结；xformers attention；token dropout 0.05/0.15；log-space 损失 |
| 变体 | MUSt3R-C（严格因果/在线）、MUSt3R（含 rendering）、(Pro)（Procrustes 位姿）|

### 关键结果

- **无标定 VO**：平均 ATE 5.5cm，超越所有稠密无约束方法。
- **位姿回归**：MUSt3R-512 全面 SOTA，比 DUSt3R-512-GA 快约 40 倍。
- **重建速度/显存**：224 版 40–86 FPS、2.5–4.1G，远优于 DUSt3R-224 的 0.74 FPS / 38.1G。

### 消融结论

1. cross-attention 可移除 [[RoPE]] 无损。
2. 对称解码器**必须**加可学习嵌入 $B$，否则崩溃。
3. 3D feedback 从 $L-1$ 层注入显著优于从最终层注入。
4. log-space 损失在 >10 视图时更稳。
5. 顺序（$s=1$）预测优于 n-by-n 批量。

---

## 批判性思考

### 优点

1. **一次前向输出多视图统一坐标系 pointmap**：从根本上消除 $O(N^2)$ 成对推理 + 全局对齐。
2. **对称共享权重**：参数减半，且能在度量空间直接预测。
3. **记忆机制可扩展**：类 KV-cache，显存低（数千图像可控），离线在线统一。
4. **无标定 VO SOTA**：在稠密无约束方法中大幅领先，甚至超过部分传统 SLAM。

### 局限

1. **参考帧依赖**：所有 pointmap 以第 1 帧为全局坐标系，视图**漂移离第 1 帧太远**时性能下降。
2. **训练视图数受限**：仅在 ≤10 视图训练，长序列外推能力靠记忆机制间接实现。
3. **仍无显式回环/BA**：在线模式长轨迹仍可能累积误差（虽比 Spann3R 好很多）。

### 潜在改进方向

1. 动态更换/多参考帧坐标系，缓解远离首帧的退化。
2. 更长序列训练或引入显式回环约束。
3. 与动态场景处理结合（本文以静态为主）。

---

## 关联笔记

### 基于

- [[DUSt3R]]：MUSt3R 是其从 two-view 到 multi-view 的对称扩展。
- [[CroCo v2]]：模型初始化来源。
- [[Point Map]]：核心输出表示。

### 对比

- [[Spann3R]]：同为在线增量式，MUSt3R 在位姿/重建/VO 上大幅超越。
- [[MASt3R]]：同谱系；本文 log 损失在多视图下比 MASt3R 默认损失更稳。
- [[Global Optimization]]：DUSt3R 需要的全局对齐，被本文的多视图记忆消除。

### 方法相关

- [[Memory Mechanism]]：类 KV-cache 的多层记忆。
- [[Procrustes Analysis]]：由局部/全局 pointmap 恢复位姿。
- [[Cross-Attention]] / [[RoPE]]：解码器机制（RoPE 在 CA 中被移除）。
- [[Visual Odometry]]：主要评测任务之一。

### 谱系相关

- [[CUT3R]]、[[TTT3R]]：DUSt3R 谱系的持久状态 / test-time training 在线重建工作。

---

## 速查卡片

> [!summary] MUSt3R
> - **核心**：DUSt3R two-view → 对称 multi-view + 多层记忆（类 KV-cache）
> - **架构**：共享权重对称 Siamese 解码器 + 可学习参考嵌入 B + Global 3D Feedback
> - **位姿**：局部 $X_{i,i}$ + 全局 $X_{i,1}$ 双 pointmap，Procrustes 恢复位姿
> - **关键结论**：无标定 VO SOTA（ATE 5.5cm），比 DUSt3R-GA 快 ~40×
> - **结果**：Co3D mAA(30) 84.1；重建 40–86 FPS、显存 4–8G
> - **局限**：参考帧依赖，远离首帧退化
> - **代码**：github.com/naver/must3r

---

*笔记创建时间: 2026-07-02*
