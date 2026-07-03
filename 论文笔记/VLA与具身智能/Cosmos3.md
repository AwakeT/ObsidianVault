---
title: "Cosmos 3: Omnimodal World Models for Physical AI"
method_name: "Cosmos3"
authors: [NVIDIA]
year: 2026
venue: arXiv
tags: [world-model, omnimodal, mixture-of-transformers, vision-language-action, diffusion-model, embodied-ai]
zotero_collection: 0-待整理
image_source: local
arxiv_html: https://arxiv.org/abs/2606.02800
created: 2026-06-17
---

# 论文笔记：Cosmos 3: Omnimodal World Models for Physical AI（架构部分）

> 本笔记只覆盖论文第 2 章 **Model Architecture**（数据 / 训练 / 实验等其余章节略）。

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA |
| 日期 | June 2026 |
| 项目主页 | research.nvidia.com/labs/cosmos-lab/cosmos3 |
| 链接 | [arXiv](https://arxiv.org/abs/2606.02800) / [Code](https://github.com/nvidia/cosmos) / [HF](https://huggingface.co/collections/nvidia/cosmos3) |

---

## 一句话总结

> Cosmos 3 用单一 [[Mixture-of-Transformers]] 架构统一处理与生成语言/图像/视频/音频/动作五种模态，把 VLM、视频生成器、世界模拟器和世界-动作模型纳入同一框架。

---

## 架构总览

Cosmos 3 同时支持多模态输入的**理解**与多模态输出的**生成**。除语言、视觉（图像+视频）、音频外，它把 **动作 (action)** 作为核心模态，引入专用的动作 token，将物理世界与语言推理、视频世界建模直接打通。

整体流程：

- **统一表征**: 各模态先经各自的 [[Modality-Specific Encoder|模态专用编码器]] 投影到统一表征空间。
- **统一骨干**: 由 [[Mixture-of-Transformers]] (MoT) 处理整条混合 token 序列。
- **双子序列**: 序列分为**自回归 (AR) 子序列**（推理/理解）与**扩散 (DM) 子序列**（生成）。
- **推理方式**: 语言 token 用 [[Next-Token Prediction|下一 token 预测]] 生成；其它模态通过 [[Diffusion Model|迭代去噪]]（实际用 [[Flow Matching|流匹配]] 预测速度）生成。

### Figure 1: 通用骨干

![[Cosmos3_fig1_overview.png]]

**说明**: 通过对语言、图像、视频、音频、动作五模态联合建模理解与生成，Cosmos 3 在单一网络中统一了多类模型：[[Vision-Language Model|视觉-语言模型]]、图像生成模型、音视频生成模型、策略/世界-动作模型、[[Forward Dynamics Model|前向动力学模型]]、[[Inverse Dynamics Model|逆动力学模型]]。输入-输出配置不同即切换不同工作模式。

### Figure 2: 作为 Physical AI 的起点

![[Cosmos3_fig2_starting_point.png]]

**说明**: 经过 Pre-Training + Mid-Training 的 Cosmos 3 可以在不改架构的前提下，针对 Text-to-Image、Image-to-Video（合成数据生成）、Policy（任务特化）、Interactive Environment（训练环境）等下游任务后训练。

---

## 2.1 模态编码器 (Encoders)

输入语言、视觉、音频、动作后，先用各模态专用编码器嵌入到统一表征空间。为了区分模态，对每个**非语言**模态额外加一个**可学习的、模态特定的 embedding 向量**，再送入 MoT 骨干。

#### 图像与视频 (Image and Video)

视觉输入用**两套独立编码器**：

- **理解用 [[Vision Transformer|ViT]] 编码器**: 经视觉-语言对齐预训练。patch size $16\times16$，后接两层 MLP 把 $2\times2$ token 合并并投影到 transformer latent。沿用 Qwen3-VL 做法，用 [[DeepStack]] 聚合 ViT 多层特征，并把基于文本的视频时间戳与视频帧交错插入。**该 ViT 与骨干联合训练**。
- **生成用 [[Video VAE|视频 VAE]] 编码器**: 取自 Wan2.2-TI2V-5B。时间上压缩 $4\times$、空间上压缩 $32\times32$（实现为 $16\times16$ 空间压缩 + $2\times2$ patch merge）。每个 VAE token 经一个线性层投影到 transformer hidden dim。**该 VAE 编码器在训练中冻结**。

#### 音频 (Audio)

采用 Lee et al. (2025b) 的 [[Audio VAE]]。48 kHz 立体声、hop size 1920，即每秒 25 个 token。音频 VAE 训练中冻结，token 经线性层投影到 hidden dim。

#### 动作 (Action)

支持自动驾驶、相机运动、机器人、第一人称人体（头与手）等多种 embodiment。各域有各自原生控制空间（关节轨迹、转向、身体姿态、相机变换），统一映射到一个共享动作接口，以支持跨域的推理、生成和策略学习。

### Figure 3: 统一动作表征

![[Cosmos3_fig3_action_repr.png]]

**说明**: 把异构 embodiment 控制映射到由共享几何分量构成的紧凑动作向量。Ego/effector 运动编码为**相对位姿伪动作**（3D 平移 + [[6D Rotation|6D 旋转]]）；grasp 状态直接编码当前操作状态（手指位置或夹爪开合）。不同 embodiment 动作向量长度不同（AV 9D、相机 9D、第一人称 57D、单臂 10D、双臂 20D、人形 29D），由**域感知输入/输出投影**处理。

**动作表征 (Action representations)**: 动作是诱发世界状态变化的因果变量。给定连续视频 token，动作 token $a_t$ 表示从上一状态 $v_{t-1}$ 到当前 $v_t$ 的转移。动作含至多三个分量：

- **ego pose**: agent 主观测帧的位姿
- **effector pose**: agent 末端执行器位姿
- **grasp state**: 操作状态

为避开 embodiment 特定的控制细节（如 PID 参数），ego/effector pose 表示为从状态差导出的**伪动作**。对连续的 [[SE(3)]] 位姿 $\mathbf{T}_{t-1}, \mathbf{T}_t$，运动表示为相对变换 $\Delta\mathbf{T}_t = \mathbf{T}_{t-1}^{-1}\mathbf{T}_t$；旋转用 [[6D Rotation|6D 表示]]（OpenCV 约定：z 轴沿手指/夹爪，x 轴向右）。grasp state 不取时间差，而直接编码 $t$ 时刻的操作状态。

**动作 token 化 (Action tokenization)**: 用**域感知的输入/输出投影层**，每个 embodiment 域有独立权重矩阵，但共享 MoT 骨干。

### 公式1: [[Action Tokenization|动作输入投影]]

$$
\mathbf{z} = \mathbf{W}_{\text{in}}^{(k)}\mathbf{x} + \mathbf{b}_{\text{in}}^{(k)}
$$

**含义**: 把归一化动作向量投影为 latent 动作 token。

**符号说明**:
- $\mathbf{z} \in \mathbb{R}^{d_{\text{model}}}$: latent 动作 token
- $\mathbf{x}$: 归一化后的动作向量
- $k \in \{1,\dots,K\}$: 域标识符
- $\mathbf{W}_{\text{in}}^{(k)} \in \mathbb{R}^{d_{\text{model}}\times d_{\text{in}}^{(k)}}$, $\mathbf{b}_{\text{in}}^{(k)} \in \mathbb{R}^{d_{\text{model}}}$: 域特定输入投影矩阵与偏置

### 公式2: [[Action Tokenization|动作输出投影]]

$$
\mathbf{x} = \mathbf{W}_{\text{out}}^{(k)}\mathbf{z} + \mathbf{b}_{\text{out}}^{(k)}
$$

**含义**: 把 token 解码回原始动作空间。

**符号说明**:
- $\mathbf{W}_{\text{out}}^{(k)} \in \mathbb{R}^{d_{\text{in}}^{(k)}\times d_{\text{model}}}$, $\mathbf{b}_{\text{out}}^{(k)} \in \mathbb{R}^{d_{\text{in}}^{(k)}}$: 域特定输出投影矩阵与偏置

所有投影参数从零初始化，与 MoT 骨干联合优化。预测的 6D 旋转用 [[Singular Value Decomposition|SVD]] 转回 $3\times3$ [[SO(3)]] 旋转矩阵。

---

## 2.2 Token 排布与生成模式

不同任务统一表示为**交错的多模态序列**，由不同模态片段组成，先经各编码器嵌入，再按统一格式打包。

#### 2.2.1 Token 排布 (Token Arrangement)

输入序列由两个子序列组成：**自回归 (AR) 子序列** + **扩散 (DM) 子序列**。

- **AR 子序列**: 负责推理与理解。含语言 token 以及 ViT 编码的视频/图像 token。所有 AR token 路由到 transformer decoder 层中一组**专用参数**（reasoner）。
- **扩散 (DM) 子序列**: 跟在 AR 子序列之后。含 VAE 编码的视频/图像 token、音频 token、动作 token。生成时模型对带噪 DM token 迭代去噪。DM token 路由到与 AR **不同的另一组参数**（generator），但通过每层的联合注意力与 AR token 交互。

任意任务都按同一格式排布：(1) AR token 在扩散 token 之前；(2) 扩散子序列内、每个模态的 clean 条件 token 在 noisy 扩散 token 之前；(3) 条件子序列与扩散子序列内部，均按 视觉 → 音频 → 动作 排序。

#### 2.2.2 生成模式 (Generation Mode)

记 clean 视觉/音频/动作 token 为 $v, s, a$，带噪版本为 $\tilde{v}, \tilde{s}, \tilde{a}$。共享 AR 前缀定义为 $\mathbf{S}_{\text{AR}} \triangleq [l_1,\dots,l_n,\langle\text{EOS}\rangle,\langle\text{BOG}\rangle]$（$l$ 为语言 token，$\langle\text{EOS}\rangle$/$\langle\text{BOG}\rangle$ 为句末/生成起始特殊 token）。

- **Language（语言）**: 仅含 AR 子序列，不激活扩散参数；图像/视频若存在则由 ViT 编码并放入 AR 子序列。此时 Cosmos 3 等同标准 [[Vision-Language Model|VLM]]。

### 公式3: Text-to-Image 序列

$$
\mathbf{S}_{\text{T2I}} = [\mathbf{S}_{\text{AR}},\ \tilde{v}_1]
$$

**含义**: AR 子序列为语言 token，扩散子序列为 VAE 编码的带噪目标图像 token $\tilde{v}_1$。

### 公式4: Text-to-Video (+Audio) 序列

$$
\mathbf{S}_{\text{T2V+Audio}} = [\mathbf{S}_{\text{AR}},\ \tilde{v}_{1:N},\ \tilde{s}]
$$

**含义**: 扩散子序列为带噪视频 token；可选联合生成音频时在视觉 token 后追加带噪音频 token。$N$ 为 latent 视频帧数。

### 公式5: Image-to-Video / Video-to-Video (+Audio) 序列

$$
\mathbf{S}_{\text{V2V}} = [\mathbf{S}_{\text{AR}},\ v_{1:P},\ \tilde{v}_{P+1:N}]
$$

**含义**: clean 条件帧 token 后接带噪目标视频 token。$P$ 为条件 latent 帧数：$P=1$ 即 Image-to-Video，$P>1$ 即 Video-to-Video。联合生成音频时同样追加音频 token。

### 公式6: Video Transfer 序列

$$
\mathbf{S}_{\text{Transfer}} = [\mathbf{S}_{\text{AR}},\ v^{\text{ctrl}}_{1:N},\ \tilde{v}_{1:N}]
$$

**含义**: 输入控制视频（如 edge/depth）+ 文本，生成对应 RGB 视频。$v^{\text{ctrl}}_{1:N}$ 为控制视频的 clean VAE token，作条件；RGB 视频 token 作带噪目标。

- **Action（动作）**: 支持三种生成模式——[[Forward Dynamics Model|前向动力学]]、[[Inverse Dynamics Model|逆动力学]]、联合视频-动作预测（policy）。对连续视频 token，$a_t$ 表示 $v_{t-1}\to v_t$ 的转移。前向动力学在已观测上下文与 clean 动作 token 条件下预测未来视觉状态；逆动力学从观测的视觉转移推断动作 token；policy 模式联合预测动作与视频 token（既生成干预又生成其视觉后果）。条件方向见 Figure 4。

### Figure 4: 动作序列配置

![[Cosmos3_fig4_action_seq.png]]

**说明**: 对一个视频-动作样本，通过改变哪些 token 是 clean、哪些是 noisy 来构造不同训练模式。动作 token 位于相邻视频 token 之间：$a_t$ 连接 $v_{t-1}\to v_t$，$a_{t+1}$ 连接 $v_t\to v_{t+1}$。前向动力学在 clean 动作条件下去噪视觉 token；逆动力学在 clean 视觉条件下去噪动作 token；video-action (policy) 同时去噪视觉与动作 token。

---

## 2.3 Mixture-of-Transformers (MoT) 架构

Cosmos 3 采用 [[Mixture-of-Transformers]] 处理来自不同模态的统一 token 序列。在**层级**上，每个 transformer decoder 层含**两组参数**：一组处理 AR 子序列的推理任务（**reasoner**），一组处理扩散子序列的生成任务（**generator**）。其 decoder 层结构与 Deng et al. (2025) 等统一生成模型相似，但在训练策略、位置嵌入、整体能力上不同。

### Figure 5: Cosmos 3 的 MoT 架构

![[Cosmos3_fig5_mot.png]]

**说明**: 左：单个 transformer 作用于由 AR + DM 两个子序列构成的同一 token 序列。AR 携带离散文本 token 与（可选）ViT 编码视觉 token，以 `<EOS>` 结束并接 `<BOG>`；DM 携带各编码器产生的连续 token（训练时加噪）。每个 transformer block 内，AR 与 DM token 经**独立的 LayerNorm 与 MLP**（均由预训练 VLM 共同初始化），在**共享自注意力算子**处汇合。$\mathbf{Q}_{\text{AR}}$ 只对 $\mathbf{K}_{\text{AR}}, \mathbf{V}_{\text{AR}}$ 做因果注意力；$\mathbf{Q}_{\text{DM}}$ 对拼接的 $[\mathbf{K}_{\text{AR}};\mathbf{K}_{\text{DM}}]$、$[\mathbf{V}_{\text{AR}};\mathbf{V}_{\text{DM}}]$ 做双向注意力。右：注意力 mask，AR 因果、DM 全连接。

#### 2.3.1 双塔层结构 (Dual-Tower Layer Structure)

标准 transformer decoder 层含自注意力、前馈网络和归一化层。MoT 不用同一套参数处理所有 token，而是用**两条 pathway（双塔）**。每条 pathway 是带自己参数的标准 transformer 层（含 LayerNorm、注意力投影矩阵、FFN）。两塔均由预训练 [[Vision-Language Model|VLM]] 权重初始化，从而继承强语言与视觉推理能力，同时学习生成高保真视频。训练与推理时，前部 AR 子序列路由到 **reasoner 塔**，后部扩散子序列路由到 **generator 塔**。

#### 2.3.2 双流联合注意力 (Dual-Stream Joint Attention)

两塔参数独立，但 DM 子序列 token 通过**双流联合注意力**与 AR 子序列交互。记 AR/DM 子序列的 Q/K/V 为 $\mathbf{Q}_{\text{AR}}, \mathbf{K}_{\text{AR}}, \mathbf{V}_{\text{AR}}$ 与 $\mathbf{Q}_{\text{DM}}, \mathbf{K}_{\text{DM}}, \mathbf{V}_{\text{DM}}$。

**自回归子序列注意力**: AR token 只在 AR 内用 [[Causal Attention|因果自注意力]]，每个 token 只能注意到同序列中之前的 token，保持继承自 VLM 的文本生成能力。

### 公式7: [[Causal Attention|AR 因果注意力]]

$$
\mathbf{O}_{\text{AR}} = \text{Attn}_{\text{causal}}(\mathbf{Q}_{\text{AR}},\ \mathbf{K}_{\text{AR}},\ \mathbf{V}_{\text{AR}})
$$

**含义**: AR token 仅对自身子序列做因果注意力。

**扩散子序列注意力**: DM token 用**全双向注意力**，键值为 AR 与 DM token 的并集。这样每个扩散 token 可自由注意到 AR 子序列的文本提示以及序列中所有其它条件/扩散 token，维持时空一致性。

### 公式8: [[Self-Attention|DM 双向联合注意力]]

$$
\mathbf{O}_{\text{DM}} = \text{Attn}_{\text{full}}\big(\mathbf{Q}_{\text{DM}},\ [\mathbf{K}_{\text{AR}};\mathbf{K}_{\text{DM}}],\ [\mathbf{V}_{\text{AR}};\mathbf{V}_{\text{DM}}]\big)
$$

**含义**: DM token 对 AR+DM 的键值做全注意力。$[\cdot;\cdot]$ 为沿序列维拼接。**AR token 永不被 DM token 更新**，保持条件 pathway 的因果完整性。

---

## 2.4 多模态位置嵌入 (Multimodal Position Embedding)

位置嵌入把时间与空间结构注入注意力。Cosmos 3 受 3D Multimodal RoPE ([[MRoPE]], Bai et al. 2025a) 启发，设计了带**绝对时间索引**的 3D MRoPE，把视频、音频、动作 token 对齐到同一物理时间轴。

原始 3D MRoPE 把每个注意力头的隐藏维度划分为 temporal/height/width 三部分，temporal 只记录离散 token 索引。这对图像/视频理解足够，但 Cosmos 3 中视频/音频/动作可能以**不同帧率/采样率同时生成**，必须对齐到绝对物理时间轴。论文先给出沿用原始设计的基础形式，再描述其扩展，尤其是**绝对时间调制**。

#### 2.4.1 位置索引分配 (Position Index Allocation)

**自回归 token**: 为兼容语言生成与图像/视频理解，AR 子序列中所有语言 token 与 ViT 编码媒体 token 的位置索引沿用原始 3D MRoPE。语言 token 设 $t=h=w$ 为同一单调递增值，使 3D MRoPE 退化为标准 1D RoPE；ViT token 的 $t$ 由同一帧所有 token 共享，$h, w$ 按空间位置独立变化。该分配与 Qwen3-VL 的 3D MRoPE 一致。

### Figure 6: 3D MRoPE 下的坐标分配示意

![[Cosmos3_fig6_coord.png]]

**说明**: 左：含语言、视频（两帧、各 $2\times2$ 空间网格）、音频、动作的打包 token 序列的 $(t,h,w)$ 三元组。语言 token $t=h=w$；视频 token 三轴都变；动作与音频 token 只用时间坐标（$h=w=0$）。模态偏移 $k$ 分隔文本与视觉时间范围。右：FPS 调制把帧索引映射到缩放后的时间位置，使 16/24/30 FPS 下相同真实时长占用相同位置范围（24 FPS 为基准帧率）。

**扩散 token**: 视频 token 三轴都变：$t$ 随时间 latent 帧索引前进，$h, w$ 每帧独立在空间网格 $(0\dots H-1, 0\dots W-1)$ 上平铺。图像视为单帧视频，只在 $\langle h,w\rangle$ 变化。每个视觉段开始时空间与时间索引都重置为零，即模型把 $t, h, w$ 当作**段内绝对坐标**而非全局序列位置。所有**音频与动作 token 只带时间坐标**（$h=w=0$）：音频 token 的时间索引随每个音频 hop 前进，动作 token 随每个采样步前进。

**AR 与扩散 token 间隔**: 直接让扩散 token 从最后一个 AR token 的时间偏移开始，会导致过饱和与棋盘伪影（在 Super 等更大变体上尤甚）。原因推测是最后的语言 token 与首帧视觉 token 占据相邻时间位置、时间嵌入近乎相同。受 Cao et al. (2025) 启发，论文在 AR 与扩散子序列间**插入固定时间间隔**，统一平移其后所有视觉/音频/动作 token 的时间索引，在位置空间制造缓冲，无需改架构或加可学习嵌入。所有模型中该 **gap 设为 15000**。

#### 2.4.2 绝对时间调制 (Absolute Temporal Modulation)

时间维上单位步在不同模态/数据源可能对应不同物理时间间隔（如 60 FPS 与 24 FPS 视频，24 FPS 的单位时间索引增量对应的物理时长是 60 FPS 的 2.5 倍）。**FPS 调制**通过调整每个时间增量的有效大小，把不同时间分辨率的 token 对齐到共享物理时间轴。

定义每秒时间步数 TPS 表征物理时间分辨率：视频 token 的 TPS = 帧率 / 时间压缩因子（本文 VAE 压缩因子 4）；音频 token $\text{TPS}_{\text{audio}} = \frac{48000}{1920} \approx 25$（48 kHz、hop 1920）；动作 token 的 TPS 即动作数据采样频率。

### 公式9: [[MRoPE|FPS 调制时间增量]]

$$
\delta t = \frac{\text{TPS}_{\text{base}}}{\text{TPS}}
$$

**含义**: 把时间维单位长度关联到基准 TPS。某扩散子序列内 token 时间索引需增一个单位步时，实际增量按此调制，使相同真实时长占用相同位置范围。

**符号说明**:
- $\text{TPS}_{\text{base}}$: 基准每秒时间步数
- $\text{TPS}$: 当前模态/数据源的每秒时间步数

由于视频占训练数据多数、24 FPS 最常见，设 $\text{TPS}_{\text{base}} = \frac{24}{4} = 6$（4 为视频 tokenizer 时间压缩比）。

---

## 2.5 模型变体 (Model Variants)

Cosmos 3 训练三种规模：**Edge / Nano / Super**，覆盖端侧部署到数据中心推理。所有变体均由预训练 VLM 初始化并采用上述 MoT 架构。Cosmos3-Nano 与 Cosmos3-Super 随本文发布，Cosmos3-Edge 后续发布。

- **Cosmos3-Edge**: 4B 参数（基于 2B dense transformer）。28 层、hidden 2048、16 注意力头、8 KV 头、head dim 128、FFN 9216。从零用 Megatron 训练，设计大体沿用 Qwen3-1.7B，但**去掉 QK normalization**、FFN 激活用 **ReLU-squared**。
- **Cosmos3-Nano**: 16B 参数（基于 8B dense transformer），采用 Qwen3-VL 8B 架构。LLM 36 层、hidden 4096、32 注意力头、8 KV 头、head dim 128、FFN 12288。
- **Cosmos3-Super**: 64B 参数（基于 32B dense transformer），采用 Qwen3-VL 32B 架构。LLM 64 层、hidden 5120、64 注意力头、8 KV 头、head dim 128、FFN 25600。

### Table 2: Cosmos 3 MoT 模型变体

| Variant | LLM Layers | Hidden Dim | Attn Heads | KV Heads | Head Dim | FFN Dim |
|---------|-----------|-----------|-----------|----------|----------|---------|
| Cosmos3-Edge | 28 | 2,048 | 16 | 8 | 128 | 9,216 |
| Cosmos3-Nano | 36 | 4,096 | 32 | 8 | 128 | 12,288 |
| Cosmos3-Super | 64 | 5,120 | 64 | 8 | 128 | 25,600 |

**说明**: 所有模型共享双塔 MoT 架构；"LLM Layers" 指 transformer decoder 层数，每层为 reasoner 与 generator 两塔各带一套独立参数。Edge 为从零训练的 2B dense transformer；Nano 与 Super 由预训练 Qwen3-VL 权重初始化。

---

## 关联笔记

### 方法相关
- [[Mixture-of-Transformers]]: 核心双塔骨干
- [[MRoPE]]: 多模态位置嵌入基础
- [[Flow Matching]]: 生成器训练目标
- [[6D Rotation]]: 动作旋转表示

### 编码器相关
- [[Vision Transformer]]: 理解用视觉编码器
- [[Video VAE]]: 生成用视觉编码器
- [[Audio VAE]]: 音频编码器

### 任务相关
- [[Forward Dynamics Model]] / [[Inverse Dynamics Model]]: 动作生成模式
- [[Vision-Language Model]]: 初始化来源与退化模式

---

## 速查卡片

> [!summary] Cosmos 3 架构
> - **核心**: 单一 MoT 双塔（reasoner + generator）统一五模态理解与生成
> - **序列**: AR 子序列（因果，语言+ViT）+ DM 子序列（双向，VAE/音频/动作，扩散去噪）
> - **位置**: 带绝对时间调制（FPS 调制）的 3D MRoPE，AR/DM 间插 15000 时间 gap
> - **动作**: 6D 旋转伪动作 + 域感知输入/输出投影，共享骨干
> - **规模**: Edge 4B / Nano 16B / Super 64B，由 Qwen3-VL 初始化
> - **代码**: github.com/nvidia/cosmos

---

*笔记创建时间: 2026-06-17（仅含 Model Architecture 章节）*
