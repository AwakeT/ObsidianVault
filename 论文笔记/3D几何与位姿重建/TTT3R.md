---
title: "TTT3R: 3D Reconstruction as Test-Time Training"
method_name: "TTT3R"
authors: [Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, Anpei Chen]
year: 2026
venue: ICLR 2026
tags: [3d-reconstruction, test-time-training, recurrent-model, length-generalization, online-reconstruction, associative-memory]
zotero_collection: 6-3D视觉
image_source: online
arxiv: https://arxiv.org/abs/2509.26645
arxiv_html: https://arxiv.org/html/2509.26645v4
created: 2026-07-02
---

# 论文笔记：TTT3R: 3D Reconstruction as Test-Time Training

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[TTT3R]] |
| 机构 | 浙江大学、西湖大学、图宾根大学（Tübingen AI Center）|
| 日期 | 2025-09-30 (arXiv v1)；v4 (2026-03-03) |
| 会议/来源 | ICLR 2026 |
| 项目主页 | [rover-xingyu.github.io/TTT3R](https://rover-xingyu.github.io/TTT3R/) |
| 代码 | [github.com/Inception3D/TTT3R](https://github.com/Inception3D/TTT3R) |
| 链接 | [arXiv](https://arxiv.org/abs/2509.26645) / [HTML](https://arxiv.org/html/2509.26645v4) |
| 相关工作 | [[CUT3R]]、[[Spann3R]]、[[Test-Time Training]]、[[DeltaNet]]、[[Linear Attention]]、[[VGGT]] |

---

## 一句话总结

> TTT3R 把 [[CUT3R]] 的循环状态更新重新解释为一个 **Test-Time Training（测试时训练）** 的在线学习问题，用记忆状态与新观测之间的**对齐置信度**推导出一个**闭式学习率**来门控状态更新，无需训练即可 plug-in 到 CUT3R，使长序列全局位姿估计取得 **2× 提升**，并以 20 FPS、仅 6GB 显存处理数千张图像。

---

## 核心贡献

1. **TTT 分析框架**：从 [[Test-Time Training]] 视角分析有状态 3D 重建模型（尤其 [[CUT3R]]），把 recurrent 3D 重建的 state update 重新解释为测试时的在线学习 / 联想记忆更新。
2. **TTT3R 更新规则**：提出一个简单有效的**推理时 state update rule**，作为 CUT3R 在线联想召回（online associative recall）的闭式状态转移。
3. **置信度引导的学习率**：用 **state query 与 observation key 之间的 cross-attention 对齐置信度**作为**每-token 自适应学习率 $\beta_t$**，无额外参数、无需训练、无额外计算开销。
4. **Training-free、plug-and-play**：直接嵌入 CUT3R 即可用于下游任务，保持其推理速度（20 FPS）与显存（6GB）。
5. **长度泛化提升**：短序列与 SOTA 在线方法持平，长序列显著领先（位姿 2×）；提供可选 **TTT3R + State Reset** 处理超 1000 帧序列。

---

## 问题背景

### 要解决的问题

3D 重建基础模型在**超出训练上下文长度**时性能显著下降（有限的 length generalization）。如何在**常数显存**下让在线 3D 重建稳健地泛化到数千帧长序列。

### 现有方法的局限

- **Transformer + full attention**（[[VGGT]]、Fast3R）：计算和显存随序列长度**二次增长**，KV-cache 压缩、flash attention 都无法根治长上下文瓶颈，长序列会 OOM。
- **[[CUT3R]]（RNN-based）**：唯一能做**常数显存**，但训练多为 64 帧序列，无法泛化到长序列，出现 state **forgetting（遗忘）**、drift（漂移）、严重退化。
- 现代 RNN（[[Linear Attention]]、Mamba、[[DeltaNet]]、Titans、GLA、RetNet）在语言任务上可比肩 Transformer，但超训练长度后性能退化，归因于 **state overfitting / forgetting / unexplored state distributions**。

### 本文动机

从 [[Test-Time Training|TTT]] 视角重审 recurrent 3D 重建的 state update：把 state 视为在测试时从 in-context tokens 学到的 **fast weight**，slow weights（模型参数）在推理时冻结、充当 meta-learner。这给出对 state overfitting 的原理性理解，并指出**基于梯度更新 + 自适应学习率平衡遗忘与学习**的联想召回能大幅提升长度泛化。

---

## 方法详解

### 序列建模统一框架（4 步）

处理在线图像流。对每帧 $\mathbf{I}_t \in \mathbb{R}^{W\times H\times 3}$ 实时估计相机位姿 $\mathbf{T}_t$、内参 $\mathbf{C}_t$、规范坐标点云 $\mathbf{P}_t$。通用流程：**Tokenize → Update → Read → De-tokenize**。

- 图像 tokenizer（DINO / [[CroCo]]）把 $\mathbf{I}_t$ patch 化为 token $\mathbf{X}_t$；用它更新前一状态 $\mathbf{S}_{t-1}$ 得 $\mathbf{S}_t$；从 $\mathbf{S}_t$ 读出 output token $\mathbf{Y}_t$；经 dense de-tokenizer（pixel shuffle + DPT head）得 pixel-aligned pointmap。

**update / read 操作是不同方法的核心区别**，分两类：

- **Full attention-based**（[[VGGT]]、Fast3R）：所有帧全局 all-to-all self-attention，等价于**状态拼接、state 随 $O(t)$ 增长**，计算 $O(t^2)$。StreamVGGT 用 causal attention 降到 $O(t)$ 计算，但 state 仍是随 $O(t)$ 增长的 KV 列表，显存持续增长。
- **RNN-based**（[[Spann3R]]、[[CUT3R]]）：每帧与定长 state 做 one-to-one [[Cross-Attention]]，计算与显存 $O(1)$，但会**遗忘**、随帧数退化。

### 从 TTT 视角重审 RNN 重建

TTT 把 state 当**定长 fast weight** $\mathbf{S}_{t-1}\in\mathbb{R}^{n\times c}$，训练与推理都用**梯度下降**更新，slow weights 推理时冻结。与 [[DeltaNet]]/线性 TTT 的联系：linear TTT 最小化重建误差 $\|\mathbf{S}_{t-1}\mathbf{K}_{\mathbf{X}_t}-\mathbf{V}_{\mathbf{X}_t}\|^2$，得到解析梯度形式，其中 key 指定"写到哪"、value 指定"写什么"、学习率作为门控控制 memory plasticity。

**关键洞察（CUT3R 的缺陷）**：把 CUT3R 的 [[Cross-Attention]] 更新写成 TTT 形式后可见，softmax 权重沿 observation-token 维**归一化求和为 1.0**，导致模型总是**优先新观测 $\mathbf{X}_t$ 而非历史 state $\mathbf{S}_{t-1}$**，造成**灾难性遗忘（catastrophic forgetting）**。相对标准 TTT，这等价于缺少灵活学习率（**等效于常数 $\beta_t=1.0$**）。

学习率 $\beta$ 的三类建模：**ScalarLR**（RetNet，可学习标量）、**ConditionLR**（DeltaNet/TTT/Mamba-2，输入相关标量）、**TokenLR**（GLA，输入相关且逐-token）。

### 置信度引导的状态更新规则（核心）

TTT3R 遵循 **TokenLR** 范式（$\beta_t\in\mathbb{R}^{n\times1}$），用 memory 与 observation 的对齐置信度作为每-token 自适应学习率：cross-attention $\mathbf{Q}_{\mathbf{S}_{t-1}}\mathbf{K}^{\top}_{\mathbf{X}_t}$ 沿图像空间维聚合到 $n$ 个 state token。

**直觉**：不对所有 state 均匀更新（均匀更新会因无纹理区域等低质量更新导致次优），而用 cross-attention 统计估计对齐置信度——**对齐置信度越高 → state 与观测匹配越强、不确定性越低 → 更新步长越大**；聚合 token 级统计抑制低质量更新。**training-free、plug-and-play**，无需 fine-tune。

---

## 关键公式

### 公式1：[[Test-Time Training|序列建模 Tokenize]]

$$
\mathbf{X}_t = \texttt{Tokenize}(\mathbf{I}_t)
$$

**含义**：图像 tokenizer 把第 $t$ 帧 patch 化为图像 token $\mathbf{X}_t\in\mathbb{R}^{(h\times w)\times c}$。

### 公式2：[[VGGT|Full attention 状态拼接]]

$$
\texttt{Update}(\mathbf{S}_{t-1},\mathbf{X}_t) = \mathbf{S}_{t-1}.\texttt{append}(\mathbf{K}_{\mathbf{X}_t}, \mathbf{V}_{\mathbf{X}_t})
$$

**含义**：state 是 KV 对列表 $\mathbf{S}_{t-1}=[(\mathbf{K}_{\mathbf{X}_1},\mathbf{V}_{\mathbf{X}_1}),\ldots]$，随帧数线性增长，复杂度 $O(t^2)$。

### 公式3：[[CUT3R|RNN 定长状态更新]]

$$
\texttt{Update}(\mathbf{S}_{t-1},\mathbf{X}_t) = \mathbf{S}_{t-1} + \texttt{softmax}(\mathbf{Q}_{\mathbf{S}_{t-1}}\mathbf{K}^{\top}_{\mathbf{X}_t})\mathbf{V}_{\mathbf{X}_t}
$$

**含义**：state $\mathbf{S}_{t-1}\in\mathbb{R}^{n\times c}$ 定长，每帧与之做 one-to-one cross-attention；计算与显存 $O(1)$，但会遗忘。这是 CUT3R 的原始更新。

### 公式4：[[Test-Time Training|TTT 梯度更新]]

$$
\texttt{Update}(\mathbf{S}_{t-1},\mathbf{X}_t) = \mathbf{S}_{t-1} - \beta_t\nabla(\mathbf{S}_{t-1},\mathbf{X}_t)
$$

**含义**：把 state 当 fast weight，用学习到的梯度函数 $\nabla$ 与学习率 $\beta_t$ 做在线梯度下降；直觉是把当前观测的 KV-cache 尽量精确地编码进定长 memory。

### 公式5：[[DeltaNet|DeltaNet 解析梯度]]

$$
\texttt{Update}(\mathbf{S}_{t-1},\mathbf{X}_t) = \mathbf{S}_{t-1} - \beta_t\nabla(\mathbf{S}_{t-1},\mathbf{X}_t)
$$

**含义**：linear TTT（DeltaNet）最小化 $\|\mathbf{S}_{t-1}\mathbf{K}_{\mathbf{X}_t}-\mathbf{V}_{\mathbf{X}_t}\|^2$ 得到与 Eq.4 同构的解析梯度：key 指定"写到哪"、value 指定"写什么"、$\beta_t$ 门控 memory plasticity。

### 公式6：CUT3R 更新的 TTT 改写

$$
\mathbf{S}_{t-1} + \texttt{softmax}(\mathbf{Q}_{\mathbf{S}_{t-1}}\mathbf{K}^{\top}_{\mathbf{X}_t})\mathbf{V}_{\mathbf{X}_t} = \mathbf{S}_{t-1} - \beta_t\nabla(\mathbf{S}_{t-1},\mathbf{X}_t)
$$

**含义**：梯度定义为观测 value 的线性组合，权重是 state query 与观测 key 的 softmax 对齐分数 $\texttt{softmax}(\mathbf{Q}_{\mathbf{S}_{t-1}}\mathbf{K}^{\top}_{\mathbf{X}_t})\in\mathbb{R}^{n\times(h\times w)}$。由此可见 CUT3R 隐含 $\beta_t=1.0$，softmax 归一化使其**总是偏向新观测 → 灾难性遗忘**。

### 公式7：[[TTT3R|置信度学习率]]

$$
\beta_t = \sigma\!\left(\sum_m \mathbf{Q}_{\mathbf{S}_{t-1}}\mathbf{K}^{\top}_{\mathbf{X}_t}\right)
$$

**含义**：TTT3R 的核心。$\sigma$ 为 sigmoid；$\sum_m \equiv \frac{1}{m}\sum_{i=1}^{m}$ 是沿图像空间维 $m=h\times w$ 的**归一化求和（均值）**；$\mathbf{Q}_{\mathbf{S}_{t-1}}\mathbf{K}^{\top}_{\mathbf{X}_t}\in\mathbb{R}^{n\times(h\times w)}$ 为图像注意力。得到每-token 自适应学习率 $\beta_t\in\mathbb{R}^{n\times1}$，作为 gated attention 的软门。

**符号说明**：

- $\mathbf{Q}_{\mathbf{S}_{t-1}}\in\mathbb{R}^{n\times c}$：state 的 query 投影。
- $\mathbf{K}_{\mathbf{X}_t}\in\mathbb{R}^{(h\times w)\times c}$：观测 key。
- $\beta_t\in\mathbb{R}^{n\times1}$：逐-state-token 的对齐置信度 / 学习率。

### 公式8：[[TTT3R|完整状态更新规则]]

$$
\texttt{Update}(\mathbf{S}_{t-1},\mathbf{X}_t) = \mathbf{S}_{t-1} - \beta_t\nabla(\mathbf{S}_{t-1},\mathbf{X}_t)
$$

**含义**：在 Eq.6 的梯度前乘以 Eq.7 的置信度门控 $\beta_t$，替代 CUT3R 隐含的 $\beta_t=1.0$，从而平衡历史保留与新观测吸收。

---

## 关键图表

> 图表来源：论文 arXiv HTML v4。共 18 张主 Figure、6 张 Table（含附录消融）。图片 URL 前缀 `https://arxiv.org/html/2509.26645v4/`。

### Figure 1: CUT3R vs TTT3R

![Figure 1](https://arxiv.org/html/2509.26645v4/x1.png)

**说明**：左：[[CUT3R]] 把观测编码进 state（memory）$\mathbf{S}_{t-1}$，随视图增多严重遗忘退化；右：TTT3R 把 state 当 fast weight 经梯度下降更新，学习率 $\beta_t$ 与梯度 $\nabla_t$ 由冻结的 slow weights（meta-learner）预测，平衡历史保留与新观测。

### Figure 2: GPU Memory Cost

![Figure 2](https://arxiv.org/html/2509.26645v4/x2.png)

**说明**：推理显存占比对比。RNN-based（含 TTT3R）常数显存，full-attention 随帧数增长直至 OOM。

### Figure 3: Sequence Modeling Layers

![Figure 3a](https://arxiv.org/html/2509.26645v4/x3.png)
![Figure 3c](https://arxiv.org/html/2509.26645v4/x5.png)

**说明**：(a) Full Attention 追加状态→二次开销；(b) Vanilla RNN 定长→线性但遗忘；(c) Test-Time Training 把 state 当测试时用自适应学习率梯度下降学到的 fast weights，提升长度泛化。

### Figure 4: TTT3R Illustration

![Figure 4a CUT3R](https://arxiv.org/html/2509.26645v4/x6.png)
![Figure 4b TTT3R](https://arxiv.org/html/2509.26645v4/x7.png)

**说明**：(a) 原始 CUT3R pipeline；(b) TTT 视角的置信度引导 state update，对齐置信度作为逐-token 学习率（Eq.8），training-free。

### Figure 5: Confidence-guided Update

![Figure 5](https://arxiv.org/html/2509.26645v4/x8.png)

**说明**：引入图像注意力 $\mathbf{Q}_{\mathbf{S}_{t-1}}\mathbf{K}^{\top}_{\mathbf{X}_t}$ 作为逐-token 学习率 $\beta_t$，缓解灾难性遗忘并促成在线 loop closure。

### Figure 6: Runtime on ScanNet

![Figure 6](https://arxiv.org/html/2509.26645v4/x9.png)

**说明**：ScanNet 上运行时对比；OOM 表示该方法超此点显存溢出。

### Figure 7: Camera Pose (long sequence)

![Figure 7](https://arxiv.org/html/2509.26645v4/x10.png)

**说明**：ScanNet/TUM-D 相机位姿。[[VGGT]] 为保留全历史的离线上界；full-attention 方法延迟高、显存耗尽；CUT3R 高效但长序列漂移；本文比 CUT3R 精度 2× 且保持实时。

### Figure 8: Video Depth

![Figure 8](https://arxiv.org/html/2509.26645v4/x12.png)

**说明**：Bonn scale-invariant 相对深度与 KITTI metric depth 对比；full-attention 方法 >150 帧 OOM。

### Figure 9: 7-scene Reconstruction

![Figure 9](https://arxiv.org/html/2509.26645v4/x14.png)

**说明**：7-scene 3D 重建（Chamfer↓、Normal Consistency↑ 随视图数变化）；CUT3R 严重退化，TTT3R 长序列稳健、Chamfer 低于 Point3R。

### Figure 10: Qualitative Reconstruction

![Figure 10](https://arxiv.org/html/2509.26645v4/x15.png)

**说明**：3D 重建定性；TTT3R 提升长度泛化、缓解遗忘、支持在线 loop closure。

### Figure 14–15: State Reset (>1000 frames)

![Figure 14](https://arxiv.org/html/2509.26645v4/x22.png)
![Figure 15](https://arxiv.org/html/2509.26645v4/x23.png)

**说明**：>1000 帧序列上，TTT3R 只缓解不完全解决遗忘；TTT3R + State Reset（每 100 帧重置并用全局 metric 位姿对齐 chunk）稳健，CUT3R+Reset 仍漂移。

### Table 1: 可学习门控机制对比（位姿 1000 帧 TUM-D；深度 500 帧 KITTI）

| Method | ATE↓ | RPE rot↓ | Abs Rel↓ | δ<1.25↑ |
|---|---|---|---|---|
| [[CUT3R]] | 0.173 | 0.494 | 0.152 | 80.2 |
| CUT3R + ScalarLR | 0.165 | 0.502 | 0.151 | 80.6 |
| CUT3R + ConditionLR | 0.166 | 0.509 | 0.149 | 81.0 |
| CUT3R + TokenLR | 0.154 | 0.497 | 0.148 | 81.5 |
| **TTT3R** | **0.106** | **0.431** | **0.131** | **86.9** |
| TTT3R + Finetune | 0.091 | 0.434 | 0.133 | 86.3 |

**表格说明**：可学习门控（含 TokenLR）受限于 64 帧训练序列，均显著弱于 training-free 的 TTT3R；TTT3R 直接从上下文推导学习率。

### Table 5: vs. non-TTT 基线（位姿 1000 帧 TUM-D；深度 500 帧 KITTI）

| Method | ATE↓ | RPE rot↓ | Abs Rel↓ | δ<1.25↑ |
|---|---|---|---|---|
| CUT3R | 0.173 | 0.494 | 0.152 | 80.2 |
| CUT3R + Reset | 0.126 | 0.403 | 0.128 | 84.9 |
| CUT3R + EMA | 0.164 | 0.525 | 0.150 | 80.7 |
| CUT3R + BurnIn | 0.144 | 3.093 | 0.151 | 80.2 |
| TTT3R | 0.106 | 0.431 | 0.131 | 86.9 |
| **TTT3R + Reset** | **0.093** | **0.375** | **0.115** | **88.5** |

**表格说明**：TTT3R 全面超过 Reset/EMA/BurnIn 等非 TTT 基线；整合 State Reset（n=100）达到最佳。

### Table 6: 短序列相机位姿（Sintel 50 / TUM-D 90 / ScanNet 90 帧），ATE↓

| Method | Online | Sintel | TUM-D | ScanNet |
|---|---|---|---|---|
| [[VGGT]] | ✗ | 0.172 | 0.012 | 0.035 |
| [[Spann3R]] | ✓ | 0.329 | 0.056 | 0.096 |
| [[CUT3R]] | ✓ | 0.213 | 0.046 | 0.099 |
| Point3R | ✓ | 0.351 | 0.075 | 0.106 |
| StreamVGGT | ✓ | 0.251 | 0.061 | 0.161 |
| STream3R | ✓ | 0.213 | 0.026 | 0.052 |
| **TTT3R** | ✓ | 0.201 | 0.028 | 0.064 |

**表格说明**：TTT3R 在**在线方法**中总体最好；短序列上尚未超过强离线方法 VGGT（离线上界）。

### Table 7: 短序列视频深度（Sintel 50 / Bonn 110 / KITTI 110 帧），Abs Rel↓ / δ<1.25↑

**Metric Scale（绝对尺度）**

| Method | Online | Sintel | Bonn | KITTI |
|---|---|---|---|---|
| [[MASt3R]] | ✗ | 1.022/14.3 | 0.272/70.6 | 0.467/15.2 |
| [[CUT3R]] | ✓ | 1.029/23.8 | 0.103/88.5 | 0.122/85.5 |
| Point3R | ✓ | 0.777/17.1 | 0.137/94.7 | 0.191/73.8 |
| STream3R | ✓ | 1.041/21.0 | 0.084/94.4 | 0.234/57.6 |
| **TTT3R** | ✓ | 0.977/24.5 | 0.090/94.2 | **0.110/89.1** |

**表格说明**：TTT3R 在 KITTI 的 metric 与 scale-invariant 均领先，Sintel/Bonn 第一或第二。

### Table 2–4: State Reset / EMA / BurnIn 超参消融

- **State Reset 周期**：最优 n=100（ATE 0.126）。
- **EMA shrinkage rate α**：最优 α=0.001（ATE 0.164）。
- **Burn-In keyframe interval**：最优 n=100（ATE 0.144）。

**表格说明**：Reset 对防 state overfitting 最有效，故整合为 TTT3R + Reset。

---

## 实验

### 数据集与设置

| 项目 | 设置 |
|------|------|
| 基座 | [[CUT3R]]（RNN-based，training-free 干预）|
| 主对比 | CUT3R、Point3R、StreamVGGT、[[VGGT]]（离线上界）等 |
| 位姿 | TUM-dynamics、ScanNet；ATE（Sim(3) 对齐）、RPE |
| 深度 | KITTI、Bonn；Abs Rel、δ<1.25（含无尺度对齐 metric）|
| 重建 | 7-scene；Chamfer Distance、Normal Consistency |
| 长序列 | 输入视图 50→1000，单卡 48GB |
| 效率 | 20 FPS、6GB 显存处理数千图像 |

### 关键结果

- **位姿**：TTT3R 长序列位姿 **2× 于 CUT3R**，速度/显存与 CUT3R 相同。
- **深度**：VGGT/StreamVGGT 约 150 帧 OOM；只有 CUT3R/Point3R/TTT3R 支持 metric，TTT3R 总体最佳且无需 fine-tune。
- **重建**：TTT3R 显著超 CUT3R/StreamVGGT，接近离线 VGGT，同时在线实时、仅 6GB 显存。

### 消融/分析（附录）

1. 可学习门控（ScalarLR/ConditionLR/TokenLR）受 64 帧训练长度限制，弱于 TTT3R。
2. Finetune TTT3R 对位姿有边际收益（ATE 0.091），但深度略降（4–64 帧微调偏向全局对齐）。
3. 相对 Reset/EMA/BurnIn，TTT3R 全面领先；Reset 防 state overfitting 最有效。

---

## 批判性思考

### 优点

1. **优雅的理论统一**：把 3D 重建的 recurrent state update 纳入 [[Test-Time Training]] / 联想记忆框架，与 [[DeltaNet]]、线性注意力打通。
2. **一针见血的诊断**：指出 CUT3R 隐含 $\beta_t=1.0$、softmax 归一化导致必然偏向新观测 → 遗忘。
3. **零成本干预**：training-free、plug-and-play、无额外参数/计算，长序列位姿直接 2×。
4. **常数显存 + 实时**：20 FPS、6GB 处理数千帧，是长序列在线重建的实用方案。

### 局限

1. **缓解但未解决遗忘**：超 1000 帧仍失败，需可选的 State Reset 兜底（符合 unexplored states 假说）。
2. **短序列增益小**：短序列相对 CUT3R 提升有限，未超强离线方法 VGGT。
3. **受基座上限约束**：本质是修 CUT3R 的状态更新规则，重建精度上限仍受 CUT3R 表达能力限制。

### 潜在改进方向

1. 更长序列训练 + TTT3R 更新规则联合，进一步逼近离线上界。
2. 探索 TTT3R 设计空间（不同梯度/学习率形式），推动更稳定可并行的 recurrent 架构。
3. 与显式回环/BA 结合，彻底解决超长序列漂移。

---

## 关联笔记

### 基于

- [[CUT3R]]：TTT3R 直接改进其状态更新规则，是本文的基座与主对比。
- [[Test-Time Training]]：核心分析视角，把 state 视为 fast weight。
- [[DeltaNet]] / [[Linear Attention]]：状态更新规则的数学联系来源。

### 对比

- [[VGGT]]：full-attention 离线上界（保留全历史但慢、易 OOM）。
- [[Spann3R]]：同为在线，位姿/重建被 TTT3R 超越。

### 方法相关

- [[Cross-Attention]]：对齐置信度的计算来源（state query × observation key）。
- [[Point Map]]：pointmap 输出（经 CUT3R 基座）。
- [[CroCo]] / DINO：可选图像 tokenizer。

### 谱系相关

- [[DUSt3R]]、[[MASt3R]]、[[MUSt3R]]：DUSt3R 谱系的成对/多视图重建工作。

---

## 速查卡片

> [!summary] TTT3R
> - **核心**：把 CUT3R 状态更新看作 Test-Time Training，用对齐置信度当闭式自适应学习率
> - **诊断**：CUT3R 隐含 $\beta_t=1.0$ + softmax 归一化 → 偏向新观测 → 灾难性遗忘
> - **方法**：$\beta_t=\sigma(\sum_m \mathbf{Q}_{\mathbf{S}}\mathbf{K}^\top_{\mathbf{X}})$，TokenLR 逐-token 门控（Eq.7-8）
> - **特点**：training-free、plug-and-play、无额外参数/计算
> - **关键结论**：长序列位姿 2× 于 CUT3R，20 FPS / 6GB / 数千帧
> - **结果**：TUM-D 1000帧 ATE 0.106（+Reset 0.093）；KITTI metric 0.110
> - **局限**：缓解未解决遗忘，>1000 帧需 State Reset
> - **项目**：rover-xingyu.github.io/TTT3R

---

*笔记创建时间: 2026-07-02*
