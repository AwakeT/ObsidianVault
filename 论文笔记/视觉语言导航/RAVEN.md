---
title: "RAVEN: Long-Horizon Reasoning & Navigation with a Visuo-Spatio-Temporal Memory"
method_name: "RAVEN"
authors: [Yixun Hu, Zhicheng Zheng, Lihan Zha, Chunwei Xing, Rajdeep Singh, Omar Hossain, Antonio Loquercio, Dhruv Shah]
year: 2026
venue: arXiv
tags: [long-horizon-navigation, visual-embedding-memory, retrieval-augmented-generation, agentic-memory, vlm-agent, robot-question-answering]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2606.25206v1
created: 2026-07-06
---

# 论文笔记：RAVEN: Long-Horizon Reasoning & Navigation with a Visuo-Spatio-Temporal Memory

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Princeton University, University of Pennsylvania |
| 日期 | June 2026 |
| 项目主页 | [ravenmem.github.io](https://ravenmem.github.io) |
| 对比基线 | [[ReMEmbR]], [[VLM Only]], [[Embedder Only]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.25206) / [Code](https://github.com/princeton-prism/RAVEN) |

---

## 一句话总结

> 直接以视觉嵌入（而非图像转文字caption）构建可检索的视觉-空间-时间记忆，配合工具调用的 [[VLM]] agent 迭代检索，实现长时程机器人问答与导航，检索成本降 10×。

---

## 核心贡献

1. **视觉-空间-时间记忆（[[Visuo-Spatio-Temporal Memory]]）**: 将视觉嵌入 + 位姿 + 时间戳作为三元组存入向量数据库，绕过有损的图像转文字 caption 瓶颈，保留细粒度视觉语义
2. **工具调用 agentic 检索**: [[VLM]] agent 以有限状态机形式迭代调用 text/time/position/image 四种检索工具，闭环推理直到证据充分，缩短注意力窗口、避免全轨迹穷举
3. **RAVEN-QA 基准 + 全面评测**: 新提出 273 条 QA 的长时程视觉-空间-时间检索基准，在 4 个真实 + 19 个仿真环境上一致超越 caption-based 方法，达到 250× 存储压缩且不掉点、10× 检索效率

---

## 问题背景

### 要解决的问题
终身部署的机器人需要一个紧凑、可扩展的记忆：既保留细粒度视觉语义、又能在空间和时间上定位观测、并支持高效存储与检索，从而回答长时程查询并导航到目标。

### 现有方法的局限
- **闭词表语义地图 / 点云**：把索引限制在预定义词表内，一旦查询细节（"带粉色蝴蝶结的蓝帽子"）超出词表就失败
- **原始视频流**：完整但太大、冗余，无法直接作记忆
- **Caption-based 开放词表（如 [[ReMEmbR]]）**：把观测 caption 成文字再嵌入，引入有损的 **caption 瓶颈**——欠caption（漏掉细节）或过caption（引入噪声/幻觉），纹理和精确空间关系难以言语化而丢失
- **[[CLIP-Fields]]** 类：靠 per-scene 优化限制了跨环境可扩展性；RNN 记忆有长期遗忘；retrieval-free 上下文模型无法扩展到超长序列

### 本文的动机
现代视觉基础模型的多模态嵌入空间已高度与文本对齐（scaling law 驱动）。直接在紧凑视觉嵌入上操作，可保留最大信息密度，同时借向量数据库的 sub-$O(N)$ 检索维持可扩展性，且**训练无关（training-free）**。

---

## 方法详解

### 模型架构

<!-- 使用 [[概念]] 内联链接所有技术术语 -->

RAVEN 采用 **两阶段（记忆构建 + 查询执行）+ agentic 检索循环** 架构：
- **输入**: 自然语言查询 $q$ + Phase I 探索采集的 egocentric RGB 帧序列、深度图、位姿、时间戳
- **Backbone**: 预训练多模态编码器（[[CLIP]] / [[SigLIP]] / Seed1.6-Embedding / QQMM-v2）+ 向量数据库（[[FAISS]] / [[Milvus]]）+ [[VLM]] agent（Gemini-2.5-Flash / GPT-5.2 / Gemini-3-Pro 等）
- **核心模块**: [[Visuo-Spatio-Temporal Memory]] 用于存储，[[Retrieval-Augmented Generation]] 工具用于检索，[[VLM]] agent 用于迭代推理
- **输出**: (i) 文本答案，(ii) 空间目标 $\hat{p}=(\hat{x},\hat{y},\hat{z})$，或 (iii) 时间参考；空间目标交给 [[A-star Algorithm|A*]] 规划器执行导航
- **训练**: 无需训练，即插即用

### 核心模块

#### 模块1: 视觉-空间-时间记忆检索（Sec 4.1）

**设计动机**: 仅靠视觉嵌入无法在空间和时间上 grounding，因此为每帧构造复合记忆条目。

**具体实现**:
- 用多模态编码器把 RGB 帧 $o_i$ 直接编码为紧凑潜表示 $z_i$（即 [[Multimodal Embedding]]）
- 与机器人位姿 $p_i$、世界时钟时间戳 $t_i$ 一同索引进向量数据库，形成三元组记忆 $\{m_i\}_N$
- 支持基于语义相似度、空间邻近、时间窗口的灵活检索
- 检索作为可调用工具暴露给 agent，用 [[FAISS]] / [[Milvus]] 实现 sub-$O(N)$ 最近邻搜索

#### 模块2: 工具调用推理 agent（Sec 4.2）

**设计动机**: 与其单次检索直接返回，不如用强 [[VLM]] agent 迭代提升检索精度。

**具体实现**:
- agent 维护当前上下文 $R_0$ 作为工作记忆，判断证据是否足够生成答案；不足则调用检索动作
- 四种原语工具：
  - **text-based**: 把 agent 生成的查询 $\hat{q}$ 与存储视觉记忆 $z_i$ 用余弦距离 $1-\langle f_{enc}(\hat{q}), z_i\rangle$ 比较
  - **time-based**: 从指定时间戳返回连续记忆（采用 **前向时序排序** 而非距离排序，见公式/附录）
  - **position-based**: 找与提议位置 $(\hat{x}_q,\hat{y}_q,\hat{z}_q)$ 的空间最近邻
  - **image-based**: 把用户查询扩展为图像（Appendix B）
- 每次检索返回 $K$ 条 $\{m'_i\}_K$ 追加到上下文，形成闭环推理的有限状态机
- 一个 trick：检索响应中**附带相似度分数** 给 VLM 作置信线索，提升推理（ReMEmbR 未提供）

#### 模块3: 探索与导航执行（Sec 4.3）

**设计动机**: 无缝接入通用机器人规划-控制-导航栈。

**具体实现**:
- **Phase I 探索**: [[Frontier Exploration|贪心前沿探索]] 采集 RGB-D，用 RGB-D SLAM 重建 3D 占据地图，投影到 2D occupancy map
- **Phase II 导航**: RAVEN 预测的空间坐标交给 [[A-star Algorithm|A*]] 轨迹规划器，PD 控制器驱动 Unitree Go1 跟踪 waypoints，闭合检索-规划环

---

## 关键公式

### 公式1: [[Visuo-Spatio-Temporal Memory|三元组记忆结构]]

$$
\{m_i\}_N := \{(p_1, t_1, z_1, o_1), (p_2, t_2, z_2, o_2), \ldots, (p_N, t_N, z_N, o_N)\}
$$

**含义**: 每个观测帧存为「位姿 + 时间戳 + 视觉嵌入 + 原图」四元组，构成可按语义/空间/时间检索的记忆库。

**符号说明**:
- $p_i = (x_i, y_i, z_i) \in \mathbb{R}^3$: 第 $i$ 帧机器人的 3D 位姿
- $t_i$: 世界时钟时间戳
- $z_i$: 多模态编码器对 RGB 帧 $o_i$ 的视觉嵌入
- $o_i \in \mathbb{R}^{H\times W\times C}$: 原始 RGB 帧
- $N$: 记忆库总条目数

### 公式2: [[Retrieval-Augmented Generation|文本检索余弦距离]]

$$
d(\hat{q}, z_i) = 1 - \langle f_{enc}(\hat{q}), z_i \rangle
$$

**含义**: text-based 检索工具用 agent 生成查询 $\hat{q}$ 的嵌入与存储视觉嵌入 $z_i$ 的余弦距离度量相关性，返回 top-$K$。

**符号说明**:
- $f_{enc}(\cdot)$: 多模态编码器的文本编码分支
- $\langle\cdot,\cdot\rangle$: 归一化内积（余弦相似度）
- $\hat{q}$: agent 生成的检索查询

### 公式3: 上下文迭代更新

$$
R_{\tau+1} = R_\tau \cup \{m'_i\}_K
$$

**含义**: 每次检索把 $K$ 条记忆条目追加到推理步 $\tau$ 的工作记忆上下文，VLM 评估 $R_{\tau+1}$ 是否证据充分，形成闭环。

**符号说明**:
- $R_\tau$: 第 $\tau$ 步的工作记忆上下文
- $\{m'_i\}_K$: 本次检索返回的 $K$ 条记忆（含位置、时间戳、图像）

### 公式4: [[SPL|内存效率]] 度量

$$
\alpha(\%) := \frac{N^{\text{retrieved}}_{\text{memory}}}{N_{\text{memory}}} \times 100\%, \qquad E := \frac{1}{\alpha(\%)}
$$

**含义**: $\alpha$ 是单次查询检索的记忆占比，其倒数 $E$ 作为内存效率——检索越少内容达成任务，效率越高。RAVEN 的 $E$ 比 VLM Only 高 10×。

**符号说明**:
- $N^{\text{retrieved}}_{\text{memory}}$: 每查询检索的记忆条数
- $N_{\text{memory}}$: 记忆库总条数

---

## 关键图表

<!-- 图片外链优先：arXiv HTML -->

### Figure 1: Overview / 系统概览

![Figure 1|74](https://arxiv.org/html/2606.25206v1/x1.png)

**说明**: RAVEN 以视觉嵌入作为长期记忆。左侧「自主记忆构建阶段」机器人 [[Frontier Exploration|前沿探索]] 采集帧，用多模态编码器算出帧级/段级视觉嵌入，与位姿、时间戳一起存入记忆库；右侧「查询与执行阶段」[[VLM]] agent 迭代调用检索工具取 top-$k$ 记忆生成答案，并用轨迹规划器引导机器人前往预测位置（例：在 [2.05, -6.62, 0.45] 的绿色办公椅上找到排球）。

### Figure 2: Pipeline / RAVEN 流水线与评测设置

![Figure 2](https://arxiv.org/html/2606.25206v1/x2.png)

**说明**: （左）记忆构建阶段用多模态编码器嵌入帧，把图像嵌入、位置、时间向量存入向量数据库；用户提问时启动向量库查询循环，[[VLM]] 给出答案。（右）安全锥查询的推理示例——VLM 用函数调用多次检索（top-1 相似度 0.087→0.061），觉得"证据不足，再搜"直到确认。

### Figure 3: Benchmark Results / 三基准性能

![Figure 3](https://arxiv.org/html/2606.25206v1/x3.png)

**说明**: 在 FindingDory、RAVEN-QA、NaVQA 上的性能对比（Embedder Only / [[ReMEmbR]] / VLM Only / Ours）。RAVEN-QA 用 Gemini-2.5-Flash + QQMM-v2，NaVQA 用 Gemini-3-Pro + QQMM-v2。RAVEN 在各项上一致领先，误差棒为多次运行标准差。

### Figure 4: Real-world Rollouts / 真实机器人部署

![Figure 4](https://arxiv.org/html/2606.25206v1/x4.png)

**说明**: 四个真实场景（Lab1/Office/Lab2/Public Area）的机器人部署 rollout。左为鸟瞰重建图（RAVEN 不可见），右为各方法在不同查询（安全锥、排球、灭火器、"打球"推理等）上的成功/失败对照，涵盖 dominant/secondary/reasoning/information-recall 类别。

### Figure 5: Compactness / 压缩可扩展性

![Figure 5](https://arxiv.org/html/2606.25206v1/figs/rebuttal_figs/judgement_accuracy_vs_stride_final.png)

**说明**: RAVEN 记忆高度紧凑可扩展。虚线标出 250× 压缩下仅掉 10% 性能；在 0.12 fps 下采样仍保留 90% 能力，相对原始 RGB 达 22,315× 压缩。

### Table 1: RAVEN-QA robot-ego 系统精度（真实 + 仿真）

| Method / Embedder | Gemini-2.5-Flash | GPT-5.2 | Gemini-3-Pro | Gemma3-27b | Qwen3-VL-32b | Embedder Only |
|-------------------|------------------|---------|--------------|------------|--------------|---------------|
| **Real-World (Scene1~4, 21 queries)** | | | | | | |
| QQMM-v2 | 92.38±3.33 | 93.33 | 90.00 | 67.62 | 30.00 | 66.67 |
| Seed (Closed) | 85.71±5.02 | 93.33 | 90.47 | 45.71 | 31.90 | 66.67 |
| SigLIP | 84.76±3.01 | **97.14** | 90.00 | 47.62 | 35.24 | 42.86 |
| VLM Only | 85.24±5.70 | 92.86 | 86.19 | 14.29 | 4.76 | N/A |
| ReMEmbR (QQMM-v2) | 76.67±3.51 | 74.76 | 80.48 | 14.76 | 28.57 | N/A |
| **Habitat Sim (Scene1~19, 157 queries)** | | | | | | |
| QQMM-v2 | 80.47 | 75.58 | 86.62 | 54.99 | 26.11 | 61.78 |
| Seed (Closed) | 81.74 | 76.22 | 86.84 | 60.30 | 27.39 | 62.42 |
| SigLIP | 81.10 | 71.97 | 86.20 | 56.05 | 22.51 | 61.78 |
| VLM Only | 76.65 | 80.47 | 88.75 | 11.04 | 9.77 | N/A |
| ReMEmbR (QQMM-v2) | 54.99 | 57.32 | 61.36 | 15.71 | 20.38 | N/A |

**说明**: 闭源模型 + 视觉嵌入的 RAVEN 在真实机器人任务上全面优于 caption-based（[[ReMEmbR]]）和 VLM-only。中等规模开源 VLM（27B~32B）下三种方法都不如 Embedder Only 基线——离线无网部署时直接用 embedder-only 反而更可取。

### Table 2: FindingDory 高层成功率（长时程子集，均值 3022 帧/视频）

| 指标 | RAVEN (QQMM-v2) | ReMEmbR (QQMM-v2) | MixedBread | VLM Only | Embedder Only | Random |
|------|-----------------|-------------------|------------|----------|---------------|--------|
| Overall Accuracy | **32.2** | 25.0 | 25.2 | 28.2 | 18.8 | 9.2 |
| Single Spatial | **45.1** | 36.7 | 36.0 | 36.0 | 31.0 | 13.4 |
| Single Temporal | 24.8 | 17.1 | 18.6 | **28.1** | 8.1 | 6.9 |
| Multi-Goal Nav | **6.7** | 4.5 | 4.5 | 2.2 | 3.4 | 0.5 |
| $\alpha$%（检索占比）↓ | **7.43%** | 11.17% | 10.68% | 100% | 0.0% | N/A |

**说明**: 长时程子集上 RAVEN 整体精度最高，且只检索 7.43% 记忆（VLM Only 需 100%），即内存效率约 10× 优势。全数据集（ep1~100）上 VLM Only 整体 38.2 略高于 RAVEN 35.1，但 RAVEN 只用 10% 记忆。

### Table 8: RAVEN-QA human-ego（Simple/Hard 拆分，节选）

| Split / Encoder | Gemini-3-Pro | GPT-5 | Gemma3-27b | Embedder Only |
|-----------------|--------------|-------|------------|---------------|
| Simple: Seed1.6-Embed | 100.0 | 98.1 | 74.1 | 85.2 |
| Simple: QQMM-v2 | 98.1 | 96.3 | 85.2 | 90.7 |
| Hard: Seed1.6-Embed | **95.1** | 82.9 | 53.7 | 63.4 |
| Hard: QQMM-v2 | 92.7 | 82.9 | 51.2 | 63.4 |

**说明**: 性能差距从 Simple 的 ~13% 拉大到 Hard 的 30%，证明直接视觉嵌入在细粒度语义（secondary 物体）上稳健，而 caption 瓶颈在此失效。

### Table 10: 嵌入模型排行榜（Text/Image 查询精度，Appendix B.5）

| Model | Group | Size | Text-Query Overall | Dom. | Sec. | Image-Query |
|-------|-------|------|--------------------|------|------|-------------|
| QQMM-Embed-v2 | Contrastive | 8.3B | **91.8** | 100 | 82.6 | 89.8 |
| Seed-1.6 | Closed-source | Unknown | 85.7 | 100 | 69.6 | 87.8 |
| SigLIP-so400M-384 | Contrastive | 880M | 81.6 | 95.2 | 65.2 | 55.1 |
| CLIP-ViT-H-14 | Contrastive | 1B | 79.6 | 95.2 | 60.9 | 69.4 |
| PaliGemma2-3B | Autoregressive | 3B | 51.0 | 66.7 | 43.5 | 67.4 |
| Random | Baseline | — | 5.5 | 5.5 | 5.5 | 5.5 |

**说明**: 对比学习类嵌入（QQMM-v2/SigLIP）显著优于自回归 VLM 隐藏态；开源嵌入已与闭源持平；SigLIP 性价比最高，适合端侧。QQMM-v2 与 Seed-1.6 精度最高。

### Table 12: 时序检索排序方式对比（FindingDory ep1）

| 排序方式 | Forward Chronological (Ours) | Centered Chronological | Nearest Neighbor |
|----------|------------------------------|------------------------|------------------|
| Accuracy | **49.00%** | 32.65% | 42.86% |

**关键发现**: 时间检索用「从目标时间戳起的前向时序排序」最优，因为它对齐了多模态模型在视频语料上训练的因果/时序归纳偏置；不居中是为避免 attention sink 让邻近时刻被忽略。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| **RAVEN-QA**（自建） | 273 QA，489… 4真实+19仿真+web视频 | 长时程视觉-空间-时间检索，标注 5 类查询 | 训练/测试 |
| [[NaVQA]] | 基于 CODa 真实城市导航 | 描述/空间/时间查询，短中长时程 | 测试 |
| [[FindingDory]] | 基于 Habitat，长时程 loco-manipulation | 单/多目标时空导航，ep91~100 均 3000 帧 | 测试 |

RAVEN-QA 数据源：YouTube 室内外 tour 视频、自录人类视角视频（有相机抖动）、Habitat 仿真 19 场景、真实机器人（Unitree Go1 + Earthrover Mini）。查询分类：Dominant(81) / Secondary(81) / Reasoning(90) / Information Recall(18) / Spatial Understanding(3)，共 273 条。

### 实现细节

- **多模态编码器**: CLIP / SigLIP / Seed1.6-Embedding / QQMM-v2（QQMM 在 RTX 5070 Ti 上因显存用 4-bit 量化）
- **VLM agent**: Gemini-2.5-Flash/Pro、Gemini-3-Pro、GPT-4o/5/5.2、Gemma3、Qwen3-VL
- **向量检索**: [[FAISS]] / [[Milvus]]，余弦/欧氏距离
- **超参**: RAVEN-QA top-K=5、max tool call=3；FindingDory top-K 更大（Time 200/Text 30/Position 1）、max tool call=15；温度 0.0001~0.7
- **硬件（真实部署）**: Unitree Go1 + ZED 2i 相机 + Jetson Orin NX 16GB；RGB-D SLAM + [[A-star Algorithm|A*]] + PD 控制

### 可视化结果
- 真实部署最高成功率 97.1%（SigLIP + GPT-5.2），主配置 92.4%（QQMM-v2 + Gemini-2.5-Flash）
- 即使探索覆盖仅 ~20% footprint 仍能近满分（Fig 18），因为 RAVEN 靠视觉嵌入不需全状态访问
- caption 失效案例：ReMEmbR 把"chicken thighs"欠caption成"meat in a refrigerator"（漏检），或对"fireplace"过caption导致检索到无壁炉帧（幻觉）

---

## 批判性思考

### 优点
1. 直击 caption 瓶颈这一真实痛点，用视觉嵌入 + 向量检索的组合优雅且 training-free、即插即用
2. 评测极其充分：多 VLM × 多 embedder × 真实+仿真+web，附录给出完整消融（排序方式、相似度分数、top-K、探索覆盖度）
3. 250×/22,315× 压缩 + 10× 检索效率的量化优势对长时程/终身部署有实际价值
4. 真机部署（Unitree Go1）验证了系统闭环可行性

### 局限性
1. 检索受限于预训练多模态编码器的对齐能力，无领域微调
2. 记忆当前是静态数据库，难以处理物体频繁变位置/状态的高动态场景（作者自述）
3. 中等规模开源 VLM（27~32B）下反而不如 embedder-only，说明 agentic 收益依赖强 VLM
4. FindingDory 全数据集上 VLM Only 整体精度反超 RAVEN（38.2 vs 35.1），优势主要在长时程与效率维度

### 潜在改进方向
1. 领域特定微调对齐嵌入与机器人任务
2. 记忆的动态更新与剪枝机制，应对环境变化
3. 结合更强的时序/动态建模处理高动态场景

### 可复现性评估
- [x] 代码开源（github.com/princeton-prism/RAVEN）
- [ ] 预训练模型（依赖第三方 embedder/VLM）
- [x] 训练细节完整（training-free + 超参表）
- [ ] 数据集可获取（RAVEN-QA 未明确开放）

---

## 关联笔记

### 基于
- [[Visuo-Spatio-Temporal Memory]]: 核心记忆表示
- [[Retrieval-Augmented Generation]]: 检索机制基础
- [[Multimodal Embedding]]: 视觉嵌入来源

### 对比
- [[ReMEmbR]]: caption-based 记忆导航，主要对比基线
- [[VLM Only]]: 直接喂全帧给 VLM 的无检索基线
- [[Embedder Only]]: 无语言模型的纯嵌入检索基线

### 方法相关
- [[CLIP]] / [[SigLIP]]: 候选多模态编码器
- [[FAISS]] / [[Milvus]]: 向量数据库
- [[Frontier Exploration]]: Phase I 探索策略
- [[A-star Algorithm]]: 导航执行规划器
- [[Agentic Memory]]: agent 迭代检索范式

### 硬件/数据相关
- [[FindingDory]] / [[NaVQA]]: 评测基准
- [[Success Rate]] / [[SPL]]: 评测指标

---

## 速查卡片

> [!summary] RAVEN
> - **核心**: 用视觉嵌入 + 位姿 + 时间戳三元组记忆替代有损 caption，training-free 的长时程机器人问答与导航
> - **方法**: 多模态编码器嵌入帧 → 向量库存储 → VLM agent 迭代调用 text/time/position/image 四种检索工具闭环推理 → A* 导航
> - **结果**: 真实部署最高 97.1% SR；250× 存储压缩不掉点，10× 检索效率；超越 ReMEmbR 与 VLM Only
> - **代码**: github.com/princeton-prism/RAVEN

---

*笔记创建时间: 2026-07-06*
