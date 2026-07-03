---
title: "LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding"
method_name: "LocateAnything"
authors: [Shihao Wang, Shilong Liu, Yuanguo Kuang, Xinyu Wei, Yangzhou Liu, Zhiqi Li, Yunze Man, Guo Chen, Andrew Tao, Guilin Liu, Jan Kautz, Lei Zhang, Zhiding Yu]
year: 2026
venue: arXiv
tags: [visual-grounding, object-detection, vision-language-model, parallel-decoding, multi-token-prediction, vlm]
zotero_collection: _待整理
image_source: mixed
arxiv_html: https://arxiv.org/html/2605.27365v1
created: 2026-06-26
---

# 论文笔记：LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA（合作单位：香港理工大学、普林斯顿大学、南京大学、UIUC） |
| 日期 | May 2026 |
| 项目主页 | https://research.nvidia.com/labs/lpr/locate-anything/ |
| 对比基线 | [[Rex-Omni]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.27365) / [Code](https://github.com/NVlabs/Eagle) / [HF Model](https://huggingface.co/nvidia/LocateAnything-3B) / [HF Demo](https://huggingface.co/spaces/nvidia/LocateAnything) |

---

## 一句话总结

> 把边界框作为不可分割的原子单元一次并行解码（[[Parallel Box Decoding|PBD]]），在保持框内几何一致性的同时打破 VLM 逐 token 串行解码瓶颈，速度提升 2.5×、高 IoU 定位质量也更优。

---

## 核心贡献

1. **Parallel Box Decoding (PBD)**: 首次将 [[Multi-Token Prediction|多 token 预测]] 应用到 VLM 检测/定位，以"框对齐"（box-aligned）方式将整个坐标集 $(x_1,y_1,x_2,y_2)$ 作为原子单元在**单步**并行预测，同时提升吞吐与精度。
2. **Hybrid 解码策略**: 检测不可靠的并行块并仅对问题块做局部 [[Next-Token Prediction|NTP]] 重解码（fallback），在保留大部分加速的同时降低最坏情况失败率。
3. **大规模数据引擎与 LocateAnything-Data**: 构建可扩展数据引擎，curate 出含 **138M** 自然语言查询、**785M** 标注框、**12M** 唯一图像的多域数据集，覆盖 6 类任务。
4. **SOTA 速度-精度前沿**: 在 layout grounding、long-tail detection、GUI grounding 等广泛评测上超越 SOTA，吞吐最高提升 **2.5×**（12.7 BPS），同时高 IoU 定位质量更高。

---

## 问题背景

### 要解决的问题

VLM 把视觉 grounding 与 detection 统一为"坐标 token 生成"问题：将每个 2D 框序列化成多个 1D token，逐 token 自回归生成。这一范式存在两个核心矛盾：
- 逐 token 解码与边界框几何的**耦合结构不匹配**（坐标 $x_1,y_1,x_2,y_2$ 之间强相关却被独立解码）；
- 严格串行生成造成**推理瓶颈**（高延迟、低吞吐）。

### 现有方法的局限

- **坐标表示**：现有方法用 **Textual Digits**（如 "1024" → "1","0","2","4"）或 **Quantized Tokens**（如 $x_1 \to y_1 \to x_2 \to y_2$），都把 2D 几何对象强行序列化为 1D 流，推理时只能 token-by-token 生成。
- **通用 [[Multi-Token Prediction|MTP]] 的不足**：语言建模里的 MTP（随机分块的 next-block 预测、或类似掩码扩散的随机掩码重建）是 **structure-agnostic（结构无关）** 的，只捕获共现驱动的相关性。对边界框这类紧耦合单元，这种监督会让模型学到跨越框边界、甚至跨越物体类别的 token 组合（见 Figure 2），从而拟合大量不可靠模式，放大误差传播，牺牲精度、可靠性和解码速度。

### 本文的动机

将 MTP 的预测块与**结构化单元对齐**：训练时把每个 box（或 point）当作一个原子单元，学习在一个并行步内预测完整坐标集 $(x_1,y_1,x_2,y_2)$。这种 **box-aligned** 训练目标避免对坐标 token 任意分块，既提升定位性能，又同时解锁并行解码的速度收益。

---

## 方法详解

### 模型架构

<!-- 见 Figure 3 -->

[[LocateAnything]] 基于一个 **native-resolution（原生分辨率）** 的 VLM：
- **输入**: 图像 $\mathcal{I}$ + 文本查询 $\mathcal{E}$
- **视觉编码器**: [[Moon-ViT]]（保留原生分辨率，提取细粒度空间细节以支持高精度定位），输出视觉 token $Z = \mathrm{Encoder}(\mathcal{I})$
- **Projector**: 两层 [[MLP Projector|MLP]] 桥接视觉与语言
- **语言解码器**: [[Qwen2.5]]（带 MTP 能力），将视觉 token 直接转换为一系列 box-aligned 的块级（block-level）预测
- **输出**: 定长 box-aligned 原子块序列 $\mathbf{B} = (b_1, b_2, \ldots, b_N)$

### Block-Based 输出形式化

放弃标准 NTP 坐标生成。连续坐标归一化到 $[0,1000]$ 并离散化为 token，再重组为块序列 $\mathbf{B}=(b_1,\dots,b_N)$。每个块 $b_i$ 是定长 $L=6$ 的原子单元（容纳一个边界框 + 两个结构 token，如 `<box>` 与 `</box>`），未占用位置用 `<null>` 填充以保证张量形状统一。

**四种功能块类型**（Figure 3）：
1. **Semantic Block（语义块）**: 编码语言身份。若表达超出单块容量则跨多个连续块切分。
2. **Box Block（框块）**: 用四个量化坐标表示边界框。
3. **Negative Block（负样本块）**: 显式指示被查询物体**不存在**（学会"弃权"，缓解幻觉）。
4. **End Block（结束块）**: 标志生成终止。

### 核心模块

#### 模块1: 双形式化（Dual-Formulation）训练设计

**设计动机**: 直接在训练阶段并行化输出会破坏模型固有的因果推理过程。为此引入双形式化训练，**联合优化两个对齐的表示**——NTP 序列保留因果推理能力，block-level MTP 形式化实现 box-aligned 预测。

**具体实现**:
- 构造单一拼接输入序列：$x_{\text{all}} = x_{\text{vis}} \oplus x_{\text{q}} \oplus x_{\text{ntp}} \oplus x_{\text{blk}}$（$\oplus$ 为序列拼接）。
- $x_{\text{vis}}$（视觉）与 $x_{\text{q}}$（文本查询）为共享上下文；$x_{\text{ntp}}$ 为标准 NTP 输入序列；$x_{\text{blk}}$ 为 block-wise MTP 输入序列。两者是同一 ground truth 的 **token-level** 与 **block-level** 两种格式。
- $x_{\text{blk}}$ 由 $x_{\text{ntp}}$ 从左到右遍历、按块规则切分填充构造：每块保留**第一个 token** 作为预测上下文，其余 token 替换为 `[mask]`，提示模型在单步内并行预测块内所有掩码 token。
- 当块大小设为 1 时，该 MTP 形式化自然退化为标准 NTP。

#### 模块2: 联合 NTP-MTP 训练的专用 Attention Mask

**设计动机**: 双序列训练的核心挑战是如何隔离 NTP 与 MTP 两条流，同时让两者都能利用共享上下文。通过专用注意力掩码（Figure 4）以三种行为控制信息流：

**具体实现**:
- **Causal Attention for NTP（NTP 因果注意力）**: 共享上下文（$x_{\text{vis}}, x_{\text{q}}$）与 NTP 序列（$x_{\text{ntp}}$）使用因果掩码，只能 attend 到前序 token；并被严格限制**不能 attend 到 $x_{\text{blk}}$**，防止数据泄漏。这与推理时标准 [[KV Cache]] 用法对齐。
- **Causal Flow Across Blocks（跨块因果流）**: $x_{\text{blk}}$ 中不同块之间严格因果。活动块内 token 可 attend 到共享上下文和所有已提交的块，但看不到未来块。这种历史可见性使模型学习不同框预测间的依赖，有效缓解重复或遗漏框。
- **Bidirectional Intra-Block Attention（块内双向注意力）**: 同一块内的 token 共享双向注意力，捕获块内复杂内部关系（如一组坐标间的几何依赖），在单个功能单元内同时求解所有内部 token。
- 两条流相互隔离防止泄漏，同时共同 attend 共享上下文。

#### 模块3: On-Demand 推理模式与 Hybrid Fallback

**设计动机**: 并行解码在高度模糊场景面临 exploration-exploitation 困境，存在两类失败（Figure 5）：
- **Format Irregularity（格式不规则）**: 复杂多实例多类别场景中，模型在类别边界处犹豫，错误混合结构与坐标 token（如 `<box><211></ref><911><887></box>`）。
- **Spatial Ambiguity（空间模糊）**: 物体密集规则排列（行/列网格）时，MTP 会模糊空间边界，输出介于两物体之间的中间坐标，产生低 IoU。

**Ambiguity Trigger（模糊触发器）**: 当同时满足两条件时激活 fallback——(1) top-1 坐标 token 概率 **< 0.7**；(2) top-5 坐标 token 在 $[0,1000]$ 归一化空间的 max-min 差 **> 80**。触发后丢弃受损块，回退到最近已验证前缀，用 NTP 自回归生成该问题块，完成后无缝切回 MTP。

**三种推理模式**:
1. **Slow Mode (NTP)**: 逐 token 标准 NTP，最高稳定性，用于高精度标注、最终精修、离线评测。
2. **Fast Mode (MTP)**: 用 MTP 预测 box-aligned 块，丢弃 `<null>` 填充 token，已提交 token 存入 KV cache 作为因果上下文；最高吞吐，适合 on-device 机器人/具身智能体等延迟受限场景。
3. **Hybrid Mode**: 默认用 Fast Mode，并行输出不可靠时无缝回退 Slow Mode；保留大部分并行加速且维持鲁棒输出，面向需兼顾速度与精度的生产管线。

---

## 关键公式

### 公式1: [[Parallel Box Decoding|块级联合概率]]

$$
P(\mathbf{B} \mid \mathcal{Z}, \mathcal{E}) = \prod_{i=1}^{N} P\big(b_i \mid b_{<i}, \mathcal{Z}, \mathcal{E}\big)
$$

**含义**: 给定视觉特征 $\mathcal{Z}$ 和文本查询 $\mathcal{E}$，块序列 $\mathbf{B}$ 的联合概率分解为各块的条件概率乘积——这是**块级（block-level）半自回归**而非 token 级自回归，块内并行、块间因果。

**符号说明**:
- $\mathbf{B} = (b_1, \ldots, b_N)$: 由 $N$ 个原子块构成的输出序列
- $b_i$: 第 $i$ 个块，定长 $L=6$ 的原子单元
- $b_{<i}$: 第 $i$ 块之前的所有已提交块
- $\mathcal{Z}$: 视觉特征（$Z = \mathrm{Encoder}(\mathcal{I})$）
- $\mathcal{E}$: 文本查询

### 公式2: 双形式化输入拼接

$$
x_{\text{all}} = x_{\text{vis}} \oplus x_{\text{q}} \oplus x_{\text{ntp}} \oplus x_{\text{blk}}
$$

**含义**: 将共享上下文、NTP 流、MTP 流拼接为单一输入序列，使 NTP 和 block-level MTP 两种表示在同一前向中联合训练。

**符号说明**:
- $\oplus$: 序列拼接操作
- $x_{\text{vis}}$: 视觉 token（共享上下文）
- $x_{\text{q}}$: 文本查询 token（共享上下文）
- $x_{\text{ntp}}$: 标准 NTP 输入序列（token-level 表示）
- $x_{\text{blk}}$: block-wise MTP 输入序列（block-level 表示）

### 公式3: [[Cross-Entropy Loss|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{ntp}} + \mathcal{L}_{\text{mtp}}
$$

**含义**: 由专用 attention mask 引导，对两条序列联合最小化交叉熵损失。NTP 损失保留因果推理能力，MTP 损失训练 box-aligned 并行预测。消融显示联合训练把 Slow Mode 上限从 50.1 推到 52.1 F1。

**符号说明**:
- $\mathcal{L}_{\text{ntp}}$: NTP 序列的交叉熵损失（token-level）
- $\mathcal{L}_{\text{mtp}}$: MTP 块级序列的交叉熵损失（block-level）

---

## 关键图表

### Figure 1: Versatile Tasks / 任务总览与解码范式演进

![Figure 1|74](https://arxiv.org/html/2605.27365v1/x1.png)
**说明**: 上半部分展示 LocateAnything 在统一 VLM 下支持的多样定位任务：Multi Object Referring、Pointing、Layout Grounding、GUI Grounding、Detection、OCR。下半部分对比三种解码范式演进——**Textual Digit Decoding**（逐位拼坐标，21 步）→ **Quantized Coordinate Decoding**（坐标 token 顺序预测，10 步）→ **Parallel Box Decoding**（单前向预测一个几何单元，仅 **2 步**，兼具任意长度 + token 效率 + 并行解码）。

### Figure 2: Comparison of Token Decoding Methods / 解码方法对比

![Figure 2](https://arxiv.org/html/2605.27365v1/x2.png)

**说明**: NTP 逐个生成坐标值；标准 [[Multi-Token Prediction|MTP]] 产生不规则分布与非连贯、无结构（Unstructured）的模式（跨越框边界）；本文 **PBD** 在并行步内生成单个原子框（或 point）单元，确保 box-aligned 的结构化输出。

### Figure 3: Architecture and Block-Based Output / 架构与块输出表示

![Figure 3](https://arxiv.org/html/2605.27365v1/x3.png)

**说明**: 输入 Image + Text query，经 [[Moon-ViT]]（native resolution）→ 两层 MLP Projector → [[Qwen2.5]]（LLM with MTP）。输出形式化为定长 box-aligned 原子块序列，定义四类功能块——**Semantic Block**（文本查询，如 `<ref> Person <ref>`）、**Box Block**（定位，如 `<box><100><675><218><800></box>`）、**Negative Block**（无物体，`<box> None <box>`）、**End Block**（终止 `<im_end>`），可重复 $\times N$。

### Figure 4: Attention Mask for Joint NTP-MTP Training / 联合训练注意力掩码

![Figure 4](https://arxiv.org/html/2605.27365v1/x4.png)

**说明**: 共享上下文与 NTP 流用因果注意力（蓝色，Profiling + Next Token Prediction）；MTP 块（橙色，$B_1, B_2$）遵循跨块 block-causal 模式、块内 token 共享双向注意力（Multi Token Prediction）。两条流相互隔离防泄漏，同时共同 attend 共享上下文。

### Figure 5: Corrected NTP Re-decoding / NTP 纠错重解码

![Figure 5](https://arxiv.org/html/2605.27365v1/x5.png)

**说明**: 当并行解码遇到 **Format Irregularity**（格式不规则，如 `<box><211></ref>...` 错误混入结构 token）或 **Spatial Ambiguity**（空间模糊，输出 `<121>` 这种介于两物体间的中间坐标）时，模型丢弃错误块并回退标准 NTP 重解码（图中绿色为 NTP Re-decoding 修正结果），保证鲁棒预测。

### Figure 6: LocateAnything-Data Overview / 数据集总览

![Figure 6](https://arxiv.org/html/2605.27365v1/x6.png)

**说明**: 三个饼图展示数据分布——**Queries (138M)**: Detection 66.9% / UI 16.5% / Referring 7.3% / Layout 3.5% / Pointing 2.2% / OCR；**Boxes (785M)**: Detection 83.1% 占绝对主导；**Images (12M)**: Detection 33.9% / UI 22.9% / Referring 18.7% / OCR 13.9% / Layout 5.7% / Pointing 4.9%。底部详细列出各任务的具体数据源（如 Object365 47.3M、OpenImages 41.0M、GroundCUA 5.5M 等）。

### Figure 7: Ablation on Box Ordering & Decoding Speed / 框排序与解码速度消融

![Figure 7](https://arxiv.org/html/2605.27365v1/x7.png)

**说明**: **左**——四种框排序策略对 F1 的影响：**X-Y Corner Order**（按左上角 x 后 y 排序）最佳 52.1，优于 Center Distance(35)、Area(28)、Random(25.1)，故作为数据构建默认。**右**——随预测框数量（20→300）增加，Textual/Quantized 等 NTP 方法生成时间陡增（严重延迟瓶颈），而 Parallel 方法时间几乎不增，吞吐从 12 BPS 升至约 25 BPS，密集场景实现 2×~6× 加速。

### Figure 8: Qualitative Results / 定性结果

![Figure 8](https://arxiv.org/html/2605.27365v1/x8.png)

**说明**: 三行分别展示 1-10 框、10-20 框、20+ 框的测试样例，不同颜色标注 attribute/part/reasoning/spatial 四类查询。模型在不同场景域、任意图像分辨率、自由文本查询和任意物体数量下都能稳定定位，展现强鲁棒性（如 "the illuminated bulbs on the red letters" 准确定位 LOVE 灯泡）。

### Figure 9: Multi-Targets Grounding Data Engine / 多目标 grounding 数据引擎

![Figure 9](https://arxiv.org/html/2605.27365v1/x9.png)

**说明**: **上**——对有 gt 框的检测数据集，用每个框类别名作 prompt 让 [[Qwen2.5]]/Qwen3-VL 合成包含属性/空间关系/推理线索的物体中心查询，再喂给 [[Molmo]] 预测候选点，保留落在 gt 框内的点作为可靠监督。**下**——对大量无标注高质量图像，Qwen3-VL 直接生成多样查询，用 Molmo 预测点后接 [[SAM|SAM 3]] 生成框，或直接用 [[Rex-Omni]] 生成框，所有框最终经 Qwen3-VL 后验证。

### Figure 10 & 11: Dataset Distribution / 数据集分布

![Figure 10](https://arxiv.org/html/2605.27365v1/x10.png)

![Figure 11](https://arxiv.org/html/2605.27365v1/x11.png)

**说明**: Figure 10——每查询目标数分布（log 尺度）呈长尾：多数查询对应少量目标，但有相当一部分涉及大量实例。Figure 11——各域查询长度（去除模板文本后的词数）分布，反映不同 grounding 范式与语言模式。

### Figure 12-14: 附录定性对比

![[LocateAnything_fig12.png|600]]

![Figure 13](https://arxiv.org/html/2605.27365v1/x13.png)

![Figure 14](https://arxiv.org/html/2605.27365v1/x14.png)

**说明**: Figure 12——Referring Expression Comprehension (REC) 定性对比；Figure 13——Dense Object Detection (DOD) 定性对比；Figure 14——OCR 定性对比。

### Table 1: LVIS 与 COCO 检测结果（F1@IoU, BPS = Boxes Per Second 吞吐）

| Method | Throughput | LVIS 0.5 | LVIS 0.95 | LVIS Mean | COCO 0.5 | COCO 0.95 | COCO Mean |
|--------|-----------|----------|-----------|-----------|----------|-----------|-----------|
| Grounding DINO-Swin-T | - | 47.7 | <u>22.7</u> | 38.8 | 69.8 | <u>23.0</u> | <u>56.6</u> |
| DINO-Swin-L (闭集) | - | - | - | - | **75.6** | **25.4** | **62.1** |
| Qwen3-VL-8B | 1.0 | 61.5 | 20.2 | 44.8 | 62.8 | 14.0 | 45.7 |
| SEED1.5-VL | - | **65.6** | 19.5 | 46.7 | 71.3 | 14.3 | 51.4 |
| [[Rex-Omni]]-3B | <u>5.0</u> | <u>64.3</u> | 20.7 | <u>46.9</u> | <u>72.0</u> | 15.9 | 52.9 |
| **LocateAnything-3B** | **12.7** | 62.3 | **31.1** | **50.7** | 70.1 | 19.3 | 54.7 |

**说明**: LocateAnything-3B 吞吐 **12.7 BPS**（远超所有方法），LVIS mean F1 +3.8%、COCO +1.8% 优于同参数量的 Rex-Omni，且高 IoU（0.95）质量显著领先（LVIS 31.1 vs 20.7）。

### Table 2: 密集检测 Dense200 与 VisDrone

| Method | Dense200 0.5 | Dense200 0.95 | Dense200 Mean | VisDrone 0.5 | VisDrone 0.95 | VisDrone Mean |
|--------|--------------|---------------|---------------|--------------|---------------|---------------|
| Grounding DINO-Swin-T | 36.9 | **19.7** | 33.1 | 55.2 | **3.9** | <u>38.5</u> |
| SEED1.5-VL | <u>76.9</u> | 5.3 | 53.2 | 55.9 | 0.6 | 27.4 |
| Rex-Omni-3B | **78.4** | 10.3 | <u>58.3</u> | <u>61.6</u> | 1.5 | 35.8 |
| **LocateAnything-3B** | 74.0 | <u>18.5</u> | **58.7** | **63.0** | <u>3.2</u> | **39.9** |

**说明**: 密集/微小物体场景下 VisDrone mean 39.9（大幅超 Rex-Omni 35.8），Dense200 mean 58.7，在重叠密集环境下展现优越的边界划分与实例分离能力。

### Table 3: GUI Grounding（ScreenSpot-Pro，Avg）

| Method | Avg |
|--------|-----|
| Rex-Omni-3B | 36.8 |
| GTA1-7B | 50.1 |
| Qwen3-VL-30B-A3B* | 53.7 |
| GUI-Owl-32B | <u>58.0</u> |
| **LocateAnything-3B** | **60.3** |

**说明**: 在 ScreenSpot-Pro 上达到 SOTA mean F1 **60.3**，超越通用大模型 Qwen3-VL-30B-A3B 和专用 UI 模型 GUI-Owl-32B（仅 3B 参数）。

### Table 4: 文档版面 grounding 与 OCR（F1@IoU Mean）

| Method | DocLayNet | M6Doc | TotalText |
|--------|-----------|-------|-----------|
| DocLayout-YOLO（专用） | **81.1** | - | - |
| Rex-Omni-3B | 70.7 | <u>55.6</u> | <u>40.6</u> |
| **LocateAnything-3B** | <u>76.8</u> | **70.1** | **43.3** |

**说明**: DocLayNet 76.8、M6Doc 70.1 树立新标准，大幅超越 Rex-Omni。

### Table 5: 指代表达理解 REC（HumanRef / RefCOCOg）

| Method | HumanRef Mean | RefCOCOg val Mean | RefCOCOg test Mean |
|--------|---------------|-------------------|--------------------|
| Qwen3-VL-8B | 72.0 | 75.2 | 74.6 |
| SEED1.5-VL | <u>81.6</u> | 71.9 | 73.2 |
| Rex-Omni-3B | 79.9 | 73.6 | 74.3 |
| **LocateAnything-3B** | 78.7 | <u>76.7</u> | <u>77.6</u> |

**说明**: HumanRef 78.7 mean F1，RefCOCOg 上与顶级模型保持高度竞争力，证明精细空间推理可延伸到复杂指代任务。

### Table 6: 消融实验（COCO 数据集，R=Recall, P=Precision）

**(a) 坐标表示**

| 表示 | Throughput | F1 |
|------|-----------|-----|
| Textual | 1.3 | 49.1 |
| Quantized | 3.9 | 50.1 |
| **PBD (Slow)** | 3.9 | **52.1** |
| PBD (Fast) | **16.9** | 49.6 |
| PBD (Hybrid) | <u>13.2</u> | <u>51.6</u> |

**(b) MTP 形式化**

| 方法 | Throughput | F1 |
|------|-----------|-----|
| SDLM-B4 | 5.2 | <u>46.5</u> |
| SDLM-B8 | <u>6.7</u> | 45.8 |
| Block Diff-B6 | 4.7 | 44.8 |
| **PBD (Fast)** | **16.9** | **49.6** |

**(c) 解码模式与损失**

| $\mathcal{L}_{\text{ntp}}$ | $\mathcal{L}_{\text{blk}}$ | Mode | Throughput | F1 |
|------|------|------|-----------|-----|
| ✓ | | Slow | 3.9 | 50.1 |
| | ✓ | Fast | <u>16.7</u> | 47.2 |
| ✓ | ✓ | Slow | 3.9 | **52.1** |
| ✓ | ✓ | Fast | **16.9** | 49.6 |
| ✓ | ✓ | Hybrid | 13.2 | <u>51.6</u> |

**关键发现**:
- **(a)** PBD (Slow) 的 box-aligned 形式化（52.1）优于 Textual(49.1)/Quantized(50.1)，证明 box-aligned 监督比 1D 序列化提供更强空间推理监督。
- **(b)** box-aligned PBD 在吞吐（16.9 BPS）和精度（49.6）上都碾压 structure-agnostic 的 SDLM、Block Diffusion；后者增大块大小只带来边际吞吐增益却持续降低 F1（严格的速度-精度 trade-off）。
- **(c)** 联合训练（$\mathcal{L}_{\text{ntp}}+\mathcal{L}_{\text{blk}}$）把 Slow Mode 上限从 50.1 推到 52.1；Hybrid Mode 在 13.2 BPS 下达到 51.6 F1，完美平衡速度与精度。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LocateAnything-Data | 138M 查询 / 785M 框 / 12M 图 / >22M 负样本 | 6 域多任务自建 | 训练 |
| [[LVIS]] | - | 长尾分布 | 测试 |
| [[COCO]] | - | 通用检测 + 消融隔离 PBD 收益 | 训练/测试 |
| VisDrone / Dense200 | - | 密集/微小物体 | 测试 |
| [[RefCOCO|RefCOCOg]] / HumanRef | - | 指代表达理解 REC | 测试 |
| ScreenSpot-Pro | - | GUI grounding | 测试 |
| DocLayNet / M6Doc / TotalText | - | 版面 grounding 与 OCR | 测试 |

### 实现细节

- **架构**: [[Moon-ViT]] 视觉编码器 + 两层 MLP Projector + [[Qwen2.5]] 语言解码器（3B），native resolution
- **训练**: 4 阶段渐进式（见 Table 7）
  - Stage 1（视觉概念初始化）: 仅 Caption，LR 2e-4，仅训 MLP，64 GPU，2000 步
  - Stage 2（综合多模态学习）: General VQA，LR 4e-5，全参数，256 GPU，20000 步
  - Stage 3（综合检测与 grounding）: 注入 138M 查询，LR 4e-5，全参数，max seq 25600，256 GPU，25000 步
  - Stage 4（密集检测增强）: 通用数据降至 20%、增大多物体图像比例，LR 1e-5，256 GPU，5000 步
- **优化器**: AdamW，weight decay 0.01，Cosine LR
- **关键系统技术**: **Stream Packing**（在线流式打包，Weighted Sampling + Best-Fit Buffering + Big-Rocks-First Seeding，>95% 打包效率）、**MagiAttention**（处理异构 attention mask 的分布式注意力框架）
- **推理超参**: nucleus sampling，temperature 0.7，top-p 0.9，repetition penalty 1.1，块大小 $n_{\text{future}}=6$，最大新生成 token 8192，BF16，batch size 1
- **吞吐评测**: 单卡 NVIDIA H100，batch size 1，默认 Hybrid Mode

### 数据引擎细节

- **From Detection Datasets**: 用 gt 框类别名 prompt Qwen3-VL 合成物体中心查询 → Molmo 预测点 → 保留落在 gt 框内的点。
- **From Unlabeled Images**: Unsplash / SA-1B 无标注图 → Qwen3-VL 生成查询 → Molmo 预测点 → SAM 3 转框，或直接 Rex-Omni 生成框 → Qwen3-VL 后验证。
- **负样本构建**: 显式生成指向图中不存在物体的查询并赋予 negative block，学会"弃权"以缓解幻觉。

### 可视化结果（定性观察）

1. **组合式 grounding**: 良好处理 attribute/part/spatial/reasoning 风格查询，空间对齐一致。
2. **对大实例数鲁棒**: 目标从稀疏到密集时预测框仍结构化且准确，Stage-2 多物体训练强化了密集定位。
3. **杂乱场景可靠定位**: 遮挡、重复纹理、网格密集布局下框紧凑且分离良好，Hybrid 模式通过检测不可靠并行块、回退 NTP 进一步稳定 hard case。

---

## 批判性思考

### 优点
1. **问题诊断精准**：明确指出"token 级解码与框几何耦合结构不匹配"以及"通用 MTP 是 structure-agnostic"两个根因，方法（box-aligned）正中靶心。
2. **速度-精度双赢**：不同于一般加速方法的 trade-off，PBD 同时提升吞吐（12.7 BPS，2.5×）和高 IoU 精度，消融充分（Table 6 三个子表）。
3. **工程完备**：Hybrid fallback + 明确的 ambiguity trigger（概率<0.7、max-min>80）使并行解码在 hard case 下不崩；Stream Packing + MagiAttention 解决双形式化训练的系统级难题。
4. **数据引擎可扩展**：138M 查询自动合成 + 多模型协同（Qwen3-VL/Molmo/SAM 3/Rex-Omni）+ 后验证，且显式构造负样本缓解幻觉。

### 局限性
1. **仅监督微调**：作者明确指出未用 RL，承认 RL 是优化块级解码策略、降低 fallback 频率、改善 hard dense/long-tail 探索的重要下一步。
2. **Hybrid 仍依赖启发式触发器**：ambiguity trigger 的阈值（0.7 / 80）是手工设定的，跨域泛化性与最优性存疑。
3. **闭集专用检测器仍占优**：在 COCO 高精度上 DINO-Swin-L（62.1 mean）仍优于 LocateAnything（54.7），生成式范式在纯检测精度上尚未完全追平专用模型。
4. **块大小固定 $L=6$**：单块仅容纳一个框 + 两结构 token，对超长语义表达需跨多块切分，复杂度与上限受限。

### 潜在改进方向
1. 引入 RL/强化学习优化解码策略与 fallback 触发（作者已点明）。
2. 可学习的 ambiguity trigger 替代手工阈值。
3. 探索可变块大小或层次化块结构以适配不同任务粒度。

### 可复现性评估
- [x] 代码开源（GitHub: NVlabs/Eagle）
- [x] 预训练模型（HF: nvidia/LocateAnything-3B）
- [x] 训练细节完整（Table 7 四阶段超参 + Stream Packing/MagiAttention 描述）
- [x] 数据集可获取（LocateAnything-Data 数据源在 Figure 6 / Table 8 列出，多为开源）

---

## 关联笔记

### 基于
- [[Qwen2.5]]: 语言解码器主干（带 MTP 能力）
- [[Moon-ViT]]: native-resolution 视觉编码器
- [[Multi-Token Prediction]]: PBD 的并行预测思想来源
- [[Next-Token Prediction]]: Slow Mode 与 fallback 的解码基础

### 对比
- [[Rex-Omni]]: 最相关工作，同为 VLM 统一检测/grounding（quantized 坐标），主要对比基线
- [[Grounding DINO]]: 开集专用检测器基线
- [[Block Diffusion]]: structure-agnostic 的块级解码对比对象

### 方法相关
- [[Parallel Box Decoding]]: 核心方法
- [[Visual Grounding]]: 任务领域
- [[Object Detection]]: 任务领域
- [[Bounding Box]]: 几何原子单元
- [[KV Cache]]: 推理时缓存机制

### 硬件/数据相关
- [[COCO]] / [[LVIS]] / [[RefCOCO]] / [[Objects365]]: 评测与训练数据集
- [[Molmo]] / [[SAM]]: 数据引擎中的点/框生成模型

---

## 速查卡片

> [!summary] LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
> - **核心**: 把边界框作为原子单元一次并行解码（PBD），打破 VLM 逐 token 串行瓶颈
> - **方法**: box-aligned MTP + 双形式化 NTP-MTP 联合训练 + Hybrid NTP fallback；Moon-ViT + Qwen2.5 (3B)
> - **结果**: 12.7 BPS（2.5× 加速），LVIS/COCO/VisDrone/GUI/DocLayout 多任务 SOTA 或领先，高 IoU 质量更优
> - **数据**: LocateAnything-Data，138M 查询 / 785M 框 / 12M 图
> - **代码**: https://github.com/NVlabs/Eagle ｜ Model: https://huggingface.co/nvidia/LocateAnything-3B

---

*笔记创建时间: 2026-06-26*
