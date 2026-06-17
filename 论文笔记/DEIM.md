---
title: "DEIM: DETR with Improved Matching for Fast Convergence"
method_name: "DEIM"
authors: [DEIM Authors]
year: 2024
venue: arXiv
tags: [object-detection, detr, matching, real-time-detection]
arxiv: https://arxiv.org/abs/2412.04234
created: 2026-06-11
image_source: online
---

# 论文笔记：DEIM: DETR with Improved Matching for Fast Convergence

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[DEIM]] |
| 作者 | [DEIM Authors] |
| 年份 | 2024 |
| 会议/来源 | arXiv |
| 链接 | [arXiv](https://arxiv.org/abs/2412.04234) / [PDF](https://arxiv.org/pdf/2412.04234.pdf) |
| 相关工作 | [[D-FINE]]、[[RT-DETR]]、[[DETR]] |

---

## 一句话总结

> DEIM 通过 Dense O2O matching 和 matchability-aware loss 改善 DETR 训练效率，是实时 DETR 训练优化的重要方向。

---

## 核心贡献

1. **范式贡献**：DEIM 通过 Dense O2O matching 和 matchability-aware loss 改善 DETR 训练效率，是实时 DETR 训练优化的重要方向。
2. **方法贡献**：通过 Dense O2O matching 和 matchability-aware loss 改善 DETR 训练信号稀疏问题。
3. **工程/研究影响**：该工作成为后续 [[D-FINE]]、[[RT-DETR]]、[[DETR]] 的重要参照，影响了检测器的结构、训练或评测方式。

---

## 问题背景

### 要解决的问题

目标检测需要同时解决类别识别、空间定位、前景/背景判别和置信度排序。该工作针对当时检测系统中的关键瓶颈，提出新的结构、损失或训练策略。

### 现有方法的局限

- 早期或前一代方法通常在候选生成、特征表达、正负样本分配、框质量估计或开放类别泛化上存在瓶颈。
- 单看 AP 或 AP50 容易掩盖部署时的固定阈值可靠性；检测系统还必须处理 FP_bg、FP_loc、重复框和类别混淆。
- 对机器人场景而言，检测器还要考虑端侧延迟、长尾室内类别、置信度标定和与分割/深度头的协同。

### 本文的动机

本文试图用更清晰的检测范式或训练机制解决上述瓶颈，使检测器在精度、速度、可训练性或类别扩展能力上获得更好的平衡。

---

## 方法详解

### 模型架构 / 流程

通过 Dense O2O matching 和 matchability-aware loss 改善 DETR 训练信号稀疏问题。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[D-FINE]]、[[RT-DETR]]。
- 后续影响：[[DETR]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\mathcal{L}_{match-aware}=w_m\mathcal{L}_{det}
$$

**含义**：检测训练通常由分类、框回归和辅助质量/匹配损失组成，不同论文的差异在于每一项如何定义。

**符号说明**：

- $\mathcal{L}$：训练损失或检测目标。
- $s$ / $p$：分类、前景或匹配分数。
- $b$：预测框或框分布。
- $y$：真实标签、真实框或监督目标。

### 公式2：通用检测输出

$$
\hat{Y} = \{(\hat{c}_i, \hat{b}_i, \hat{s}_i)\}_{i=1}^N
$$

**含义**：检测器输出类别、边界框和分数的集合。不同论文的核心差异在于 $N$ 如何产生、$\hat{s}$ 表示什么、以及 $\hat{b}$ 如何被监督。

---

## 关键图表

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2412.04234](https://ar5iv.labs.arxiv.org/html/2412.04234)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 15 张、Table 6 张；下方按论文 HTML 顺序保留。

### Figure 1: (a) Faster : training is more compute-efficient

![Figure 1](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/x1.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 2: (b) Better : exceeding all real-time detectors

![Figure 2](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/x2.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 3: (a) O2M : 1 target and 4 pos.

![Figure 3](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/figs/O2M.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 4: (b) O2O : 1 target and 1 pos.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/figs/O2O.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 5: (c) Dense O2O by stitching : 4 targets and 4 pos.

![Figure 5](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/figs/DenseO2O.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 6: (d) Low-quality matching

![Figure 6](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/figs/Low-Quality-Matching.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 7: (e) Loss landscape of VFL

![Figure 7](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/figs/Loss_VFL.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 8: (f) Loss landscape of MAL

![Figure 8](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/figs/Loss_MAL.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 9: (a) Matching distribution

![Figure 9](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/x3.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 10: (b) Ratios between O2M and O2O

![Figure 10](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/x4.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 11: (a) Low quality : IoU = 0.05

![Figure 11](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/figs/2d_losses_1.5_0.75_0.05.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 12: (b) High quality : IoU = 0.95

![Figure 12](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/figs/2d_losses_1.5_0.75_0.95.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 13: Figure 4 : An illustrated example of our proposed novel training scheme for learning rate and data augmentation scheduler.

![Figure 13](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/x5.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 14: Figure 5 : # # \# Positive Samples with and without Dense O2O in One Epoch of Training. Base indicates without Dense O2O.

![Figure 14](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/x6.png)

**图表解读**：重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### Figure 15: Figure 6 : Qualitative Comparison between D-FINE-L and DEIM. In each paired image, the left is from D-FINE-L while the right is predicted by DEIM-D-FINE-L (Score threshold = = 0.5).

![Figure 15](https://ar5iv.labs.arxiv.org/html/2412.04234/assets/x7.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 Dense O2O 和 matchability-aware loss；可借鉴来缓解一对一匹配监督稀疏。

### 关键表格

#### Table 1: DEIM Table 1

| Model | #Epochs | #Params | GFLOPs | Latency (ms) | AP val | AP 50 v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 50 {}^{val}_{50} | AP 75 v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 75 {}^{val}_{75} | AP S v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 𝑆 {}^{val}_{S} | AP M v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 𝑀 {}^{val}_{M} | AP L v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 𝐿 {}^{val}_{L} |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO-based Real-time Object Detectors |  |  |  |  |  |  |  |  |  |  |
| YOLOv8-L [ 12 ] | 500 | 43 | 165 | 12.31 | 52.9 | 69.8 | 57.5 | 35.3 | 58.3 | 69.8 |
| YOLOv8-X [ 12 ] | 500 | 68 | 257 | 16.59 | 53.9 | 71.0 | 58.7 | 35.7 | 59.3 | 70.7 |
| YOLOv9-C [ 34 ] | 500 | 25 | 102 | 10.66 | 53.0 | 70.2 | 57.8 | 36.2 | 58.5 | 69.3 |
| YOLOv9-E [ 34 ] | 500 | 57 | 189 | 20.53 | 55.6 | 72.8 | 60.6 | 40.2 | 61.0 | 71.4 |
| Gold-YOLO-L [ 33 ] | 300 | 75 | 152 | 9.21 | 53.3 | 70.9 | - | 33.8 | 58.9 | 69.9 |
| YOLOv10-L ⋆ [ 32 ] | 500 | 24 | 120 | 7.66 | 53.2 | 70.1 | 58.1 | 35.8 | 58.5 | 69.4 |
| YOLOv10-X ⋆ [ 32 ] | 500 | 30 | 160 | 10.74 | 54.4 | 71.3 | 59.3 | 37.0 | 59.8 | 70.9 |
| YOLO11-L ⋆ [ 13 ] | 500 | 25 | 87 | 6.31 | 52.9 | 69.4 | 57.7 | 35.2 | 58.7 | 68.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: DEIM Table 2

| Model | #Epochs | #Params | GFLOPs | AP val | AP 50 v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 50 {}^{val}_{50} | AP 75 v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 75 {}^{val}_{75} | AP S v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 𝑆 {}^{val}_{S} | AP M v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 𝑀 {}^{val}_{M} | AP L v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 𝐿 {}^{val}_{L} |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet50 [ 14 ] -based |  |  |  |  |  |  |  |  |  |
| DETR-DC5 [ 3 ] | 500 | 41 | 187 | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1 |
| Anchor-DETR-DC5 [ 35 ] | 50 | 39 | 172 | 44.2 | 64.7 | 47.5 | 24.7 | 48.2 | 60.6 |
| Conditional-DETR-DC5 [ 26 ] | 108 | 44 | 195 | 45.1 | 65.4 | 48.5 | 25.3 | 49.0 | 62.2 |
| Efficient-DETR [ 36 ] | 36 | 35 | 210 | 45.1 | 63.1 | 49.1 | 28.3 | 48.4 | 59.0 |
| SMCA-DETR [ 11 ] | 108 | 40 | 152 | 45.6 | 65.5 | 49.1 | 25.9 | 49.3 | 62.6 |
| Deformable-DETR [ 45 ] | 50 | 40 | 173 | 46.2 | 65.2 | 50.0 | 28.8 | 49.2 | 61.7 |
| DAB-Deformable-DETR [ 21 ] | 50 | 48 | 195 | 46.9 | 66.0 | 50.8 | 30.1 | 50.4 | 62.5 |
| DAB-Deformable-DETR++ [ 21 ] | 50 | 47 | - | 48.7 | 67.2 | 53.0 | 31.4 | 51.6 | 63.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: DEIM Table 3

| Method | AP | AP 50 | AP 75 | AP s | AP m | AP l |
| --- | --- | --- | --- | --- | --- | --- |
| D-FINE-L | 56.0 | 87.2 | 59.4 | 29.0 | 46.1 | 54.6 |
| w/ DEIM | 57.5 | 87.6 | 62.9 | 33.2 | 48.7 | 55.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: DEIM Table 4

| Mosaic Prob. | Mixup Prob. | AP | AP 50 | AP 75 | AP s | AP m | AP l |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training 12 Epochs |  |  |  |  |  |  |  |
| 0.0 | 0.0 | 49.6 | 67.1 | 53.6 | 31.3 | 54.2 | 67.8 |
| 0.5 | 0.0 | 50.4 | 68.4 | 54.5 | 32.7 | 54.6 | 68.1 |
| 0.0 | 0.5 | 50.1 | 67.7 | 54.0 | 31.1 | 54.5 | 68.7 |
| 0.5 | 0.5 | 50.4 | 68.1 | 54.2 | 32.7 | 54.7 | 68.2 |
| Training 24 Epochs |  |  |  |  |  |  |  |
| 0.0 | 0.0 | 51.7 | 69.5 | 55.8 | 32.8 | 56.4 | 69.7 |
| 0.5 | 0.0 | 51.9 | 70.1 | 55.9 | 34.9 | 56.1 | 69.3 |
| 0.0 | 0.5 | 51.5 | 69.4 | 55.5 | 33.2 | 56.3 | 69.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: DEIM Table 5

| γ 𝛾 \gamma | 1.3 | 1.5 | 1.8 | 2.0 |
| --- | --- | --- | --- | --- |
| AP | 52.2 | 52.4 | 52.1 | 51.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: DEIM Table 6

| Epochs | Dense O2O | MAL | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- | --- |
| RT-DETRv2-R50 [ 24 ] |  |  |  |  |  |
| 72 |  |  | 53.4 | 71.6 | 57.4 |
| 36 | ✓ |  | 53.6 | 71.9 | 58.2 |
| ✓ | ✓ | 53.9 | 71.7 | 58.6 |  |
| D-FINE-L [ 27 ] |  |  |  |  |  |
| 72 |  |  | 54.0 | 71.6 | 58.4 |
| 36 | ✓ |  | 54.2 | 72.1 | 58.9 |
| ✓ | ✓ | 54.6 | 72.2 | 59.5 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

We introduce DEIM, an innovative and efficient training framework designed to accelerate convergence in real-time object detection with Transformer-based architectures (DETR). To mitigate the sparse supervision inherent in one-to-one (O2O) matching in DETR models, DEIM employs a Dense O2O matching strategy. This approach increases the number of positive samples per image by incorporating additional targets, using standard data augmentation techniques. While Dense O2O matching speeds up convergen

阅读时建议重点记录：

- 与同时代强基线相比，AP / AP50 / AP75 的提升是否来自核心方法，而非更强 backbone 或训练技巧。
- 小物体、中物体、大物体表现是否均衡。
- 速度和显存是否满足实时/端侧部署。
- 消融实验是否证明关键模块真的必要。

---

## 优缺点分析

### 优点

- 提供了一个在目标检测史上清晰可复用的思想节点。
- 对检测 pipeline 中某个关键瓶颈给出了结构化解法。
- 很多设计仍能迁移到现代 DINOv2/ViT 特征或 FCOS++/DETR 系框架中。

### 局限

需要区分论文的概念贡献和工程配方收益；部分方法依赖特定数据、增强或 backbone，直接迁移到冻结 DINOv2 时需要重新验证。

---

## 对 DINO-GeoSemMap 的启发

对 DINO-GeoSemMap 的价值主要体现在：是否能提升高置信召回、降低 FP_bg、改善 FP_loc、增强长尾类别覆盖，或为 FCOS++/ROI verifier 提供结构依据。

落到当前检测头优化，可对应到：

```text
FCOS(D2a+DFL)
  -> objectness / foreground branch
  -> hard negative mining
  -> ATSS / TAL assignment
  -> ROI verifier
  -> Objects365 + in-domain 数据
```

---

## 相关论文

- [[D-FINE]]
- [[RT-DETR]]
- [[DETR]]
