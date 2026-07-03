---
title: "RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer"
method_name: "RT-DETRv2"
authors: [RT-DETRv2 Authors]
year: 2024
venue: arXiv
tags: [object-detection, real-time-detection, detr, training-tricks]
arxiv: https://arxiv.org/abs/2407.17140
created: 2026-06-11
image_source: online
---

# 论文笔记：RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[RT-DETRv2]] |
| 作者 | [RT-DETRv2 Authors] |
| 年份 | 2024 |
| 会议/来源 | arXiv |
| 链接 | [arXiv](https://arxiv.org/abs/2407.17140) / [PDF](https://arxiv.org/pdf/2407.17140.pdf) |
| 相关工作 | [[RT-DETR]]、[[D-FINE]]、[[DEIM]] |

---

## 一句话总结

> RT-DETRv2 在 RT-DETR 基础上通过一系列训练与工程改进，增强实时 DETR 的性能和可复现性。

---

## 核心贡献

1. **范式贡献**：RT-DETRv2 在 RT-DETR 基础上通过一系列训练与工程改进，增强实时 DETR 的性能和可复现性。
2. **方法贡献**：在 RT-DETR 上系统加入训练和工程 bag-of-freebies，提升实时 DETR baseline。
3. **工程/研究影响**：该工作成为后续 [[RT-DETR]]、[[D-FINE]]、[[DEIM]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

在 RT-DETR 上系统加入训练和工程 bag-of-freebies，提升实时 DETR baseline。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[RT-DETR]]、[[D-FINE]]。
- 后续影响：[[DEIM]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\mathcal{L}_{total}=\mathcal{L}_{det}+\mathcal{L}_{aux}
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2407.17140](https://ar5iv.labs.arxiv.org/html/2407.17140)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 2 张、Table 4 张；下方按论文 HTML 顺序保留。

### Figure 1: Reference Figure: RT-DETR family overview; RT-DETRv2 inherits this detector structure and improves training/deployment details.

![Figure 1](https://zhao-yian.github.io/RTDETR/static/images/overview.jpg)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 selective multi-scale deformable attention 和训练 bag-of-freebies；适合拆成可控消融。

### Figure 2: Reference Figure: RT-DETR hybrid encoder, the architectural base that RT-DETRv2 refines with bag-of-freebies.

![Figure 2](https://zhao-yian.github.io/RTDETR/static/images/encoder.jpg)

**图表解读**：重点看 selective multi-scale deformable attention 和训练 bag-of-freebies；适合拆成可控消融。

### 关键表格

#### Table 1: RT-DETRv2 Table 1

| Model | Backbone | lr backbone | lr det |
| --- | --- | --- | --- |
| RT-DETRv2-S | ResNet18 | 1e-4 | 1e-4 |
| RT-DETRv2-M | ResNet34 | 5e-5 | 1e-4 |
| RT-DETRv2-L | ResNet50 | 1e-5 | 1e-4 |
| RT-DETRv2-X | ResNet101 | 1e-6 | 1e-4 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: RT-DETRv2 Table 2

| Model | Backbone | Dataset | #Params (M) | FPS bs=1 | AP val | AP 50 v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 50 {}^{val}_{50} |
| --- | --- | --- | --- | --- | --- | --- |
| RT-DETR-S | ResNet18 | COCO | 20 | 217 | 46.5 | 63.8 |
| RT-DETR-M | ResNet34 | COCO | 31 | 161 | 48.9 | 66.8 |
| RT-DETR-M ∗ | ResNet50 | COCO | 36 | 145 | 51.3 | 69.6 |
| RT-DETR-L | ResNet50 | COCO | 42 | 108 | 53.1 | 71.3 |
| RT-DETR-X | ResNet101 | COCO | 76 | 74 | 54.3 | 72.7 |
| RT-DETRv2-S | ResNet18 | COCO | 20 | 217 | 47.9 ( ↑ ↑ \uparrow 1.4 ) | 64.9 ( ↑ ↑ \uparrow 1.1 ) |
| RT-DETRv2-M | ResNet34 | COCO | 31 | 161 | 49.9 ( ↑ ↑ \uparrow 1.0 ) | 67.5 ( ↑ ↑ \uparrow 0.7 ) |
| RT-DETRv2-M ∗ | ResNet50 | COCO | 36 | 145 | 51.9 ( ↑ ↑ \uparrow 0.6 ) | 69.9 ( ↑ ↑ \uparrow 0.3 ) |
| RT-DETRv2-L | ResNet50 | COCO | 42 | 108 | 53.4 ( ↑ ↑ \uparrow 0.3 ) | 71.6 ( ↑ ↑ \uparrow 0.3 ) |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: RT-DETRv2 Table 3

| Model | Sampling method | #Points | AP val | AP 50 v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 50 {}^{val}_{50} |
| --- | --- | --- | --- | --- |
| RT-DETRv2-S | grid_sample | 86,400 | 47.9 | 64.9 |
| RT-DETRv2-S | grid_sample | 64,800 | 47.8 | 64.8 ( ↓ ↓ \downarrow 0.1 ) |
| RT-DETRv2-S | grid_sample | 43,200 | 47.7 | 64.7 ( ↓ ↓ \downarrow 0.2 ) |
| RT-DETRv2-S | grid_sample | 21,600 | 47.3 | 64.3 ( ↓ ↓ \downarrow 0.6 ) |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: RT-DETRv2 Table 4

| Model | Backbone | Sampling method | AP val | AP 50 v ​ a ​ l subscript superscript absent 𝑣 𝑎 𝑙 50 {}^{val}_{50} |
| --- | --- | --- | --- | --- |
| RT-DETRv2-S | ResNet18 | discrete_sample | 47.4 | 64.8 ( ↓ ↓ \downarrow 0.1 ) |
| RT-DETRv2-M | ResNet34 | discrete_sample | 49.2 | 67.1 ( ↓ ↓ \downarrow 0.4 ) |
| RT-DETRv2-M ∗ | ResNet50 | discrete_sample | 51.4 | 69.7 ( ↓ ↓ \downarrow 0.2 ) |
| RT-DETRv2-L | ResNet50 | discrete_sample | 52.9 | 71.3 ( ↓ ↓ \downarrow 0.3 ) |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

In this report, we present RT-DETRv2, an improved Real-Time DEtection TRansformer (RT-DETR). RT-DETRv2 builds upon the previous state-of-the-art real-time detector, RT-DETR, and opens up a set of bag-of-freebies for flexibility and practicality, as well as optimizing the training strategy to achieve enhanced performance. To improve the flexibility, we suggest setting a distinct number of sampling points for features at different scales in the deformable attention to achieve selective multi-scale

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

- [[RT-DETR]]
- [[D-FINE]]
- [[DEIM]]
