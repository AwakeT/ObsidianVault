---
title: "YOLOX: Exceeding YOLO Series in 2021"
method_name: "YOLOX"
authors: [Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun]
year: 2021
venue: arXiv
tags: [object-detection, yolo, anchor-free, simota, real-time-detection]
arxiv: https://arxiv.org/abs/2107.08430
created: 2026-06-11
image_source: online
---

# 论文笔记：YOLOX: Exceeding YOLO Series in 2021

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[YOLOX]] |
| 作者 | [Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun] |
| 年份 | 2021 |
| 会议/来源 | arXiv |
| 链接 | [arXiv](https://arxiv.org/abs/2107.08430) / [PDF](https://arxiv.org/pdf/2107.08430.pdf) |
| 相关工作 | [[YOLOv1]]、[[RetinaNet]]、[[FCOS]]、[[ATSS]]、[[YOLO-World]] |

---

## 一句话总结

> YOLOX 将 YOLO 系列改造成 anchor-free、decoupled head 和 SimOTA 动态匹配的现代一阶段检测器。

---

## 核心贡献

1. **范式贡献**：YOLOX 将 YOLO 系列改造成 anchor-free、decoupled head 和 SimOTA 动态匹配的现代一阶段检测器。
2. **方法贡献**：去掉 anchor，采用 decoupled head 和 SimOTA 动态匹配，同时保留 objectness 分支。
3. **工程/研究影响**：该工作成为后续 [[YOLOv1]]、[[RetinaNet]]、[[FCOS]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

去掉 anchor，采用 decoupled head 和 SimOTA 动态匹配，同时保留 objectness 分支。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[YOLOv1]]、[[RetinaNet]]。
- 后续影响：[[FCOS]]、[[ATSS]]、[[YOLO-World]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
Cost=\lambda_{cls}L_{cls}+\lambda_{iou}L_{iou}
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2107.08430](https://ar5iv.labs.arxiv.org/html/2107.08430)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 4 张、Table 5 张；下方按论文 HTML 顺序保留。

### Figure 1: Refer to caption

![Figure 1](https://ar5iv.labs.arxiv.org/html/2107.08430/assets/x1.png)

**图表解读**：重点看 decoupled head、anchor-free 输出和 SimOTA；它是 FCOS++ 引入 objectness 与动态匹配的工程参考。

### Figure 2: Refer to caption

![Figure 2](https://ar5iv.labs.arxiv.org/html/2107.08430/assets/x2.png)

**图表解读**：重点看 decoupled head、anchor-free 输出和 SimOTA；它是 FCOS++ 引入 objectness 与动态匹配的工程参考。

### Figure 3: Figure 2 : Illustration of the difference between YOLOv3 head and the proposed decoupled head. For each level of FPN feature, we first adopt a 1 × 1 1 1 1\times 1 conv layer to reduce the feature channel to 256 and then add two parallel branches with two 3 × 3 3 3 3\times 3 conv layers each for classification and regression tasks respectively. IoU branch is added on the regression branch.

![Figure 3](https://ar5iv.labs.arxiv.org/html/2107.08430/assets/x3.png)

**图表解读**：重点看 decoupled head、anchor-free 输出和 SimOTA；它是 FCOS++ 引入 objectness 与动态匹配的工程参考。

### Figure 4: Figure 3 : Training curves for detectors with YOLOv3 head or decoupled head. We evaluate the AP on COCO val every 10 epochs. It is obvious that the decoupled head converges much faster than the YOLOv3 head and achieves better result finally.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2107.08430/assets/x4.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 decoupled head、anchor-free 输出和 SimOTA；它是 FCOS++ 引入 objectness 与动态匹配的工程参考。

### 关键表格

#### Table 1: YOLOX Table 1

| Methods | AP (%) | Parameters | GFLOPs | Latency | FPS |
| --- | --- | --- | --- | --- | --- |
| YOLOv3-ultralytics 2 | 44.3 | 63.00 M | 157.3 | 10.5 ms | 95.2 |
| YOLOv3 baseline | 38.5 | 63.00 M | 157.3 | 10.5 ms | 95.2 |
| +decoupled head | 39.6 (+1.1) | 63.86 M | 186.0 | 11.6 ms | 86.2 |
| +strong augmentation | 42.0 (+2.4) | 63.86 M | 186.0 | 11.6 ms | 86.2 |
| +anchor-free | 42.9 (+0.9) | 63.72 M | 185.3 | 11.1 ms | 90.1 |
| +multi positives | 45.0 (+2.1) | 63.72 M | 185.3 | 11.1 ms | 90.1 |
| +SimOTA | 47.3 (+2.3) | 63.72 M | 185.3 | 11.1 ms | 90.1 |
| +NMS free (optional) | 46.5 (-0.8) | 67.27 M | 205.1 | 13.5 ms | 74.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: YOLOX Table 2

| Models | AP (%) | Parameters | GFLOPs | Latency |
| --- | --- | --- | --- | --- |
| YOLOv5-S | 36.7 | 7.3 M | 17.1 | 8.7 ms |
| YOLOX-S | 39.6 (+2.9) | 9.0 M | 26.8 | 9.8 ms |
| YOLOv5-M | 44.5 | 21.4 M | 51.4 | 11.1 ms |
| YOLOX-M | 46.4 (+1.9) | 25.3 M | 73.8 | 12.3 ms |
| YOLOv5-L | 48.2 | 47.1 M | 115.6 | 13.7 ms |
| YOLOX-L | 50.0 (+1.8) | 54.2 M | 155.6 | 14.5 ms |
| YOLOv5-X | 50.4 | 87.8 M | 219.0 | 16.0 ms |
| YOLOX-X | 51.2 (+0.8) | 99.1 M | 281.9 | 17.3 ms |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: YOLOX Table 3

| Models | AP (%) | Parameters | GFLOPs |
| --- | --- | --- | --- |
| YOLOv4-Tiny [ 30 ] | 21.7 | 6.06 M | 6.96 |
| PPYOLO-Tiny | 22.7 | 4.20 M | - |
| YOLOX-Tiny | 32.8 (+10.1) | 5.06 M | 6.45 |
| NanoDet 3 | 23.5 | 0.95 M | 1.20 |
| YOLOX-Nano | 25.3 (+1.8) | 0.91 M | 1.08 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: YOLOX Table 4

| Models | Scale Jit. | Extra Aug. | AP (%) |
| --- | --- | --- | --- |
| YOLOX-Nano | [0.5, 1.5] | - | 25.3 |
| [0.1, 2.0] | MixUp | 24.0 (-1.3) |  |
| YOLOX-L | [0.1, 2.0] | - | 48.6 |
| [0.1, 2.0] | MixUp | 49.5 (+0.9) |  |
| [0.1, 2.0] | Copypaste [ 6 ] | 49.4 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: YOLOX Table 5

| Method | Backbone | Size | FPS | AP (%) | AP 50 | AP 75 | AP S | AP M | AP L |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (V100) |  |  |  |  |  |  |  |  |  |  |
| YOLOv3 + ASFF* [ 18 ] | Darknet-53 | 608 | 45.5 | 42.4 | 63.0 | 47.4 | 25.5 | 45.7 | 52.3 |  |
| YOLOv3 + ASFF* [ 18 ] | Darknet-53 | 800 | 29.4 | 43.9 | 64.1 | 49.2 | 27.0 | 46.6 | 53.4 |  |
| EfficientDet-D0 [ 28 ] | Efficient-B0 | 512 | 98.0 | 33.8 | 52.2 | 35.8 | 12.0 | 38.3 | 51.2 |  |
| EfficientDet-D1 [ 28 ] | Efficient-B1 | 640 | 74.1 | 39.6 | 58.6 | 42.3 | 17.9 | 44.3 | 56.0 |  |
| EfficientDet-D2 [ 28 ] | Efficient-B2 | 768 | 56.5 | 43.0 | 62.3 | 46.2 | 22.5 | 47.0 | 58.4 |  |
| EfficientDet-D3 [ 28 ] | Efficient-B3 | 896 | 34.5 | 45.8 | 65.0 | 49.3 | 26.6 | 49.4 | 59.8 |  |
| PP-YOLOv2 [ 11 ] | ResNet50-vd-dcn | 640 | 68.9 | 49.5 | 68.2 | 54.4 | 30.7 | 52.9 | 61.2 |  |
| PP-YOLOv2 [ 11 ] | ResNet101-vd-dcn | 640 | 50.3 | 50.3 | 69.0 | 55.3 | 31.6 | 53.9 | 62.4 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector -- YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one

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

objectness + decoupled head + SimOTA 是 FCOS++ 的直接工程参考。

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

- [[YOLOv1]]
- [[RetinaNet]]
- [[FCOS]]
- [[ATSS]]
- [[YOLO-World]]
