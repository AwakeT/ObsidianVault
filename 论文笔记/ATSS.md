---
title: "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection"
method_name: "ATSS"
authors: [Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, Stan Z. Li]
year: 2019
venue: CVPR
tags: [object-detection, sample-assignment, anchor-free, anchor-based]
arxiv: https://arxiv.org/abs/1912.02424
created: 2026-06-11
image_source: online
---

# 论文笔记：Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[ATSS]] |
| 作者 | [Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, Stan Z. Li] |
| 年份 | 2019 |
| 会议/来源 | CVPR |
| 链接 | [arXiv](https://arxiv.org/abs/1912.02424) / [PDF](https://arxiv.org/pdf/1912.02424.pdf) |
| 相关工作 | [[FCOS]]、[[GFL]]、[[TOOD]]、[[YOLOX]] |

---

## 一句话总结

> ATSS 指出 anchor-based 和 anchor-free 的核心差异不在 anchor，而在正负样本分配策略，并提出自适应样本选择。

---

## 核心贡献

1. **范式贡献**：ATSS 指出 anchor-based 和 anchor-free 的核心差异不在 anchor，而在正负样本分配策略，并提出自适应样本选择。
2. **方法贡献**：为每个 GT 在各 FPN level 选 top-k 候选，用 IoU 均值+方差自适应确定正样本阈值。
3. **工程/研究影响**：该工作成为后续 [[FCOS]]、[[GFL]]、[[TOOD]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

为每个 GT 在各 FPN level 选 top-k 候选，用 IoU 均值+方差自适应确定正样本阈值。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[FCOS]]、[[GFL]]。
- 后续影响：[[TOOD]]、[[YOLOX]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
T_g=\mu(\mathcal{I}_g)+\sigma(\mathcal{I}_g)
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1912.02424](https://ar5iv.labs.arxiv.org/html/1912.02424)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 6 张、Table 6 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: Definition of positives ( 1 ) and negatives ( 0 ). Blue box, red box and red point are ground-truth, anchor box and anchor point. (a) RetinaNet uses IoU to select positives ( 1 ) in spatial and scale dimension simultaneously. (b) FCOS first finds candidate positives ( ? ) in spatial dimension, then selects final positives ( 1 ) in scale dimension.

![Figure 1](https://ar5iv.labs.arxiv.org/html/1912.02424/assets/cls_v2.jpg)

**图表解读**：重点看不同 FPN level 的候选点如何由 IoU 均值+方差自适应筛选；这是替换固定 center sampling 的直接依据。

### Figure 2: (a) Positive sample

![Figure 2](https://ar5iv.labs.arxiv.org/html/1912.02424/assets/reg1.jpg)

**图表解读**：重点看不同 FPN level 的候选点如何由 IoU 均值+方差自适应筛选；这是替换固定 center sampling 的直接依据。

### Figure 3: (b) RetinaNet

![Figure 3](https://ar5iv.labs.arxiv.org/html/1912.02424/assets/reg2.jpg)

**图表解读**：重点看不同 FPN level 的候选点如何由 IoU 均值+方差自适应筛选；这是替换固定 center sampling 的直接依据。

### Figure 4: (c) FCOS

![Figure 4](https://ar5iv.labs.arxiv.org/html/1912.02424/assets/reg3.jpg)

**图表解读**：重点看不同 FPN level 的候选点如何由 IoU 均值+方差自适应筛选；这是替换固定 center sampling 的直接依据。

### Figure 5: Refer to caption

![Figure 5](https://ar5iv.labs.arxiv.org/html/1912.02424/assets/atss1.jpg)

**图表解读**：重点看不同 FPN level 的候选点如何由 IoU 均值+方差自适应筛选；这是替换固定 center sampling 的直接依据。

### Figure 6: Refer to caption

![Figure 6](https://ar5iv.labs.arxiv.org/html/1912.02424/assets/atss2.jpg)

**图表解读**：重点看不同 FPN level 的候选点如何由 IoU 均值+方差自适应筛选；这是替换固定 center sampling 的直接依据。

### 关键表格

#### Table 1: ATSS Table 1

| Method | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- |
| RetinaNet (#A=1) | 37.0 | 55.1 | 39.9 | 21.4 | 41.2 | 48.6 |
| RetinaNet (#A=1) + ATSS | 39.3 | 57.5 | 42.8 | 24.3 | 43.3 | 51.3 |
| FCOS | 37.8 | 55.6 | 40.7 | 22.1 | 41.8 | 48.8 |
| FCOS + Center sampling | 38.6 | 57.4 | 41.4 | 22.3 | 42.5 | 49.8 |
| FCOS + ATSS | 39.2 | 57.3 | 42.4 | 22.7 | 43.1 | 51.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: ATSS Table 2

| k 𝑘 k | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP ( % percent \% ) | 38.0 | 38.8 | 39.1 | 39.3 | 39.1 | 39.0 | 39.1 | 39.2 | 38.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: ATSS Table 3

| Scale | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 39.0 | 57.9 | 41.9 | 23.2 | 42.8 | 50.5 |
| 6 | 39.2 | 57.6 | 42.5 | 23.5 | 42.8 | 51.1 |
| 7 | 39.3 | 57.6 | 42.4 | 22.9 | 43.2 | 51.3 |
| 8 | 39.3 | 57.5 | 42.8 | 24.3 | 43.3 | 51.3 |
| 9 | 38.9 | 56.5 | 42.0 | 22.9 | 42.4 | 50.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: ATSS Table 4

| Aspect Ratio | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- |
| 1:4 | 39.1 | 57.2 | 42.3 | 23.1 | 43.1 | 51.4 |
| 1:2 | 39.0 | 56.9 | 42.5 | 23.3 | 43.5 | 50.6 |
| 1:1 | 39.3 | 57.5 | 42.8 | 24.3 | 43.3 | 51.3 |
| 2:1 | 39.3 | 57.4 | 42.3 | 22.8 | 43.4 | 51.0 |
| 4:1 | 39.1 | 56.9 | 42.6 | 22.9 | 42.9 | 50.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: ATSS Table 5

| Method | #sc | #ar | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RetinaNet (#A=9) | 3 | 3 | 36.3 | 55.2 | 38.8 | 19.8 | 39.8 | 48.8 |
| +Imprs. | 3 | 3 | 38.4 | 56.2 | 41.6 | 22.2 | 42.4 | 50.1 |
| +Imprs.+ATSS | 3 | 3 | 39.2 | 57.6 | 42.7 | 23.8 | 42.8 | 50.9 |
| +Imprs.+ATSS | 3 | 1 | 39.3 | 57.7 | 42.6 | 23.8 | 43.5 | 51.2 |
| +Imprs.+ATSS | 1 | 3 | 39.2 | 57.1 | 42.5 | 23.2 | 43.1 | 50.3 |
| +Imprs.+ATSS | 1 | 1 | 39.3 | 57.5 | 42.8 | 24.3 | 43.3 | 51.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: ATSS Table 6

| Method | Data | Backbone | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor-based two-stage: |  |  |  |  |  |  |  |  |
| MLKP [ 58 ] | trainval35 | ResNet-101 | 28.6 | 52.4 | 31.6 | 10.8 | 33.4 | 45.1 |
| R-FCN [ 9 ] | trainval | ResNet-101 | 29.9 | 51.9 | - | 10.8 | 32.8 | 45.0 |
| CoupleNet [ 74 ] | trainval | ResNet-101 | 34.4 | 54.8 | 37.2 | 13.4 | 38.1 | 50.8 |
| TDM [ 53 ] | trainval | Inception-ResNet-v2-TDM | 36.8 | 57.7 | 39.2 | 16.2 | 39.8 | 52.1 |
| Hu et al. [ 18 ] | trainval35k | ResNet-101 | 39.0 | 58.6 | 42.9 | - | - | - |
| DeepRegionlets [ 64 ] | trainval35k | ResNet-101 | 39.3 | 59.8 | - | 21.7 | 43.7 | 50.9 |
| FitnessNMS [ 57 ] | trainval | DeNet-101 | 39.5 | 58.0 | 42.6 | 18.9 | 43.5 | 54.1 |
| Gu et al. [ 15 ] | trainval35k | ResNet-101 | 39.9 | 63.1 | 43.1 | 22.2 | 43.4 | 51.6 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

Object detection has been dominated by anchor-based detectors for several years. Recently, anchor-free detectors have become popular due to the proposal of FPN and Focal Loss. In this paper, we first point out that the essential difference between anchor-based and anchor-free detection is actually how to define positive and negative training samples, which leads to the performance gap between them. If they adopt the same definition of positive and negative samples during training, there is no ob

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

固定 center sampling 可能制造不干净正负样本。ATSS 是下一轮最值得做的 assignment 消融之一。

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

- [[FCOS]]
- [[GFL]]
- [[TOOD]]
- [[YOLOX]]
