---
title: "Focal Loss for Dense Object Detection"
method_name: "RetinaNet"
authors: [Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar]
year: 2017
venue: ICCV
tags: [object-detection, focal-loss, one-stage-detection, dense-detection]
arxiv: https://arxiv.org/abs/1708.02002
created: 2026-06-11
image_source: online
---

# 论文笔记：Focal Loss for Dense Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[RetinaNet]] |
| 作者 | [Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar] |
| 年份 | 2017 |
| 会议/来源 | ICCV |
| 链接 | [arXiv](https://arxiv.org/abs/1708.02002) / [PDF](https://arxiv.org/pdf/1708.02002.pdf) |
| 相关工作 | [[FPN]]、[[FCOS]]、[[GFL]]、[[YOLOX]] |

---

## 一句话总结

> RetinaNet 用 Focal Loss 解决一阶段密集检测中的前景/背景极度不均衡问题，让一阶段检测达到两阶段精度。

---

## 核心贡献

1. **范式贡献**：RetinaNet 用 Focal Loss 解决一阶段密集检测中的前景/背景极度不均衡问题，让一阶段检测达到两阶段精度。
2. **方法贡献**：构建 FPN one-stage detector，并提出 Focal Loss 解决 dense negative overwhelming。
3. **工程/研究影响**：该工作成为后续 [[FPN]]、[[FCOS]]、[[GFL]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

构建 FPN one-stage detector，并提出 Focal Loss 解决 dense negative overwhelming。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[FPN]]、[[FCOS]]。
- 后续影响：[[GFL]]、[[YOLOX]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
FL(p_t)=-(1-p_t)^\gamma\log(p_t)
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1708.02002](https://ar5iv.labs.arxiv.org/html/1708.02002)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 7 张、Table 8 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 2: Speed (ms) versus accuracy (AP) on COCO test-dev . Enabled by the focal loss, our simple one-stage RetinaNet detector outperforms all previous one-stage and two-stage detectors, including the best reported Faster R-CNN [ 28 ] system from [ 20 ] . We show variants of RetinaNet with ResNet-50-FPN (blue circles) and ResNet-101-FPN (orange diamonds) at five scales (400-800 pixels). Ignoring the low-accuracy regime (AP < < 25), RetinaNet forms an upper envelope of all current detectors, and an improved variant (not shown) achieves 40.8 AP. Details are given in § 5 .

![Figure 1](https://ar5iv.labs.arxiv.org/html/1708.02002/assets/x1.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 2: Figure 3: The one-stage RetinaNet network architecture uses a Feature Pyramid Network (FPN) [ 20 ] backbone on top of a feedforward ResNet architecture [ 16 ] (a) to generate a rich, multi-scale convolutional feature pyramid (b). To this backbone RetinaNet attaches two subnetworks, one for classifying anchor boxes (c) and one for regressing from anchor boxes to ground-truth object boxes (d). The network design is intentionally simple, which enables this work to focus on a novel focal loss function that eliminates the accuracy gap between our one-stage detector and state-of-the-art two-stage detectors like Faster R-CNN with FPN [ 20 ] while running at faster speeds.

![Figure 2](https://ar5iv.labs.arxiv.org/html/1708.02002/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 3: Figure 4: Cumulative distribution functions of the normalized loss for positive and negative samples for different values of γ 𝛾 \gamma for a converged model. The effect of changing γ 𝛾 \gamma on the distribution of the loss for positive examples is minor. For negatives, however, increasing γ 𝛾 \gamma heavily concentrates the loss on hard examples, focusing nearly all attention away from easy negatives.

![Figure 3](https://ar5iv.labs.arxiv.org/html/1708.02002/assets/x3.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 4: Figure 4: Cumulative distribution functions of the normalized loss for positive and negative samples for different values of γ 𝛾 \gamma for a converged model. The effect of changing γ 𝛾 \gamma on the distribution of the loss for positive examples is minor. For negatives, however, increasing γ 𝛾 \gamma heavily concentrates the loss on hard examples, focusing nearly all attention away from easy negatives.

![Figure 4](https://ar5iv.labs.arxiv.org/html/1708.02002/assets/x4.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 5: Figure 5: Focal loss variants compared to the cross entropy as a function of x t = y ​ x subscript 𝑥 t 𝑦 𝑥 x_{\textrm{t}}=yx . Both the original FL and alternate variant FL ∗ superscript FL \textrm{FL}^{*} reduce the relative loss for well-classified examples ( x t > 0 subscript 𝑥 t 0 x_{\textrm{t}}>0 ).

![Figure 5](https://ar5iv.labs.arxiv.org/html/1708.02002/assets/x5.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 6: Figure 6: Derivates of the loss functions from Figure 5 w.r.t. x 𝑥 x .

![Figure 6](https://ar5iv.labs.arxiv.org/html/1708.02002/assets/x6.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 7: Figure 7: Effectiveness of FL ∗ superscript FL \textrm{FL}^{*} with various settings γ 𝛾 \gamma and β 𝛽 \beta . The plots are color coded such that effective settings are shown in blue.

![Figure 7](https://ar5iv.labs.arxiv.org/html/1708.02002/assets/x7.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### 关键表格

#### Table 1: RetinaNet Table 1

|  | AP | time |
| --- | --- | --- |
| [A] YOLOv2 † [ 27 ] | 21.6 | 25 |
| [B] SSD321 [ 22 ] | 28.0 | 61 |
| [C] DSSD321 [ 9 ] | 28.0 | 85 |
| [D] R-FCN ‡ [ 3 ] | 29.9 | 85 |
| [E] SSD513 [ 22 ] | 31.2 | 125 |
| [F] DSSD513 [ 9 ] | 33.2 | 156 |
| [G] FPN FRCN [ 20 ] | 36.2 | 172 |
| RetinaNet-50-500 | 32.5 | 73 |
| RetinaNet-101-500 | 34.4 | 90 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: RetinaNet Table 2

| α 𝛼 \alpha | AP | AP 50 | AP 75 |
| --- | --- | --- | --- |
| .10 | 0.0 | 0.0 | 0.0 |
| .25 | 10.8 | 16.0 | 11.7 |
| .50 | 30.2 | 46.7 | 32.8 |
| .75 | 31.1 | 49.4 | 33.0 |
| .90 | 30.8 | 49.7 | 32.3 |
| .99 | 28.7 | 47.4 | 29.9 |
| .999 | 25.1 | 41.7 | 26.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: RetinaNet Table 3

| γ 𝛾 \gamma | α 𝛼 \alpha | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- |
| 0 | .75 | 31.1 | 49.4 | 33.0 |
| 0.1 | .75 | 31.4 | 49.9 | 33.1 |
| 0.2 | .75 | 31.9 | 50.7 | 33.4 |
| 0.5 | .50 | 32.9 | 51.7 | 35.2 |
| 1.0 | .25 | 33.7 | 52.0 | 36.2 |
| 2.0 | .25 | 34.0 | 52.5 | 36.5 |
| 5.0 | .25 | 32.2 | 49.6 | 34.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: RetinaNet Table 4

| #sc | #ar | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 30.3 | 49.0 | 31.8 |
| 2 | 1 | 31.9 | 50.0 | 34.0 |
| 3 | 1 | 31.8 | 49.4 | 33.7 |
| 1 | 3 | 32.4 | 52.3 | 33.9 |
| 2 | 3 | 34.2 | 53.1 | 36.5 |
| 3 | 3 | 34.0 | 52.5 | 36.5 |
| 4 | 3 | 33.8 | 52.1 | 36.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: RetinaNet Table 5

| method | batch | nms | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- | --- |
| size | thr |  |  |  |  |
| OHEM | 128 | .7 | 31.1 | 47.2 | 33.2 |
| OHEM | 256 | .7 | 31.8 | 48.8 | 33.9 |
| OHEM | 512 | .7 | 30.6 | 47.0 | 32.6 |
| OHEM | 128 | .5 | 32.8 | 50.3 | 35.1 |
| OHEM | 256 | .5 | 31.0 | 47.4 | 33.0 |
| OHEM | 512 | .5 | 27.6 | 42.0 | 29.2 |
| OHEM 1:3 | 128 | .5 | 31.1 | 47.2 | 33.2 |
| OHEM 1:3 | 256 | .5 | 28.3 | 42.4 | 30.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: RetinaNet Table 6

| depth | scale | AP | AP 50 | AP 75 | AP S | AP M | AP L | time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 400 | 30.5 | 47.8 | 32.7 | 11.2 | 33.8 | 46.1 | 64 |
| 50 | 500 | 32.5 | 50.9 | 34.8 | 13.9 | 35.8 | 46.7 | 72 |
| 50 | 600 | 34.3 | 53.2 | 36.9 | 16.2 | 37.4 | 47.4 | 98 |
| 50 | 700 | 35.1 | 54.2 | 37.7 | 18.0 | 39.3 | 46.4 | 121 |
| 50 | 800 | 35.7 | 55.0 | 38.5 | 18.9 | 38.9 | 46.3 | 153 |
| 101 | 400 | 31.9 | 49.5 | 34.1 | 11.6 | 35.8 | 48.5 | 81 |
| 101 | 500 | 34.4 | 53.1 | 36.8 | 14.7 | 38.5 | 49.1 | 90 |
| 101 | 600 | 36.0 | 55.2 | 38.7 | 17.4 | 39.6 | 49.7 | 122 |
| 101 | 700 | 37.1 | 56.6 | 39.8 | 19.1 | 40.6 | 49.4 | 154 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: RetinaNet Table 7

|  | backbone | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Two-stage methods |  |  |  |  |  |  |  |
| Faster R-CNN+++ [ 16 ] | ResNet-101-C4 | 34.9 | 55.7 | 37.4 | 15.6 | 38.7 | 50.9 |
| Faster R-CNN w FPN [ 20 ] | ResNet-101-FPN | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2 |
| Faster R-CNN by G-RMI [ 17 ] | Inception-ResNet-v2 [ 34 ] | 34.7 | 55.5 | 36.7 | 13.5 | 38.1 | 52.0 |
| Faster R-CNN w TDM [ 32 ] | Inception-ResNet-v2-TDM | 36.8 | 57.7 | 39.2 | 16.2 | 39.8 | 52.1 |
| One-stage methods |  |  |  |  |  |  |  |
| YOLOv2 [ 27 ] | DarkNet-19 [ 27 ] | 21.6 | 44.0 | 19.2 | 5.0 | 22.4 | 35.5 |
| SSD513 [ 22 , 9 ] | ResNet-101-SSD | 31.2 | 50.4 | 33.3 | 10.2 | 34.5 | 49.8 |
| DSSD513 [ 9 ] | ResNet-101-DSSD | 33.2 | 53.3 | 35.2 | 13.0 | 35.4 | 51.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: RetinaNet Table 8

| loss | γ 𝛾 \gamma | β 𝛽 \beta | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- | --- |
| CE | – | – | 31.1 | 49.4 | 33.0 |
| FL | 2.0 | – | 34.0 | 52.5 | 36.5 |
| FL ∗ superscript FL \textrm{FL}^{*} | 2.0 | 1.0 | 33.8 | 52.7 | 36.3 |
| FL ∗ superscript FL \textrm{FL}^{*} | 4.0 | 0.0 | 33.9 | 51.8 | 36.4 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background 

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

Focal Loss 解释了背景不均衡，但项目实验证明 focal 会压低 TP 分数，因此应走 QFL + objectness + calibration，而不是回退到纯 focal。

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

- [[FPN]]
- [[FCOS]]
- [[GFL]]
- [[YOLOX]]
