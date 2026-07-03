---
title: "You Only Look Once: Unified, Real-Time Object Detection"
method_name: "YOLOv1"
authors: [Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi]
year: 2015
venue: CVPR
tags: [object-detection, real-time-detection, one-stage-detection, yolo]
arxiv: https://arxiv.org/abs/1506.02640
created: 2026-06-11
image_source: online
---

# 论文笔记：You Only Look Once: Unified, Real-Time Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[YOLOv1]] |
| 作者 | [Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi] |
| 年份 | 2015 |
| 会议/来源 | CVPR |
| 链接 | [arXiv](https://arxiv.org/abs/1506.02640) / [PDF](https://arxiv.org/pdf/1506.02640.pdf) |
| 相关工作 | [[SSD]]、[[RetinaNet]]、[[YOLOX]]、[[YOLO-World]] |

---

## 一句话总结

> YOLOv1 将目标检测统一为单次前向回归任务，开创了实时一阶段检测器路线。

---

## 核心贡献

1. **范式贡献**：YOLOv1 将目标检测统一为单次前向回归任务，开创了实时一阶段检测器路线。
2. **方法贡献**：把检测视为整图一次回归，网格直接预测 bounding boxes、confidence 和 class probabilities。
3. **工程/研究影响**：该工作成为后续 [[SSD]]、[[RetinaNet]]、[[YOLOX]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

把检测视为整图一次回归，网格直接预测 bounding boxes、confidence 和 class probabilities。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[SSD]]、[[RetinaNet]]。
- 后续影响：[[YOLOX]]、[[YOLO-World]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\text{score}=P(c\mid object)\cdot P(object)\cdot IoU
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1506.02640](https://ar5iv.labs.arxiv.org/html/1506.02640)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 6 张、Table 4 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1 : The YOLO Detection System. Processing images with YOLO is simple and straightforward. Our system (1) resizes the input image to 448 × 448 448 448 448\times 448 , (2) runs a single convolutional network on the image, and (3) thresholds the resulting detections by the model’s confidence.

![Figure 1](https://ar5iv.labs.arxiv.org/html/1506.02640/assets/x1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 2: Figure 2 : The Model. Our system models detection as a regression problem. It divides the image into an S × S 𝑆 𝑆 S\times S grid and for each grid cell predicts B 𝐵 B bounding boxes, confidence for those boxes, and C 𝐶 C class probabilities. These predictions are encoded as an S × S × ( B ∗ 5 + C ) 𝑆 𝑆 𝐵 5 𝐶 S\times S\times(B*5+C) tensor.

![Figure 2](https://ar5iv.labs.arxiv.org/html/1506.02640/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 3: Figure 3 : The Architecture. Our detection network has 24 convolutional layers followed by 2 fully connected layers. Alternating 1 × 1 1 1 1\times 1 convolutional layers reduce the features space from preceding layers. We pretrain the convolutional layers on the ImageNet classification task at half the resolution ( 224 × 224 224 224 224\times 224 input image) and then double the resolution for detection.

![Figure 3](https://ar5iv.labs.arxiv.org/html/1506.02640/assets/x3.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 4: Figure 4 : Error Analysis: Fast R-CNN vs. YOLO These charts show the percentage of localization and background errors in the top N detections for various categories (N = # objects in that category).

![Figure 4](https://ar5iv.labs.arxiv.org/html/1506.02640/assets/x4.png)

**图表解读**：这是消融/分析图，重点看每个模块独立贡献，避免把数据增强、backbone 或训练轮数带来的收益误认为核心方法收益。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 5: (a) Picasso Dataset precision-recall curves.

![Figure 5](https://ar5iv.labs.arxiv.org/html/1506.02640/assets/x5.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 6: Figure 6 : Qualitative Results. YOLO running on sample artwork and natural images from the internet. It is mostly accurate although it does think one person is an airplane.

![Figure 6](https://ar5iv.labs.arxiv.org/html/1506.02640/assets/art.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### 关键表格

#### Table 1: YOLOv1 Table 1

| Real-Time Detectors | Train | mAP | FPS |
| --- | --- | --- | --- |
| 100Hz DPM [ 31 ] | 2007 | 16.0 | 100 |
| 30Hz DPM [ 31 ] | 2007 | 26.1 | 30 |
| Fast YOLO | 2007+2012 | 52.7 | 155 |
| YOLO | 2007+2012 | 63.4 | 45 |
| Less Than Real-Time |  |  |  |
| Fastest DPM [ 38 ] | 2007 | 30.4 | 15 |
| R-CNN Minus R [ 20 ] | 2007 | 53.5 | 6 |
| Fast R-CNN [ 14 ] | 2007+2012 | 70.0 | 0.5 |
| Faster R-CNN VGG-16 [ 28 ] | 2007+2012 | 73.2 | 7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: YOLOv1 Table 2

|  | mAP | Combined | Gain |
| --- | --- | --- | --- |
| Fast R-CNN | 71.8 | - | - |
| Fast R-CNN (2007 data) | 66.9 | 72.4 | .6 |
| Fast R-CNN (VGG-M) | 59.2 | 72.4 | .6 |
| Fast R-CNN (CaffeNet) | 57.1 | 72.1 | .3 |
| YOLO | 63.4 | 75.0 | 3.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: YOLOv1 Table 3

| VOC 2012 test | mAP | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MR_CNN_MORE_DATA [ 11 ] | 73.9 | 85.5 | 82.9 | 76.6 | 57.8 | 62.7 | 79.4 | 77.2 | 86.6 | 55.0 | 79.1 | 62.2 | 87.0 | 83.4 | 84.7 | 78.9 | 45.3 | 73.4 | 65.8 | 80.3 | 74.0 |
| HyperNet_VGG | 71.4 | 84.2 | 78.5 | 73.6 | 55.6 | 53.7 | 78.7 | 79.8 | 87.7 | 49.6 | 74.9 | 52.1 | 86.0 | 81.7 | 83.3 | 81.8 | 48.6 | 73.5 | 59.4 | 79.9 | 65.7 |
| HyperNet_SP | 71.3 | 84.1 | 78.3 | 73.3 | 55.5 | 53.6 | 78.6 | 79.6 | 87.5 | 49.5 | 74.9 | 52.1 | 85.6 | 81.6 | 83.2 | 81.6 | 48.4 | 73.2 | 59.3 | 79.7 | 65.6 |
| Fast R-CNN + YOLO | 70.7 | 83.4 | 78.5 | 73.5 | 55.8 | 43.4 | 79.1 | 73.1 | 89.4 | 49.4 | 75.5 | 57.0 | 87.5 | 80.9 | 81.0 | 74.7 | 41.8 | 71.5 | 68.5 | 82.1 | 67.2 |
| MR_CNN_S_CNN [ 11 ] | 70.7 | 85.0 | 79.6 | 71.5 | 55.3 | 57.7 | 76.0 | 73.9 | 84.6 | 50.5 | 74.3 | 61.7 | 85.5 | 79.9 | 81.7 | 76.4 | 41.0 | 69.0 | 61.2 | 77.7 | 72.1 |
| Faster R-CNN [ 28 ] | 70.4 | 84.9 | 79.8 | 74.3 | 53.9 | 49.8 | 77.5 | 75.9 | 88.5 | 45.6 | 77.1 | 55.3 | 86.9 | 81.7 | 80.9 | 79.6 | 40.1 | 72.6 | 60.9 | 81.2 | 61.5 |
| DEEP_ENS_COCO | 70.1 | 84.0 | 79.4 | 71.6 | 51.9 | 51.1 | 74.1 | 72.1 | 88.6 | 48.3 | 73.4 | 57.8 | 86.1 | 80.0 | 80.7 | 70.4 | 46.6 | 69.6 | 68.8 | 75.9 | 71.4 |
| NoC [ 29 ] | 68.8 | 82.8 | 79.0 | 71.6 | 52.3 | 53.7 | 74.1 | 69.0 | 84.9 | 46.9 | 74.3 | 53.1 | 85.0 | 81.3 | 79.5 | 72.2 | 38.9 | 72.4 | 59.5 | 76.7 | 68.1 |
| Fast R-CNN [ 14 ] | 68.4 | 82.3 | 78.4 | 70.8 | 52.3 | 38.7 | 77.8 | 71.6 | 89.3 | 44.2 | 73.0 | 55.0 | 87.5 | 80.5 | 80.8 | 72.0 | 35.1 | 68.3 | 65.7 | 80.4 | 64.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: YOLOv1 Table 4

|  | VOC 2007 | Picasso | People-Art |  |
| --- | --- | --- | --- | --- |
|  | AP | AP | Best F 1 subscript 𝐹 1 F_{1} | AP |
| YOLO | 59.2 | 53.3 | 0.590 | 45 |
| R-CNN | 54.2 | 10.4 | 0.226 | 26 |
| DPM | 43.2 | 37.8 | 0.458 | 32 |
| Poselets [ 2 ] | 36.5 | 17.8 | 0.271 |  |
| D&T [ 4 ] | - | 1.9 | 0.051 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

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

- [[SSD]]
- [[RetinaNet]]
- [[YOLOX]]
- [[YOLO-World]]
