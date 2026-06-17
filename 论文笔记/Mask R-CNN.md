---
title: "Mask R-CNN"
method_name: "Mask R-CNN"
authors: [Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick]
year: 2017
venue: ICCV
tags: [object-detection, instance-segmentation, roi-align, two-stage-detection]
arxiv: https://arxiv.org/abs/1703.06870
created: 2026-06-11
image_source: online
---

# 论文笔记：Mask R-CNN

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[Mask R-CNN]] |
| 作者 | [Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick] |
| 年份 | 2017 |
| 会议/来源 | ICCV |
| 链接 | [arXiv](https://arxiv.org/abs/1703.06870) / [PDF](https://arxiv.org/pdf/1703.06870.pdf) |
| 相关工作 | [[Faster R-CNN]]、[[FPN]]、[[Cascade R-CNN]]、[[Florence-2]] |

---

## 一句话总结

> Mask R-CNN 在 Faster R-CNN 上加入 mask branch，并用 ROIAlign 修正特征对齐问题，成为实例分割经典框架。

---

## 核心贡献

1. **范式贡献**：Mask R-CNN 在 Faster R-CNN 上加入 mask branch，并用 ROIAlign 修正特征对齐问题，成为实例分割经典框架。
2. **方法贡献**：在 Faster R-CNN 上添加 mask branch，并用 ROIAlign 解决 ROI Pooling 的量化错位，使检测、实例分割和关键点估计统一。
3. **工程/研究影响**：该工作成为后续 [[Faster R-CNN]]、[[FPN]]、[[Cascade R-CNN]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

在 Faster R-CNN 上添加 mask branch，并用 ROIAlign 解决 ROI Pooling 的量化错位，使检测、实例分割和关键点估计统一。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[Faster R-CNN]]、[[FPN]]。
- 后续影响：[[Cascade R-CNN]]、[[Florence-2]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
L=L_{cls}+L_{box}+L_{mask}
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1703.06870](https://ar5iv.labs.arxiv.org/html/1703.06870)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 7 张、Table 12 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: The Mask R-CNN framework for instance segmentation.

![Figure 1](https://ar5iv.labs.arxiv.org/html/1703.06870/assets/x1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 2: Figure 2: Mask R-CNN results on the COCO test set. These results are based on ResNet-101 [ 19 ] , achieving a mask AP of 35.7 and running at 5 fps. Masks are shown in color, and bounding box, category, and confidences are also shown.

![Figure 2](https://ar5iv.labs.arxiv.org/html/1703.06870/assets/x2.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 3: Figure 3: RoIAlign: The dashed grid represents a feature map, the solid lines an RoI (with 2 × \times 2 bins in this example), and the dots the 4 sampling points in each bin. RoIAlign computes the value of each sampling point by bilinear interpolation from the nearby grid points on the feature map. No quantization is performed on any coordinates involved in the RoI, its bins, or the sampling points.

![Figure 3](https://ar5iv.labs.arxiv.org/html/1703.06870/assets/x3.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 4: Figure 5: More results of Mask R-CNN on COCO test images, using ResNet-101-FPN and running at 5 fps, with 35.7 mask AP (Table 1 ).

![Figure 4](https://ar5iv.labs.arxiv.org/html/1703.06870/assets/x4.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 5: Figure 6: FCIS+++ [ 26 ] (top) vs . Mask R-CNN (bottom, ResNet-101-FPN). FCIS exhibits systematic artifacts on overlapping objects.

![Figure 5](https://ar5iv.labs.arxiv.org/html/1703.06870/assets/x5.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 6: Figure 7: Keypoint detection results on COCO test using Mask R-CNN (ResNet-50-FPN), with person segmentation masks predicted from the same model. This model has a keypoint AP of 63.1 and runs at 5 fps.

![Figure 6](https://ar5iv.labs.arxiv.org/html/1703.06870/assets/x6.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 7: Figure 8: Mask R-CNN results on Cityscapes test (32.0 AP). The bottom-right image shows a failure prediction.

![Figure 7](https://ar5iv.labs.arxiv.org/html/1703.06870/assets/x7.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### 关键表格

#### Table 1: Mask R-CNN Table 1

|  | backbone | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNC [ 10 ] | ResNet-101-C4 | 24.6 | 44.3 | 24.8 | 4.7 | 25.9 | 43.6 |
| FCIS [ 26 ] +OHEM | ResNet-101-C5-dilated | 29.2 | 49.5 | - | 7.1 | 31.3 | 50.0 |
| FCIS+++ [ 26 ] +OHEM | ResNet-101-C5-dilated | 33.6 | 54.5 | - | - | - | - |
| Mask R-CNN | ResNet-101-C4 | 33.1 | 54.9 | 34.8 | 12.1 | 35.6 | 51.1 |
| Mask R-CNN | ResNet-101-FPN | 35.7 | 58.0 | 37.8 | 15.5 | 38.1 | 52.4 |
| Mask R-CNN | ResNeXt-101-FPN | 37.1 | 60.0 | 39.4 | 16.9 | 39.9 | 53.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: Mask R-CNN Table 2

| net-depth-features | AP | AP 50 | AP 75 |
| --- | --- | --- | --- |
| ResNet-50-C4 | 30.3 | 51.2 | 31.5 |
| ResNet-101-C4 | 32.7 | 54.2 | 34.3 |
| ResNet-50-FPN | 33.6 | 55.2 | 35.3 |
| ResNet-101-FPN | 35.4 | 57.3 | 37.5 |
| ResNeXt-101-FPN | 36.7 | 59.5 | 38.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: Mask R-CNN Table 3

|  | AP | AP 50 | AP 75 |
| --- | --- | --- | --- |
| softmax | 24.8 | 44.1 | 25.1 |
| sigmoid | 30.3 | 51.2 | 31.5 |
|  | +5.5 | +7.1 | +6.4 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: Mask R-CNN Table 4

|  | align? | bilinear? | agg. | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- | --- | --- |
| RoIPool [ 12 ] |  |  | max | 26.9 | 48.8 | 26.4 |
| RoIWarp [ 10 ] |  | ✓ | max | 27.2 | 49.2 | 27.1 |
|  | ✓ | ave | 27.1 | 48.9 | 27.1 |  |
| RoIAlign | ✓ | ✓ | max | 30.2 | 51.0 | 31.8 |
| ✓ | ✓ | ave | 30.3 | 51.2 | 31.5 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: Mask R-CNN Table 5

|  | AP | AP 50 | AP 75 | AP bb bb {}^{\text{bb}} | AP 50 bb subscript superscript absent bb 50 {}^{\text{bb}}_{50} | AP 75 bb subscript superscript absent bb 75 {}^{\text{bb}}_{75} |
| --- | --- | --- | --- | --- | --- | --- |
| RoIPool | 23.6 | 46.5 | 21.6 | 28.2 | 52.7 | 26.9 |
| RoIAlign | 30.9 | 51.8 | 32.1 | 34.0 | 55.3 | 36.4 |
|  | +7.3 | + 5.3 | +10.5 | +5.8 | +2.6 | +9.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: Mask R-CNN Table 6

|  | mask branch | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- |
| MLP | fc: 1024 → → \rightarrow 1024 → → \rightarrow 80 ⋅ 28 2 ⋅ 80 superscript 28 2 80 | 31.5 | 53.7 | 32.8 |
| MLP | fc: 1024 → → \rightarrow 1024 → → \rightarrow 1024 → → \rightarrow 80 ⋅ 28 2 ⋅ 8 | 31.5 | 54.0 | 32.6 |
| FCN | conv: 256 → → \rightarrow 256 → → \rightarrow 256 → → \rightarrow 256 → → \right | 33.6 | 55.2 | 35.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: Mask R-CNN Table 7

|  | backbone | AP bb bb {}^{\text{bb}} | AP 50 bb subscript superscript absent bb 50 {}^{\text{bb}}_{50} | AP 75 bb subscript superscript absent bb 75 {}^{\text{bb}}_{75} | AP S bb subscript superscript absent bb 𝑆 {}^{\text{bb}}_{S} | AP M bb subscript superscript absent bb 𝑀 {}^{\text{bb}}_{M} | AP L bb subscript superscript absent bb 𝐿 {}^{\text{bb}}_{L} |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN+++ [ 19 ] | ResNet-101-C4 | 34.9 | 55.7 | 37.4 | 15.6 | 38.7 | 50.9 |
| Faster R-CNN w FPN [ 27 ] | ResNet-101-FPN | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2 |
| Faster R-CNN by G-RMI [ 21 ] | Inception-ResNet-v2 [ 41 ] | 34.7 | 55.5 | 36.7 | 13.5 | 38.1 | 52.0 |
| Faster R-CNN w TDM [ 39 ] | Inception-ResNet-v2-TDM | 36.8 | 57.7 | 39.2 | 16.2 | 39.8 | 52.1 |
| Faster R-CNN, RoIAlign | ResNet-101-FPN | 37.3 | 59.6 | 40.3 | 19.8 | 40.2 | 48.8 |
| Mask R-CNN | ResNet-101-FPN | 38.2 | 60.3 | 41.7 | 20.1 | 41.1 | 50.2 |
| Mask R-CNN | ResNeXt-101-FPN | 39.8 | 62.3 | 43.4 | 22.1 | 43.2 | 51.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: Mask R-CNN Table 8

|  | AP kp kp {}^{\text{kp}} | AP 50 kp subscript superscript absent kp 50 {}^{\text{kp}}_{50} | AP 75 kp subscript superscript absent kp 75 {}^{\text{kp}}_{75} | AP M kp subscript superscript absent kp 𝑀 {}^{\text{kp}}_{M} | AP L kp subscript superscript absent kp 𝐿 {}^{\text{kp}}_{L} |
| --- | --- | --- | --- | --- | --- |
| CMU-Pose+++ [ 6 ] | 61.8 | 84.9 | 67.5 | 57.1 | 68.2 |
| G-RMI [ 32 ] † | 62.4 | 84.0 | 68.5 | 59.1 | 68.1 |
| Mask R-CNN , keypoint-only | 62.7 | 87.0 | 68.4 | 57.4 | 71.1 |
| Mask R-CNN , keypoint & mask | 63.1 | 87.3 | 68.7 | 57.8 | 71.4 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 9: Mask R-CNN Table 9

|  | AP person bb subscript superscript absent bb person {}^{\text{bb}}_{\text{\emph{ | AP person mask subscript superscript absent mask person {}^{\text{mask}}_{\text{ | AP kp kp {}^{\text{kp}} |
| --- | --- | --- | --- |
| Faster R-CNN | 52.5 | - | - |
| Mask R-CNN, mask-only | 53.6 | 45.8 | - |
| Mask R-CNN, keypoint-only | 50.7 | - | 64.2 |
| Mask R-CNN, keypoint & mask | 52.0 | 45.1 | 64.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 10: Mask R-CNN Table 10

|  | AP kp kp {}^{\text{kp}} | AP 50 kp subscript superscript absent kp 50 {}^{\text{kp}}_{50} | AP 75 kp subscript superscript absent kp 75 {}^{\text{kp}}_{75} | AP M kp subscript superscript absent kp 𝑀 {}^{\text{kp}}_{M} | AP L kp subscript superscript absent kp 𝐿 {}^{\text{kp}}_{L} |
| --- | --- | --- | --- | --- | --- |
| RoIPool | 59.8 | 86.2 | 66.7 | 55.1 | 67.4 |
| RoIAlign | 64.2 | 86.6 | 69.7 | 58.7 | 73.0 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 11: Mask R-CNN Table 11

|  | training data | AP [ val ] | AP | AP 50 | person | rider | car | truck | bus | train | mcycle | bicycle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| InstanceCut [ 23 ] | fine + coarse | 15.8 | 13.0 | 27.9 | 10.0 | 8.0 | 23.7 | 14.0 | 19.5 | 15.2 | 9.3 | 4.7 |
| DWT [ 4 ] | fine | 19.8 | 15.6 | 30.0 | 15.1 | 11.7 | 32.9 | 17.1 | 20.4 | 15.0 | 7.9 | 4.9 |
| SAIS [ 17 ] | fine | - | 17.4 | 36.7 | 14.6 | 12.9 | 35.7 | 16.0 | 23.2 | 19.0 | 10.3 | 7.8 |
| DIN [ 3 ] | fine + coarse | - | 20.0 | 38.8 | 16.5 | 16.7 | 25.7 | 20.6 | 30.0 | 23.4 | 17.1 | 10.1 |
| SGN [ 29 ] | fine + coarse | 29.2 | 25.0 | 44.9 | 21.8 | 20.1 | 39.4 | 24.8 | 33.2 | 30.8 | 17.7 | 12.4 |
| Mask R-CNN | fine | 31.5 | 26.2 | 49.9 | 30.5 | 23.7 | 46.9 | 22.8 | 32.2 | 18.6 | 19.1 | 16.0 |
| Mask R-CNN | fine + COCO | 36.4 | 32.0 | 58.1 | 34.8 | 27.0 | 49.1 | 30.1 | 40.9 | 30.9 | 24.1 | 18.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 12: Mask R-CNN Table 12

| description | backbone | AP | AP 50 | AP 75 | AP bb bb {}^{\text{bb}} | AP 50 bb subscript superscript absent bb 50 {}^{\text{bb}}_{50} | AP 75 bb subscript superscript absent bb 75 {}^{\text{bb}}_{75} |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original baseline | X-101-FPN | 36.7 | 59.5 | 38.9 | 39.6 | 61.5 | 43.2 |
| + + updated baseline | X-101-FPN | 37.0 | 59.7 | 39.0 | 40.5 | 63.0 | 43.7 |
| + + e2e training | X-101-FPN | 37.6 | 60.4 | 39.9 | 41.7 | 64.1 | 45.2 |
| + + ImageNet-5k | X-101-FPN | 38.6 | 61.7 | 40.9 | 42.7 | 65.1 | 46.6 |
| + + train-time augm. | X-101-FPN | 39.2 | 62.5 | 41.6 | 43.5 | 65.9 | 47.2 |
| + + deeper | X-152-FPN | 39.7 | 63.2 | 42.2 | 44.1 | 66.4 | 48.4 |
| + + Non-local [ 43 ] | X-152-FPN-NL | 40.3 | 64.4 | 42.8 | 45.0 | 67.8 | 48.9 |
| + + test-time augm. | X-152-FPN-NL | 41.8 | 66.0 | 44.8 | 47.3 | 69.3 | 51.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps.

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

- [[Faster R-CNN]]
- [[FPN]]
- [[Cascade R-CNN]]
- [[Florence-2]]
