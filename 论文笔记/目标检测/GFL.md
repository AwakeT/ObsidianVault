---
title: "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection"
method_name: "GFL"
authors: [Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang]
year: 2020
venue: NeurIPS
tags: [object-detection, quality-focal-loss, distribution-focal-loss, dense-detection]
arxiv: https://arxiv.org/abs/2006.04388
created: 2026-06-11
image_source: online
---

# 论文笔记：Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[GFL]] |
| 作者 | [Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang] |
| 年份 | 2020 |
| 会议/来源 | NeurIPS |
| 链接 | [arXiv](https://arxiv.org/abs/2006.04388) / [PDF](https://arxiv.org/pdf/2006.04388.pdf) |
| 相关工作 | [[RetinaNet]]、[[FCOS]]、[[ATSS]]、[[TOOD]]、[[D-FINE]] |

---

## 一句话总结

> GFL 将分类置信度和定位质量统一建模，并用分布式框回归提升定位精度，是现代 dense detector 质量估计的重要基础。

---

## 核心贡献

1. **范式贡献**：GFL 将分类置信度和定位质量统一建模，并用分布式框回归提升定位精度，是现代 dense detector 质量估计的重要基础。
2. **方法贡献**：把分类目标从硬标签改成 IoU 质量软标签，同时用 Distribution Focal Loss 建模边界距离分布。
3. **工程/研究影响**：该工作成为后续 [[RetinaNet]]、[[FCOS]]、[[ATSS]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

把分类目标从硬标签改成 IoU 质量软标签，同时用 Distribution Focal Loss 建模边界距离分布。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[RetinaNet]]、[[FCOS]]。
- 后续影响：[[ATSS]]、[[TOOD]]、[[D-FINE]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\hat{y}=\sum_i p_i y_i
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2006.04388](https://ar5iv.labs.arxiv.org/html/2006.04388)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 16 张、Table 8 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1 : Comparisons between existing separate representation and proposed joint representation of classification and localization quality estimation. (a): Current practices jiang2018acquisition ; tian2019fcos ; wu2020iou ; zhu2019iou ; zhang2019bridging for the separate usage of the quality branch (i.e., IoU or centerness score) during training and test. (b): Our joint representation of classification and localization quality enables high consistency between training and inference.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x1.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 2: Figure 2 : Unreliable IoU predictions of current dense detector with IoU-branch. (a): We demonstrate some background patches ( A and B ) with extremely high predicted quality scores (e.g., IoU score > 0.9), based on the optimized IoU-branch model in Fig. 1 (a). The scatter diagram in (b) denotes the randomly sampled instances with their predicted scores, where the blue points clearly illustrate the weak correlation between predicted classification scores and predicted IoU scores for separate representations. The part in red circle contains many possible negatives with large localization quality predictions, which may potentially rank in front of true positives and impair the performance. Instead, our joint representation ( green points) forces them to be equal and thus avoids such risks.

![Figure 2](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 3: Figure 3 : Due to occlusion, shadow, blur, etc., the boundaries of many objects are not clear enough, so that the ground-truth labels (white boxes) are sometimes not credible and Dirac delta distribution is limited to indicate such issues. Instead, the proposed learned representation of General distribution for bounding boxes can reflect the underlying information by its shape, where a flatten distribution denotes the unclear and ambiguous boundaries (see red circles) and a sharp one stands for the clear cases. The predicted boxes by our model are marked green .

![Figure 3](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x3.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 4: Figure 4 : The comparisons between conventional methods and our proposed GFL in the head of dense detectors. GFL includes QFL and DFL. QFL effectively learns a joint representation of classification score and localization quality estimation. DFL models the locations of bounding boxes as General distributions whilst forcing the networks to rapidly focus on learning the probabilities of values close to the target coordinates.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x4.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 5: Figure 5 : (a): The illustration of QFL under quality label y = 0.5 𝑦 0.5 y=0.5 . (b): Different flexible distributions can obtain the same integral target according to Eq. ( 4 ), thus we need to focus on learning probabilities of values around the target for more reasonable and confident predictions (e.g., (3)). (c): The histogram of bounding box regression targets of ATSS over all training samples on COCO trainval35k .

![Figure 5](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x5.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 6: Figure 6 : Illustrations of modified versions for separate/implicit and joint representation. The baseline without quality branch is also provided.

![Figure 6](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x6.png)

**图表解读**：重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 7: Figure 7 : Qualitative comparisons between Dirac delta (a), Gaussian (b) and our proposed General (c) distribution for bounding box regression on COCO minival , based on ATSS zhang2019bridging . White boxes denote the ground-truth labels, and the predicted ones are marked green.

![Figure 7](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x7.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 8: Figure 8 : Single-model single-scale speed (ms) vs. accuracy (AP) on COCO test-dev among state-of-the-art approaches. GFL achieves better speed-accuracy trade-off than many competitive counterparts.

![Figure 8](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x8.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 9: Figure 9: Illustrations of three distributions, from rigid (Dirac delta) to flexible (General). The proposed General distribution is more flexible as its shape can be arbitrary. In contrast, Dirac delta distribution roots at a fixed point and Gaussian distribution follows a relatively rigid, symmetric expression, e.g., 1 σ ​ 2 ​ π ​ e − ( x − μ ) 2 2 ​ σ 2 1 𝜎 2 𝜋 superscript 𝑒 superscript 𝑥 𝜇 2 2 superscript 𝜎 2 \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}} , which both have more limitations in modeling real data distribution.

![Figure 9](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x9.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 10: Figure 10: We demonstrate an example in 2D space by fixing the input feature vector and introduce a small disturbance (norm of 0.1) over it. The regression targets are 1.5, 2.5, 3.5 respectively. It is observed that Dirac delta distribution leads to more regression errors after the same disturbance, and the error increases with the growth of regression target. In contrast, our proposed General distribution remains stable and insensitive to the disturbance.

![Figure 10](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x10.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 11: Figure 11: We demonstrate possible cases of ground-truth/predicted bounding box along with the positive points. The matrix points denote the feature pyramid layer with stride = 8. Centerness label is easier to get very small values by its definition, whilst IoU label is more reliable as the supervisions from bounding boxes will always push it close to 1.0.

![Figure 11](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x11.png)

**图表解读**：重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 12: Figure 12: Label distributions over all positive training samples on COCO, based on pretrained GFL detector (ResNet-50 backbone).

![Figure 12](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x12.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 13: Figure 13: Examples with huge boundary ambiguities and uncertainties, where the learned General distributions tend to be flatten. In some cases, we even observe a distribution with two peaks. Interestingly, they do correspond to two different most likely boundaries in the image, e.g., the boundaries of the umbrella whether its heavily occluded handle is considered. Predictions are marked green in images, whilst ground-truth boxes are white.

![Figure 13](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x13.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 14: Figure 13: Examples with huge boundary ambiguities and uncertainties, where the learned General distributions tend to be flatten. In some cases, we even observe a distribution with two peaks. Interestingly, they do correspond to two different most likely boundaries in the image, e.g., the boundaries of the umbrella whether its heavily occluded handle is considered. Predictions are marked green in images, whilst ground-truth boxes are white.

![Figure 14](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x14.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 15: Figure 14: Examples with extremely clear boundaries. The learned General distributions are relatively sharp whilst producing very accurate box estimations. Predictions are marked green in images, whilst ground-truth boxes are white.

![Figure 15](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x15.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### Figure 16: Figure 14: Examples with extremely clear boundaries. The learned General distributions are relatively sharp whilst producing very accurate box estimations. Predictions are marked green in images, whilst ground-truth boxes are white.

![Figure 16](https://ar5iv.labs.arxiv.org/html/2006.04388/assets/x16.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 quality-aware classification 与 distributional box regression；对当前 QFL/DFL 头的置信度校准非常关键。

### 关键表格

#### Table 1: GFL Table 1

| Type | FCOS tian2019fcos | ATSS zhang2019bridging |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP | AP 50 | AP 75 | AP S | AP M | AP L | AP | AP 50 | AP 75 | AP S | AP M | AP L |  |
| w/o quality branch | 37.8 | 56.2 | 40.8 | 21.2 | 42.1 | 48.2 | 38.0 | 56.5 | 40.7 | 20.6 | 42.1 | 49.1 |
| centerness-branch tian2019fcos | 38.5 | 56.8 | 41.6 | 22.4 | 42.4 | 49.1 | 39.2 | 57.4 | 42.2 | 23.0 | 42.8 | 51.1 |
| IoU-branch wu2020iou ; jiang2018acquisition | 38.7 | 56.7 | 42.0 | 21.6 | 43.0 | 50.3 | 39.6 | 57.6 | 43.0 | 23.3 | 43.7 | 51.2 |
| centerness-guided wu2019iou | 37.9 | 56.7 | 40.7 | 21.2 | 42.1 | 49.4 | 38.2 | 56.2 | 41.0 | 21.5 | 41.9 | 49.7 |
| IoU-guided wu2019iou | 38.2 | 57.0 | 41.1 | 22.5 | 42.2 | 48.9 | 38.9 | 57.4 | 41.8 | 22.8 | 42.4 | 50.6 |
| joint w/ QFL (ours) | 39.0 | 57.8 | 41.9 | 22.0 | 43.1 | 51.0 | 39.9 | 58.5 | 43.0 | 22.4 | 43.9 | 52.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: GFL Table 2

| Method | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- |
| FoveaBox kong2019foveabox | 36.4 | 55.8 | 38.8 | 19.4 | 40.4 | 47.7 |
| FoveaBox kong2019foveabox + joint w/ QFL | 37.0 | 55.7 | 39.6 | 20.2 | 41.2 | 48.8 |
| RetinaNet lin2017focal | 35.6 | 55.5 | 38.1 | 20.1 | 39.4 | 46.8 |
| RetinaNet lin2017focal + joint w/ QFL | 36.4 | 56.3 | 39.1 | 20.4 | 40.0 | 48.7 |
| SSD512 liu2016ssd | 29.4 | 49.1 | 30.6 | 11.4 | 34.1 | 44.9 |
| SSD512 liu2016ssd + joint w/ QFL | 30.2 | 50.3 | 31.7 | 13.3 | 34.4 | 45.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: GFL Table 3

| β 𝛽 \beta (QFL) | AP | AP 50 | AP 75 |
| --- | --- | --- | --- |
| 0 | 37.6 | 55.4 | 40.3 |
| 1 | 39.0 | 58.1 | 41.7 |
| 2 | 39.9 | 58.5 | 43.0 |
| 2.5 | 39.7 | 58.1 | 42.7 |
| 4 | 38.2 | 55.4 | 41.6 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: GFL Table 4

| Prior Distribution | FCOS tian2019fcos | ATSS zhang2019bridging |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP | AP 50 | AP 75 | AP S | AP M | AP L | AP | AP 50 | AP 75 | AP S | AP M | AP L |  |
| Dirac delta tian2019fcos ; zhang2019bridging | 38.5 | 56.8 | 41.6 | 22.4 | 42.4 | 49.1 | 39.2 | 57.4 | 42.2 | 23.0 | 42.8 | 51.1 |
| Gaussian he2019bounding ; choi2019gaussian | 38.6 | 56.5 | 41.6 | 21.7 | 42.5 | 50.0 | 39.3 | 57.0 | 42.4 | 23.6 | 42.9 | 51.0 |
| General (ours) | 38.8 | 56.6 | 42.0 | 22.5 | 42.9 | 49.8 | 39.3 | 57.1 | 42.5 | 23.5 | 43.0 | 51.2 |
| General w/ DFL (ours) | 39.0 | 57.0 | 42.3 | 22.6 | 43.0 | 50.6 | 39.5 | 57.3 | 42.8 | 23.6 | 43.2 | 51.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: GFL Table 5

| n | Δ Δ \varDelta | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 1 | 40.1 | 58.4 | 43.1 | 23.1 | 43.8 | 52.5 |
| 14 | 40.2 | 58.3 | 43.6 | 23.3 | 44.2 | 52.2 |  |
| 16 | 40.2 | 58.6 | 43.4 | 23.0 | 44.3 | 53.0 |  |
| 18 | 40.1 | 58.1 | 43.1 | 22.6 | 43.9 | 52.6 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: GFL Table 6

| n | Δ Δ \varDelta | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.5 | 40.2 | 58.4 | 43.0 | 22.3 | 43.8 | 53.1 |
| 1 | 40.2 | 58.6 | 43.4 | 23.0 | 44.3 | 53.0 |  |
| 2 | 39.9 | 58.3 | 42.9 | 22.5 | 43.8 | 51.8 |  |
| 4 | 39.8 | 58.5 | 42.8 | 22.8 | 43.4 | 52.3 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: GFL Table 7

| QFL | DFL | FPS | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- | --- |
|  |  | 19.4 | 39.2 | 57.4 | 42.2 |
| ✓ ✓ \checkmark |  | 19.4 | 39.9 | 58.5 | 43.0 |
|  | ✓ ✓ \checkmark | 19.4 | 39.5 | 57.3 | 42.8 |
| ✓ ✓ \checkmark | ✓ ✓ \checkmark | 19.4 | 40.2 | 58.6 | 43.4 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: GFL Table 8

| Method | Backbone | Epoch | MS train train {}_{\text{train}} | FPS | AP | AP 50 | AP 75 | AP S | AP M | AP L | Reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi-stage: |  |  |  |  |  |  |  |  |  |  |  |
| Faster R-CNN w/ FPN lin2017feature | R-101 | 24 |  | 14.2 | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2 | CVPR17 |
| Cascade R-CNN cai2018cascade | R-101 | 18 |  | 11.9 | 42.8 | 62.1 | 46.3 | 23.7 | 45.5 | 55.2 | CVPR18 |
| Grid R-CNN lu2019grid | R-101 | 20 |  | 11.4 | 41.5 | 60.9 | 44.5 | 23.3 | 44.9 | 53.1 | CVPR19 |
| Libra R-CNN pang2019libra | R-101 | 24 |  | 13.6 | 41.1 | 62.1 | 44.7 | 23.4 | 43.7 | 52.5 | CVPR19 |
| Libra R-CNN pang2019libra | X-101-64x4d | 12 |  | 8.5 | 43.0 | 64.0 | 47.0 | 25.3 | 45.6 | 54.6 | CVPR19 |
| RepPoints yang2019reppoints | R-101 | 24 |  | 13.3 | 41.0 | 62.9 | 44.3 | 23.6 | 44.1 | 51.7 | ICCV19 |
| RepPoints yang2019reppoints | R-101-DCN | 24 | ✓ ✓ \checkmark | 11.8 | 45.0 | 66.1 | 49.0 | 26.6 | 48.6 | 57.5 | ICCV19 |
| TridentNet li2019scale | R-101 | 24 | ✓ ✓ \checkmark | 2.7 ∗ | 42.7 | 63.6 | 46.5 | 23.9 | 46.6 | 56.6 | ICCV19 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

One-stage detector basically formulates object detection as dense classification and localization. The classification is usually optimized by Focal Loss and the box location is commonly learned under Dirac delta distribution. A recent trend for one-stage detectors is to introduce an individual prediction branch to estimate the quality of localization, where the predicted quality facilitates the classification to improve detection performance. This paper delves into the representations of the abo

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

QFL/DFL 已经验证方向正确，但需要补独立 objectness，避免 quality-aware cls 把背景也推高。

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

- [[RetinaNet]]
- [[FCOS]]
- [[ATSS]]
- [[TOOD]]
- [[D-FINE]]
