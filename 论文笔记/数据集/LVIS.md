---
title: "LVIS: A Dataset for Large Vocabulary Instance Segmentation"
method_name: "LVIS"
authors: [Agrim Gupta, Piotr Dollar, Ross Girshick]
year: 2019
venue: CVPR
tags: [dataset, long-tail, instance-segmentation, object-detection]
arxiv: https://arxiv.org/abs/1908.03195
created: 2026-06-11
image_source: online
---

# 论文笔记：LVIS: A Dataset for Large Vocabulary Instance Segmentation

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[LVIS]] |
| 作者 | [Agrim Gupta, Piotr Dollar, Ross Girshick] |
| 年份 | 2019 |
| 会议/来源 | CVPR |
| 链接 | 无 arXiv 链接 / 经典论文 |
| 相关工作 | [[Detic]]、[[COCO]]、[[Grounding DINO]] |

---

## 一句话总结

> LVIS 是大词表、长尾分布的实例分割/检测数据集，推动了长尾检测和大词表识别研究。

---

## 核心贡献

1. **范式贡献**：LVIS 是大词表、长尾分布的实例分割/检测数据集，推动了长尾检测和大词表识别研究。
2. **方法贡献**：构建大词表长尾实例分割数据集，引入 federated 标注挑战，推动 long-tail detection/segmentation。
3. **工程/研究影响**：该工作成为后续 [[Detic]]、[[COCO]]、[[Grounding DINO]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

构建大词表长尾实例分割数据集，引入 federated 标注挑战，推动 long-tail detection/segmentation。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[Detic]]、[[COCO]]。
- 后续影响：[[Grounding DINO]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
AP=\frac{1}{|C|}\sum_{c\in C}AP_c
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1908.03195](https://ar5iv.labs.arxiv.org/html/1908.03195)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 22 张、Table 8 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: Example annotations. We present LVIS , a new dataset for benchmarking Large Vocabulary Instance Segmentation in the 1000+ category regime with a challenging long tail of rare objects.

![Figure 1](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x1.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 2: Figure 2: Category relationships from left to right: non-disjoint category pairs may be in partially overlapping , parent-child , or equivalent (synonym) relationships, implying that a single object may have multiple valid labels. The fair evaluation of an object detector must take the issue of multiple valid labels into account.

![Figure 2](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x2.png)

**图表解读**：重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 3: Figure 3: Example LVIS annotations (one category per image for clarity). See http://www.lvisdataset.org/explore .

![Figure 3](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x3.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 4: Figure 4: Our annotation pipeline comprises six stages. Stage 1: Object Spotting elicits annotators to mark a single instance of many different categories per image. This stage is iterative and causes annotators to discover a long tail of categories. Stage 2: Exhaustive Instance Marking extends the stage 1 annotations to cover all instances of each spotted category. Here we show additional instances of book . Stages 3 and 4: Instance Segmentation and Verification are repeated back and forth until ∼ similar-to \scriptstyle\sim 99% of all segmentations pass a quality check. Stage 5: Exhaustive Annotations Verification checks that all instances are in fact segmented and flags categories that are missing one or more instances. Stage 6: Negative Labels are assigned by verifying that a subset of categories do not appear in the image.

![Figure 4](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x4.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 5: Figure 5: Distribution of object centers in normalized image coordinates for four datasets. ADE20K exhibits the greatest spatial diversity, with LVIS achieving greater complexity than COCO and the Open Images v4 training set. 3

![Figure 5](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/figs/center_distribution.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 6: (a) Distribution of category count per image. LVIS has a heavier tail than COCO and Open Images training set. ADE20K is the most uniform.

![Figure 6](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x5.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 7: (b) The number of instances per category (on 5k images) reveals the long tail with few examples. Orange dots: categories in common with COCO.

![Figure 7](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x6.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 8: (c) Relative segmentation mask size (square root of mask-area-divided-by-image-area) compared between LVIS , COCO, and ADE20K.

![Figure 8](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x7.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 9: (a) LVIS segmentation quality measured by mask IoU between matched instances from two runs of our annotation pipeline. Masks from the runs are consistent with a dataset average IoU of 0.85.

![Figure 9](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x8.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 10: (b) LVIS recognition quality measured by F 1 subscript 𝐹 1 F_{1} score given matched instances across two runs of our annotation pipeline. Category labeling is consistent with a dataset average F 1 subscript 𝐹 1 F_{1} score of 0.87.

![Figure 10](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x9.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 11: (c) Illustration of mask IoU vs . boundary quality to provide intuition for interpreting Fig. 7a (left) and Tab. 1a (dataset annotations vs . expert annotators, below).

![Figure 11](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x10.png)

**图表解读**：重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 12: (a) Given fixed detections, we show how AP varies with | 𝒩 c | subscript 𝒩 𝑐 |\mathcal{N}_{c}| , the number of negative images per category used in evaluation.

![Figure 12](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x11.png)

**图表解读**：重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 13: (b) With the same detections from Fig. 8a and | 𝒩 c | = 50 subscript 𝒩 𝑐 50 |\mathcal{N}_{c}|=50 , we show how AP varies as we vary | 𝒫 c | subscript 𝒫 𝑐 |\mathcal{P}_{c}| , the positive set size.

![Figure 13](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x12.png)

**图表解读**：重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 14: (c) Low-shot detection is an open problem: training Mask R-CNN on 1k images decreases COCO val2017 mask AP from 36% to 10%.

![Figure 14](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x13.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 15: Figure 9: (Left) As more images are annotated, new categories are discovered. (Right) Consequently, the percentage of low-shot categories (blue curve) remains large, decreasing slowly.

![Figure 15](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x14.png)

**图表解读**：重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 16: Figure 9: (Left) As more images are annotated, new categories are discovered. (Right) Consequently, the percentage of low-shot categories (blue curve) remains large, decreasing slowly.

![Figure 16](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x15.png)

**图表解读**：重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 17: Figure 10: Category growth (left) and frequency statistics (right) for LVIS v0.5. Best viewed digitally. Compare with Fig. 9 .

![Figure 17](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x16.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 18: Figure 10: Category growth (left) and frequency statistics (right) for LVIS v0.5. Best viewed digitally. Compare with Fig. 9 .

![Figure 18](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x17.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 19: Figure 11: Distribution of object centers in normalized image coordinates for LVIS val , LVIS val v0.5 ( i.e . after quality control), and LVIS train v0.5. The distributions are nearly identical.

![Figure 19](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/figs/center_distribution_levels_50_limit_50000_mode_kde_v0_5.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 20: Figure 12: The distribution of rare, common, and frequent categories (defined w.r.t. train v0.5) within random image subsets of a given size changes as a function of that size. The shaded region (imperceptible without zoom) illustrates one standard deviation around the mean over 10 draws of subsets for each size.

![Figure 20](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x18.png)

**图表解读**：重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 21: (a) AP of simulated classifiers as a function of the evaluation set size and the fraction of positive examples f c subscript 𝑓 𝑐 f_{c} (the number below each data point indicates the number of positives at that point, the shaded region indicates the standard error when averaged over 300 trials). The left plot shows the behavior of a random Gaussian classifier; the right shows a classifier that mimics the empirical score distribution of a trained classifier. While smaller f c subscript 𝑓 𝑐 f_{c} leads to decreased AP, we also observe a consistent decrease in AP as the evaluation set size increases (until convergence).

![Figure 21](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x19.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### Figure 22: (b) AP bb bb {}^{\textrm{bb}} @75 of a detector on three COCO categories when evaluated on random subsets of different sizes. Toasters are rare ( f c = 0.002 subscript 𝑓 𝑐 0.002 f_{c}=0.002 ) while cats and dining tables appear more frequently. As in simulation, the AP can decreases with larger test set size, especially for rare categories.

![Figure 22](https://ar5iv.labs.arxiv.org/html/1908.03195/assets/x20.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看长尾类别分布与 federated annotation；它提醒训练时必须处理未穷尽标注。

### 关键表格

#### Table 1: LVIS Table 1

|  |  | mask IoU | boundary quality |  |  |
| --- | --- | --- | --- | --- | --- |
| dataset | comparison | mean | median | mean | median |
| COCO | dataset vs . experts | 0.83 – 0.87 | 0.88 – 0.91 | 0.77 – 0.82 | 0.79 – 0.88 |
| expert1 vs . expert2 | 0.91 – 0.95 | 0.96 – 0.98 | 0.92 – 0.96 | 0.97 – 0.99 |  |
| ADE20K | dataset vs . experts | 0.84 – 0.88 | 0.90 – 0.93 | 0.83 – 0.87 | 0.84 – 0.92 |
| expert1 vs . expert2 | 0.90 – 0.94 | 0.95 – 0.97 | 0.90 – 0.95 | 0.99 – 1.00 |  |
| LVIS | dataset vs . experts | 0.90 – 0.92 | 0.94 – 0.96 | 0.87 – 0.91 | 0.93 – 0.98 |
| expert1 vs . expert2 | 0.93 – 0.96 | 0.96 – 0.98 | 0.91 – 0.96 | 0.97 – 1.00 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: LVIS Table 2

|  | annotation | boundary complexity |  |
| --- | --- | --- | --- |
| dataset | source | mean | median |
| COCO | dataset | 5.59 – 6.04 | 5.13 – 5.51 |
| experts | 6.94 – 7.84 | 5.86 – 6.80 |  |
| ADE20K | dataset | 6.00 – 6.84 | 4.79 – 5.31 |
| experts | 6.34 – 7.43 | 4.83 – 5.53 |  |
| LVIS | dataset | 6.35 – 7.07 | 5.44 – 6.00 |
| experts | 7.13 – 8.48 | 5.91 – 6.82 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: LVIS Table 3

| Mask R-CNN | test anno. | box AP | mask AP |
| --- | --- | --- | --- |
| ResNet-50-FPN | COCO | 38.2 | 34.1 |
| model id: 35859007 | LVIS | 38.8 | 34.4 |
| ResNet-101-FPN | COCO | 40.6 | 36.0 |
| model id: 35861858 | LVIS | 40.9 | 36.0 |
| ResNeXt-101-64x4d-FPN | COCO | 47.8 | 41.2 |
| model id: 37129812 | LVIS | 48.6 | 41.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: LVIS Table 4

| score thr | det/img | AP | AP r r {}_{\textrm{r}} | AP c c {}_{\textrm{c}} | AP f f {}_{\textrm{f}} | AP bb bb {}^{\textrm{bb}} |
| --- | --- | --- | --- | --- | --- | --- |
| 0.050 | 100 | 14.8 | 0.8 | 10.9 | 25.3 | 14.8 |
| 0.050 | 300 | 15.7 | 0.8 | 12.1 | 26.1 | 15.6 |
| 0.001 | 300 | 20.8 | 3.3 | 20.7 | 27.9 | 20.3 |
| 0.000 | 300 | 20.9 | 3.4 | 20.9 | 27.9 | 20.4 |
| 0.050 | 100 | 14.8 ± plus-or-minus \pm 0.19 | 0.6 ± plus-or-minus \pm 0.21 | 11.0 ± plus-or-minus \pm 0.36 | 25.2 ± plus-or-minus \pm 0.10 | 14.8 ± plus-or-minus \pm 0.17 |
| 0.000 | 300 | 21.0 ± plus-or-minus \pm 0.17 | 3.2 ± plus-or-minus \pm 0.35 | 21.3 ± plus-or-minus \pm 0.45 | 27.7 ± plus-or-minus \pm 0.12 | 20.5 ± plus-or-minus \pm 0.21 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: LVIS Table 5

| t 𝑡 t | AP | AP r r {}_{\textrm{r}} | AP c c {}_{\textrm{c}} | AP f f {}_{\textrm{f}} |
| --- | --- | --- | --- | --- |
| 0 | 21.0 ± plus-or-minus \pm 0.17 | 3.2 ± plus-or-minus \pm 0.35 | 21.3 ± plus-or-minus \pm 0.45 | 27.7 ± plus-or-minus \pm 0.12 |
| 0.0001 | 21.2 ± plus-or-minus \pm 0.14 | 4.5 ± plus-or-minus \pm 0.47 | 21.5 ± plus-or-minus \pm 0.37 | 27.6 ± plus-or-minus \pm 0.14 |
| 0.0010 | 23.2 ± plus-or-minus \pm 0.21 | 13.4 ± plus-or-minus \pm 0.80 | 23.2 ± plus-or-minus \pm 0.32 | 27.1 ± plus-or-minus \pm 0.07 |
| 0.0100 | 21.8 ± plus-or-minus \pm 0.25 | 9.8 ± plus-or-minus \pm 1.27 | 22.7 ± plus-or-minus \pm 0.48 | 25.6 ± plus-or-minus \pm 0.13 |
| 0.1000 | 21.3 ± plus-or-minus \pm 0.24 | 9.6 ± plus-or-minus \pm 0.83 | 21.7 ± plus-or-minus \pm 0.32 | 25.5 ± plus-or-minus \pm 0.10 |
| CAS | 18.7 ± plus-or-minus \pm 0.46 | 8.5 ± plus-or-minus \pm 1.56 | 19.0 ± plus-or-minus \pm 0.45 | 22.3 ± plus-or-minus \pm 0.19 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: LVIS Table 6

| enhancement | AP | AP r r {}_{\textrm{r}} | AP c c {}_{\textrm{c}} | AP f f {}_{\textrm{f}} |
| --- | --- | --- | --- | --- |
| Table 3b best | 23.2 ± plus-or-minus \pm 0.21 | 13.4 ± plus-or-minus \pm 0.80 | 23.2 ± plus-or-minus \pm 0.32 | 27.1 ± plus-or-minus \pm 0.07 |
| + scale jitter | 24.4 ± plus-or-minus \pm 0.06 | 14.5 ± plus-or-minus \pm 0.67 | 24.3 ± plus-or-minus \pm 0.37 | 28.4 ± plus-or-minus \pm 0.12 |
| + ResNet-101 | 26.0 ± plus-or-minus \pm 0.18 | 15.8 ± plus-or-minus \pm 0.95 | 26.1 ± plus-or-minus \pm 0.21 | 29.8 ± plus-or-minus \pm 0.22 |
| + ResNeXt-101-32 × \times 8d | 27.1 ± plus-or-minus \pm 0.43 | 15.6 ± plus-or-minus \pm 1.14 | 27.5 ± plus-or-minus \pm 0.77 | 31.4 ± plus-or-minus \pm 0.12 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: LVIS Table 7

| subset size | AP | AP r r {}_{\textrm{r}} | AP c c {}_{\textrm{c}} | AP f f {}_{\textrm{f}} |
| --- | --- | --- | --- | --- |
| 5k | 24.8 ± plus-or-minus \pm 0.51 | 11.5 ± plus-or-minus \pm 1.71 | 25.5 ± plus-or-minus \pm 0.86 | 30.1 ± plus-or-minus \pm 0.28 |
| 10k | 22.1 ± plus-or-minus \pm 0.31 | 10.5 ± plus-or-minus \pm 0.66 | 22.9 ± plus-or-minus \pm 0.56 | 29.2 ± plus-or-minus \pm 0.20 |
| 15k | 20.8 ± plus-or-minus \pm 0.23 | 10.0 ± plus-or-minus \pm 0.54 | 21.7 ± plus-or-minus \pm 0.37 | 28.9 ± plus-or-minus \pm 0.11 |
| 20k (full) | 18.4 ± plus-or-minus \pm 0.00 | 8.8 ± plus-or-minus \pm 0.00 | 18.7 ± plus-or-minus \pm 0.00 | 27.2 ± plus-or-minus \pm 0.00 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: LVIS Table 8

| model | eval. set | AP | AP r r {}_{\textrm{r}} | AP c c {}_{\textrm{c}} | AP f f {}_{\textrm{f}} |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 | val | 24.4 | 14.5 | 24.3 | 28.4 |
| test | 18.4 | 8.8 | 18.7 | 27.2 |  |
| ResNet-101 | val | 26.0 | 15.8 | 26.1 | 29.8 |
| test | 20.0 | 9.4 | 21.0 | 28.7 |  |
| ResNeXt-101-32 × \times 8d | val | 27.1 | 15.6 | 27.5 | 31.4 |
| test | 20.5 | 9.8 | 21.1 | 30.0 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

论文通常在 PASCAL VOC、COCO、LVIS 或开放词汇基准上验证效果。阅读时应关注 AP、AP50/AP75、延迟和消融实验。

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

非穷尽标注会破坏高置信训练，接入时必须用 federated/ignore 策略。

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

- [[Detic]]
- [[COCO]]
- [[Grounding DINO]]
