---
title: "Grounded Language-Image Pre-training"
method_name: "GLIP"
authors: [Liunian Harold Li, et al.]
year: 2021
venue: CVPR
tags: [open-vocabulary-detection, grounding, vision-language, pretraining]
arxiv: https://arxiv.org/abs/2112.03857
created: 2026-06-11
image_source: online
---

# 论文笔记：Grounded Language-Image Pre-training

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[GLIP]] |
| 作者 | [Liunian Harold Li, et al.] |
| 年份 | 2021 |
| 会议/来源 | CVPR |
| 链接 | [arXiv](https://arxiv.org/abs/2112.03857) / [PDF](https://arxiv.org/pdf/2112.03857.pdf) |
| 相关工作 | [[Detic]]、[[Grounding DINO]]、[[YOLO-World]] |

---

## 一句话总结

> GLIP 将目标检测和 phrase grounding 统一到 grounded language-image pretraining 中，是开放词汇检测的重要早期工作。

---

## 核心贡献

1. **范式贡献**：GLIP 将目标检测和 phrase grounding 统一到 grounded language-image pretraining 中，是开放词汇检测的重要早期工作。
2. **方法贡献**：把 detection 和 phrase grounding 统一为 grounded language-image pretraining，学习区域与文本短语对齐。
3. **工程/研究影响**：该工作成为后续 [[Detic]]、[[Grounding DINO]]、[[YOLO-World]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

把 detection 和 phrase grounding 统一为 grounded language-image pretraining，学习区域与文本短语对齐。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[Detic]]、[[Grounding DINO]]。
- 后续影响：[[YOLO-World]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
s(r,t)=\phi_I(r)^\top\phi_T(t)
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2112.03857](https://ar5iv.labs.arxiv.org/html/2112.03857)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 8 张、Table 12 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1 : GLIP zero-shot transfers to various detection tasks, by writing the categories of interest into a text prompt.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2112.03857/assets/x1.png)

**图表解读**：重点看 detection 与 grounding 的统一预训练图；可作为开放词汇 teacher 或长尾类别伪标注来源。

### Figure 2: Figure 2 : A unified framework for detection and grounding. Unlike a classical object detection model which predicts a categorical class for each detected object, we reformulate detection as a grounding task by aligning each region/box to phrases in a text prompt. GLIP jointly trains an image encoder and a language encoder to predict the correct pairings of regions and words. We further add the cross-modality deep fusion to early fuse information from two modalities and to learn a language-aware visual representation.

![Figure 2](https://ar5iv.labs.arxiv.org/html/2112.03857/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 detection 与 grounding 的统一预训练图；可作为开放词汇 teacher 或长尾类别伪标注来源。

### Figure 3: Figure 2 : A unified framework for detection and grounding. Unlike a classical object detection model which predicts a categorical class for each detected object, we reformulate detection as a grounding task by aligning each region/box to phrases in a text prompt. GLIP jointly trains an image encoder and a language encoder to predict the correct pairings of regions and words. We further add the cross-modality deep fusion to early fuse information from two modalities and to learn a language-aware visual representation.

![Figure 3](https://ar5iv.labs.arxiv.org/html/2112.03857/assets/x3.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 detection 与 grounding 的统一预训练图；可作为开放词汇 teacher 或长尾类别伪标注来源。

### Figure 4: Figure 4 : Data efficiency of models. X-axis is the amount of task-specific data, from zero-shot to all data. Y-axis is the average AP across 13 datasets. GLIP exhibits great data efficiency, while each of our proposed approach contributes to the data efficiency.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2112.03857/assets/x4.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 detection 与 grounding 的统一预训练图；可作为开放词汇 teacher 或长尾类别伪标注来源。

### Figure 5: Figure 5 : Per dataset zero-shot performance. The first 3 datasets contain novel categories not present in the Objects365 vocabulary while the last 2 datasets’ categories are covered by Objects365 data. Grounding data bring significant benefit to novel categories.

![Figure 5](https://ar5iv.labs.arxiv.org/html/2112.03857/assets/x5.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 detection 与 grounding 的统一预训练图；可作为开放词汇 teacher 或长尾类别伪标注来源。

### Figure 6: Figure 6 : A manual prompt tuning example from the Aquarium dataset in ODinW. Given an expressive prompt (“flat and round”), zero-shot GLIP can detect the novel entity “stingray” better.

![Figure 6](https://ar5iv.labs.arxiv.org/html/2112.03857/assets/x6.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 detection 与 grounding 的统一预训练图；可作为开放词汇 teacher 或长尾类别伪标注来源。

### Figure 7: Figure 7 : Effectiveness of prompt tuning. Solid lines are full-model tuning performance; dashed lines are prompt/linear probing performance. By only tuning the prompt embeddings, GLIP-T and GLIP-L can achieve performance close to full-model tuning, allowing for efficient deployment.

![Figure 7](https://ar5iv.labs.arxiv.org/html/2112.03857/assets/x7.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 detection 与 grounding 的统一预训练图；可作为开放词汇 teacher 或长尾类别伪标注来源。

### Figure 8: Figure 8 : Predictions from the teacher model on 6 examples from Conceptual Captions 12M. Phrases and corresponding boxes are matched with the same colors.

![Figure 8|78](https://ar5iv.labs.arxiv.org/html/2112.03857/assets/x8.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 detection 与 grounding 的统一预训练图；可作为开放词汇 teacher 或长尾类别伪标注来源。

### 关键表格

#### Table 1: GLIP Table 1

| Model | Backbone | Deep Fusion | Pre-Train Data |  |  |
| --- | --- | --- | --- | --- | --- |
| Detection | Grounding | Caption |  |  |  |
| GLIP-T (A) | Swin-T | ✗ | Objects365 | - | - |
| GLIP-T (B) | Swin-T | ✓ | Objects365 | - | - |
| GLIP-T (C) | Swin-T | ✓ | Objects365 | GoldG | - |
| GLIP-T | Swin-T | ✓ | Objects365 | GoldG | Cap4M |
| GLIP-L | Swin-L | ✓ | FourODs | GoldG | Cap24M |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: GLIP Table 2

| Model | Backbone | Pre-Train Data | Zero-Shot | Fine-Tune |
| --- | --- | --- | --- | --- |
| 2017val | 2017val / test-dev |  |  |  |
| \rowfont Faster RCNN | RN50-FPN | - | - | 40.2 / - |
| \rowfont Faster RCNN | RN101-FPN | - | - | 42.0 / - |
| \rowfont DyHead-T [ 10 ] | Swin-T | - | - | 49.7 / - |
| \rowfont DyHead-L [ 10 ] | Swin-L | - | - | 58.4 / 58.7 |
| \rowfont DyHead-L [ 10 ] | Swin-L | O365,ImageNet21K | - | 60.3 / 60.6 |
| \rowfont SoftTeacher [ 65 ] | Swin-L | O365,SS-COCO | - | 60.7 / 61.3 |
| DyHead-T | Swin-T | O365 | 43.6 | 53.3 / - |
| GLIP-T (A) | Swin-T | O365 | 42.9 | 52.9 / - |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: GLIP Table 3

| Model | Backbone | MiniVal [ 23 ] | Val v1.0 |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| APr | APc | APf | AP | APr | APc | APf | AP |  |  |
| \rowfont MDETR [ 23 ] | RN101 | 20.9 | 24.9 | 24.3 | 24.2 | - | - | - | - |
| \rowfont MaskRCNN [ 23 ] | RN101 | 26.3 | 34.0 | 33.9 | 33.3 | - | - | - | - |
| \rowfont Supervised-RFS [ 15 ] | RN50 | - | - | - | - | 12.3 | 24.3 | 32.4 | 25.4 |
| GLIP-T (A) | Swin-T | 14.2 | 13.9 | 23.4 | 18.5 | 6.0 | 8.0 | 19.4 | 12.3 |
| GLIP-T (B) | Swin-T | 13.5 | 12.8 | 22.2 | 17.8 | 4.2 | 7.6 | 18.6 | 11.3 |
| GLIP-T (C) | Swin-T | 17.7 | 19.5 | 31.0 | 24.9 | 7.5 | 11.6 | 26.1 | 16.5 |
| GLIP-T | Swin-T | 20.8 | 21.4 | 31.0 | 26.0 | 10.1 | 12.5 | 25.5 | 17.2 |
| GLIP-L | Swin-L | 28.2 | 34.3 | 41.5 | 37.3 | 17.1 | 23.3 | 35.4 | 26.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: GLIP Table 4

| Row | Pre-Training Data | COCO |  | LVIS MiniVal |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2017val |  | AP r | AP c | AP f | AP |  |  |
| 1 | VG w/o COCO | 26.9 |  | 4.9 | 10.4 | 23.2 | 16.1 |
| 2 | + GoldG | 29.2 |  | 7.8 | 14.0 | 24.5 | 18.5 |
| 3 | OpenImages | 29.9 |  | 12.8 | 12.1 | 17.8 | 14.9 |
| 4 | + GoldG | 33.6 |  | 15.2 | 16.9 | 24.5 | 20.4 |
| 5 | O365 | 44.9 |  | 13.5 | 12.8 | 22.2 | 17.8 |
| 6 | +GoldG | 46.7 |  | 17.7 | 19.5 | 31.0 | 24.9 |
| 7 | O365,GoldG,Cap4M | 46.3 |  | 20.8 | 21.4 | 31.0 | 26.0 |
| 8 | FourODs | 46.3 |  | 15.0 | 22.5 | 32.8 | 26.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: GLIP Table 5

| Pre-Train Data | COCO | LVIS minival | Flickr30K val |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-Shot | Fine-Tune | APr | APc | APf | AP | R@1 | R@5 | R@10 |  |
| O365,GoldG,Cap4M | 46.3 | 54.9 | 20.8 | 21.4 | 31.0 | 26.0 | 85.7 | 95.4 | 96.9 |
| O365,GoldG,CC3M,SBU | 46.6 | 55.2 | 20.1 | 21.3 | 31.1 | 25.9 | 85.3 | 95.7 | 97.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: GLIP Table 6

| Model | Fusion | Inference (P100) | Train (V100) |  |  |
| --- | --- | --- | --- | --- | --- |
| Speed | Memory | Speed | Memory |  |  |
| GLIP-T | ✗ | 4.84 FPS | 1.0 GB | 2.79 FPS | 11.5 GB |
| ✓ | 2.52 FPS | 2.4 GB | 1.62 FPS | 16.0 GB |  |
| GLIP-L | ✗ | 0.54 FPS | 4.8 GB | 1.27 FPS | 19.7 GB |
| ✓ | 0.32 FPS | 7.7 GB | 0.88 FPS | 23.4 GB |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: GLIP Table 7

| Deep Fusion | Data | COCO | LVIS minival | Flickr30K val | ODinW |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-Shot | Fine-Tune | APr | APc | APf | AP | R@1 | R@5 | R@10 | 0-Shot | 1-Shot | 3-Shot | 5-Shot | 10-Shot | Full-Shot |  |  |
| ✗ | O365 | 42.9 | 52.9 | 14.2 | 13.9 | 23.4 | 18.5 | 46.4 | 63.2 | 66.9 | 28.7 | 43.5 | 48.8 | 50.4 | 54.1 | 63.6 |
| ✓ | O365 | 44.9 | 53.8 | 13.5 | 12.8 | 22.2 | 17.8 | 41.4 | 57.7 | 61.0 | 33.2 | 48.0 | 52.0 | 53.2 | 54.9 | 62.7 |
| ✗ | O365,GoldG | 41.6 | 52.9 | 15.8 | 23.0 | 30.8 | 26.1 | 82.4 | 94.7 | 96.6 | 35.5 | 47.2 | 51.9 | 53.8 | 54.3 | 65.1 |
| ✓ | O365,GoldG | 46.7 | 55.1 | 17.7 | 19.5 | 31.0 | 24.9 | 84.8 | 94.9 | 96.3 | 44.4 | 49.6 | 53.8 | 54.8 | 57.2 | 63.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: GLIP Table 8

| Dataset | Objects of Interest | Train/Val/Test | URL |
| --- | --- | --- | --- |
| PascalVOC | Common objects (PascalVOC 2012) | 13690/3422/- | https://public.roboflow.com/object-detection/pascal-voc-2012 |
| AerialDrone | Boats, cars, etc. from drone images | 52/15/7 | https://public.roboflow.com/object-detection/aerial-maritime |
| Aquarium | Penguins, starfish, etc. in an aquarium | 448/127/63 | https://public.roboflow.com/object-detection/aquarium |
| Rabbits | Cottontail rabbits | 1980/19/10 | https://public.roboflow.com/object-detection/cottontail-rabbits-video-dataset |
| EgoHands | Hands in ego-centric images | 3840/480/480 | https://public.roboflow.com/object-detection/hands |
| Mushrooms | Two kinds of mushrooms | 41/5/5 | https://public.roboflow.com/object-detection/na-mushrooms |
| Packages | Delivery packages | 19/4/3 | https://public.roboflow.com/object-detection/packages-dataset |
| Raccoon | Raccoon | 150/29/17 | https://public.roboflow.com/object-detection/raccoon |
| Shellfish | Shrimp, lobster, and crab | 406/116/58 | https://public.roboflow.com/object-detection/shellfish-openimages |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 9: GLIP Table 9

| Dataset | Original Prompt | AP | Manually Designed Prompts | AP |
| --- | --- | --- | --- | --- |
| Aquarium | penguin | 17.7 | penguin , which is black and white | 18.4 |
| puffin | puffin with orange beaks |  |  |  |
| stingray | stingray which is flat and round |  |  |  |
| Rabbits | Cottontail-Rabbits | 68.0 | rabbit | 70.2 |
| EgoHands | hand | 22.1 | hand of a person | 50.0 |
| Mushrooms | Cow. Chanterelle | 13.6 | flat mushroom . yellow mushroom | 73.8 |
| Packages | package | 50.0 | there is a package on the porch | 72.3 |
| Pothole | pothole | 17.8 | there are some holes on the road | 17.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 10: GLIP Table 10

| Model | Zero Shot | Full Tuning |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 5 | 10 | All |  |  |
| DyHead-T COCO | - | 31.9 ± plus-or-minus \pm 4.1 | 44.2 ± plus-or-minus \pm 0.4 | 44.7 ± plus-or-minus \pm 2.1 | 50.1 ± plus-or-minus \pm 2.0 | 63.2 |
| DyHead-T O365 | - | 33.8 ± plus-or-minus \pm 4.3 | 43.6 ± plus-or-minus \pm 1.2 | 46.4 ± plus-or-minus \pm 1.4 | 50.8 ± plus-or-minus \pm 1.6 | 60.8 |
| GLIP-T (A) | 28.7 | 43.5 ± plus-or-minus \pm 1.5 | 48.8 ± plus-or-minus \pm 0.4 | 50.4 ± plus-or-minus \pm 0.7 | 54.1 ± plus-or-minus \pm 0.5 | 63.6 |
| GLIP-T (B) | 33.2 | 48.0 ± plus-or-minus \pm 0.8 | 52.0 ± plus-or-minus \pm 0.4 | 53.2 ± plus-or-minus \pm 0.9 | 54.9 ± plus-or-minus \pm 0.7 | 62.7 |
| GLIP-T (C) | 44.4 | 49.6 ± plus-or-minus \pm 0.3 | 53.8 ± plus-or-minus \pm 0.2 | 54.8 ± plus-or-minus \pm 1.0 | 57.2 ± plus-or-minus \pm 1.1 | 63.9 |
| GLIP-T | 46.5 | 51.1 ± plus-or-minus \pm 0.1 | 54.9 ± plus-or-minus \pm 0.3 | 56.4 ± plus-or-minus \pm 0.5 | 58.4 ± plus-or-minus \pm 0.2 | 64.9 |
| GLIP-L | 52.1 | 59.9 ± plus-or-minus \pm 1.7 | 62.1 ± plus-or-minus \pm 0.8 | 64.2 ± plus-or-minus \pm 0.4 | 64.9 ± plus-or-minus \pm 0.9 | 68.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 11: GLIP Table 11

| Model | Linear Probing |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | 5 | 10 | All |  |
| DyHead-T COCO | 22.7 ± plus-or-minus \pm 1.1 | 32.7 ± plus-or-minus \pm 1.4 | 30.5 ± plus-or-minus \pm 2.9 | 34.1 ± plus-or-minus \pm 1.4 | 43.1 |
| DyHead-T COCO-Cosine | 21.8 ± plus-or-minus \pm 4.4 | 30.6 ± plus-or-minus \pm 2.2 | 33.3 ± plus-or-minus \pm 1.2 | 35.5 ± plus-or-minus \pm 1.2 | 43.5 |
| DyHead-T O365 | 30.7 ± plus-or-minus \pm 3.3 | 36.2 ± plus-or-minus \pm 3.3 | 39.6 ± plus-or-minus \pm 0.4 | 40.0 ± plus-or-minus \pm 2.7 | 48.2 |
| DyHead-T O365-Cosine | 25.2 ± plus-or-minus \pm 2.6 | 37.6 ± plus-or-minus \pm 0.5 | 38.9 ± plus-or-minus \pm 0.7 | 41.5 ± plus-or-minus \pm 0.5 | 49.4 |
| GLIP-T (A) | 34.6 ± plus-or-minus \pm 0.7 | 35.9 ± plus-or-minus \pm 0.2 | 37.6 ± plus-or-minus \pm 0.1 | 37.9 ± plus-or-minus \pm 0.2 | 44.1 |
| GLIP-T (B) | 40.9 ± plus-or-minus \pm 0.3 | 42.8 ± plus-or-minus \pm 0.4 | 44.0 ± plus-or-minus \pm 0.2 | 44.4 ± plus-or-minus \pm 0.3 | 51.8 |
| GLIP-T (C) | 43.9 ± plus-or-minus \pm 0.1 | 45.4 ± plus-or-minus \pm 0.1 | 45.9 ± plus-or-minus \pm 0.2 | 46.7 ± plus-or-minus \pm 0.3 | 52.7 |
| GLIP-T | 48.9 ± plus-or-minus \pm 0.2 | 50.5 ± plus-or-minus \pm 0.1 | 50.4 ± plus-or-minus \pm 0.3 | 51.2 ± plus-or-minus \pm 0.2 | 55.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 12: GLIP Table 12

| Model | PascalVOC | AerialDrone | Aquarium | Rabbits | EgoHands | Mushrooms | Packages | Raccoon | Shellfish | Vehicles | Pistols | Pothole | Thermal | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLIP-T (A) | 47.7 | 9.8 | 16.8 | 60.5 | 1.6 | 13.7 | 48.5 | 44.4 | 20.4 | 52.4 | 25.3 | 0.8 | 32.3 | 28.8 |
| GLIP-T (B) | 50.6 | 4.9 | 19.4 | 71.6 | 0.5 | 21.8 | 29.7 | 47.0 | 21.4 | 56.0 | 47.4 | 3.6 | 57.1 | 33.2 |
| GLIP-T (C) | 51.6 | 8.1 | 22.6 | 71.1 | 49.1 | 69.4 | 65.6 | 51.5 | 29.3 | 49.9 | 42.7 | 17.0 | 49.2 | 44.4 |
| GLIP-T | 56.2 | 12.5 | 18.4 | 70.2 | 50.0 | 73.8 | 72.3 | 57.8 | 26.3 | 56.0 | 49.6 | 17.7 | 44.1 | 46.5 |
| GLIP-L | 61.7 | 7.1 | 26.9 | 75.0 | 45.5 | 49.0 | 62.8 | 63.3 | 68.9 | 57.3 | 68.6 | 25.7 | 66.0 | 52.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

This paper presents a grounded language-image pre-training (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making 

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

- [[Detic]]
- [[Grounding DINO]]
- [[YOLO-World]]
