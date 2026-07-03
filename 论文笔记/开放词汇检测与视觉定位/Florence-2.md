---
title: "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks"
method_name: "Florence-2"
authors: [Bin Xiao, et al.]
year: 2023
venue: CVPR
tags: [vision-foundation-model, object-detection, grounding, unified-vision]
arxiv: https://arxiv.org/abs/2311.06242
created: 2026-06-11
image_source: online
---

# 论文笔记：Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[Florence-2]] |
| 作者 | [Bin Xiao, et al.] |
| 年份 | 2023 |
| 会议/来源 | CVPR |
| 链接 | [arXiv](https://arxiv.org/abs/2311.06242) / [PDF](https://arxiv.org/pdf/2311.06242.pdf) |
| 相关工作 | [[Grounding DINO]]、[[Mask R-CNN]]、[[YOLO-World]] |

---

## 一句话总结

> Florence-2 用 prompt-based 方式统一 caption、detection、grounding、segmentation 等视觉任务，是视觉基础模型统一表示路线的代表。

---

## 核心贡献

1. **范式贡献**：Florence-2 用 prompt-based 方式统一 caption、detection、grounding、segmentation 等视觉任务，是视觉基础模型统一表示路线的代表。
2. **方法贡献**：用 prompt-based 统一表示处理 caption、detection、grounding、segmentation 等任务。
3. **工程/研究影响**：该工作成为后续 [[Grounding DINO]]、[[Mask R-CNN]]、[[YOLO-World]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

用 prompt-based 统一表示处理 caption、detection、grounding、segmentation 等任务。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[Grounding DINO]]、[[Mask R-CNN]]。
- 后续影响：[[YOLO-World]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
p(y\mid x,prompt)=\prod_t p(y_t\mid y_{<t},x,prompt)
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2311.06242](https://ar5iv.labs.arxiv.org/html/2311.06242)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 55 张、Table 12 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1 : We aim to build a vision foundation model to enable extensive perception capabilities including spatial hierarchy and semantic granularity. To achieve this, a single unified model Florence-2 is pre-trained on our FLD-5B dataset encompassing a total of 5.4B comprehensive annotations across 126M images, which are collected by our Florence data engine.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 2: Figure 2 : Florence-2 consists of an image encoder and standard multi-modality encoder-decoder. We train Florence-2 on our FLD-5B data in a unified multitask learning paradigm, resulting in a generaslist vision foundation model, which can perform various vision tasks.

![Figure 2](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 3: Figure 3 : Florence-2 data engine consists of three essential phrases: (1) initial annotation employing specialist models, (2) data filtering to correct errors and remove irrelevant annotations, and (3) an iterative process for data refinement. Our final dataset ( FLD-5B ) of over 5B annotations contains 126M images, 500M text annotations, 1.3B region-text annotations, and 3.6B text-phrase-region annotations.

![Figure 3](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x3.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 4: Figure 4 : An illustrative example of an image and its corresponding annotations in FLD-5B dataset. Each image in FLD-5B is annotated with text, region-text pairs, and text-phrase-region triplets by Florence data engine, which covers multiple spatial hierarchies, brief-to-detailed progressive granularity, and a wide semantics spectrum, enabling more comprehensive visual understanding from diverse perspectives.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x4.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 5: (a)

![Figure 5](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x5.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 6: (b)

![Figure 6](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x6.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 7: (c)

![Figure 7](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x7.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 8: (d)

![Figure 8](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x8.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 9: (a) Mask-RCNN on COCO detection.

![Figure 9](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x9.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 10: (b) DINO on COCO detection.

![Figure 10](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x10.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 11: (c) UpperNet on ADE20K.

![Figure 11](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x11.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 12: Figure 7 : Multitask transfer. We conduct experiments with three different versions of Florence-2 models, each trained on a different level of image annotation: image level, image and region level, and image, region, and pixel level. We then evaluate the transfer learning performance of these models on four downstream tasks: COCO caption, COCO object detection, Flickr30k grounding, and Refcoco referring segmentation.

![Figure 12](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x12.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 13: Table 15: Appendix D

![Figure 13](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x13.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 14: Appendix D Figure 8

![Figure 14](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x14.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 15: Florence-2 Figure 15

![Figure 15](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/cap_1.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 16: Florence-2 Figure 16

![Figure 16](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/cap_2.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 17: Florence-2 Figure 17

![Figure 17](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/cap_3.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 18: Figure 10 E.2

![Figure 18](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/cap_4.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 19: Florence-2 Figure 19

![Figure 19](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_6.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 20: Florence-2 Figure 20

![Figure 20](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_7.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 21: Florence-2 Figure 21

![Figure 21](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_8.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 22: Florence-2 Figure 22

![Figure 22](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_1.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 23: Florence-2 Figure 23

![Figure 23](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_2.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 24: Florence-2 Figure 24

![Figure 24](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/grounding_5.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 25: Florence-2 Figure 25

![Figure 25](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_1.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 26: Florence-2 Figure 26

![Figure 26](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_2.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 27: Florence-2 Figure 27

![Figure 27](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_3.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 28: Figure 13

![Figure 28](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_4.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 29: Figure 13 E.4

![Figure 29](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_5.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 30: Figure 13 E.4

![Figure 30](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/dense_cap_6.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 31: Florence-2 Figure 31

![Figure 31](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_1.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 32: Florence-2 Figure 32

![Figure 32](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_2.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 33: Florence-2 Figure 33

![Figure 33](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_3.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 34: Florence-2 Figure 34

![Figure 34](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_4.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 35: Figure 14 E.5

![Figure 35](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_5.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 36: Figure 14 E.5

![Figure 36](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/od_6.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 37: Florence-2 Figure 37

![Figure 37](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/ocr_1.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 38: Florence-2 Figure 38

![Figure 38](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/ocr_4.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 39: Florence-2 Figure 39

![Figure 39](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/ocr_2.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 40: Florence-2 Figure 40

![Figure 40](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_1.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 41: Florence-2 Figure 41

![Figure 41](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_2.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 42: Florence-2 Figure 42

![Figure 42](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_3.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 43: Florence-2 Figure 43

![Figure 43](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_4.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 44: Florence-2 Figure 44

![Figure 44](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_5.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 45: Figure 16 Appendix F

![Figure 45](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/pred_results/seg_6.png)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 46: Florence-2 Figure 46

![Figure 46](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/x15.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 47: Florence-2 Figure 47

![Figure 47](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/com_5.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 48: Florence-2 Figure 48

![Figure 48](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/com_2.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 49: Florence-2 Figure 49

![Figure 49](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/com_3.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 50: Florence-2 Figure 50

![Figure 50](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/ouput_1_kosmos2.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 51: Florence-2 Figure 51

![Figure 51](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/output1_fld2.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 52: Florence-2 Figure 52

![Figure 52](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/output_2_kosmos2.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 53: Florence-2 Figure 53

![Figure 53](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/output2_fld2.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 54: Florence-2 Figure 54

![Figure 54](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/output_3_kosmos2.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### Figure 55: Florence-2 Figure 55

![Figure 55](https://ar5iv.labs.arxiv.org/html/2311.06242/assets/figures/appendix/comparison/output3_fld2.jpg)

**图表解读**：重点看 prompt-based 多任务统一；它更像统一感知接口，不是轻量检测头本体。

### 关键表格

#### Table 1: Florence-2 Table 1

| Dataset | Rep. Model | #Images | #Annotations | Spatial hierarchy | Semantics granularity |
| --- | --- | --- | --- | --- | --- |
| JFT300M [ 21 ] | ViT | 300M | 300M | Image-level | Coarse |
| WIT [ 64 ] | CLIP | 400M | 400M | Image-level | Coarse |
| SA-1B [ 32 ] | SAM | 11M | 1B | Region-level | Non-semantic |
| GrIT [ 60 ] | Kosmos-2 | 91M | 137M | Image & Region-level | Fine-grained |
| M3W [ 2 ] | Flamingo | 185M | 43.3M* | Multi-image-level | Fine-grained |
| FLD-5B (ours) | Florence-2 (ours) | 126M | 5B | Image & Region-level | Coarse to fine-grained |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: Florence-2 Table 2

| Method | #params |  | COCO Cap. |  | NoCaps |  | TextCaps |  | COCO Det. |  | Flickr30k |  | Refcoco |  | Refcoco+ |  | Refcocog |  | Refcoco RES |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | test |  | val |  | val |  | val2017 |  | test |  | val | test-A | test-B |  | val | test-A | test-B |  | val | test |  | val |  |  |
|  | CIDEr |  | CIDEr |  | CIDEr |  | mAP |  | R@1 |  | Accuracy |  | Accuracy |  | Accuracy |  | mIoU |  |  |  |  |  |  |  |
| Flamingo [ 2 ] | 80B |  | 84.3 |  | - |  | - |  | - |  | - |  | - | - | - |  | - | - | - |  | - | - |  | - |
| Kosmos-2 [ 60 ] | 1.6B |  | - |  | - |  | - |  | - |  | 78.7 |  | 52.3 | 57.4 | 47.3 |  | 45.5 | 50.7 | 42.2 |  | 60.6 | 61.7 |  | - |
| Florence-2-B | 0.23B |  | 133.0 |  | 118.7 |  | 70.1 |  | 34.7 |  | 83.6 |  | 53.9 | 58.4 | 49.7 |  | 51.5 | 56.4 | 47.9 |  | 66.3 | 65.1 |  | 34.6 |
| Florence-2-L | 0.77B |  | 135.6 |  | 120.8 |  | 72.8 |  | 37.5 |  | 84.4 |  | 56.3 | 61.6 | 51.4 |  | 53.6 | 57.9 | 49.9 |  | 68.0 | 67.0 |  | 35.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: Florence-2 Table 3

| Method | #params | COCO Caption | NoCaps | TextCaps | VQAv2 | TextVQA | VizWiz VQA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Karpathy test | val | val | test-dev | test-dev | test-dev |  |  |
| CIDEr | CIDEr | CIDEr | Acc | Acc | Acc |  |  |
| Specialist Models |  |  |  |  |  |  |  |
| CoCa [ 92 ] | 2.1B | 143.6 | 122.4 | - | 82.3 | - | - |
| BLIP-2 [ 44 ] | 7.8B | 144.5 | 121.6 | - | 82.2 | - | - |
| GIT2 [ 78 ] | 5.1B | 145 | 126.9 | 148.6 | 81.7 | 67.3 | 71.0 |
| Flamingo [ 2 ] | 80B | 138.1 | - | - | 82.0 | 54.1 | 65.7 |
| PaLI [ 15 ] | 17B | 149.1 | 127.0 | 160.0 △ | 84.3 | 58.8 / 73.1 △ | 71.6 / 74.4 △ |
| PaLI-X [ 12 ] | 55B | 149.2 | 126.3 | 147 / 163.7 △ | 86.0 | 71.4 / 80.8 △ | 70.9 / 74.6 △ |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: Florence-2 Table 4

| Method | #params |  | COCO Det. |  | Flickr30k |  | Refcoco |  | Refcoco+ |  | Refcocog |  | Refcoco RES |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | val2017 |  | test |  | val | test-A | test-B |  | val | test-A | test-B |  | val | test |  | val |  |  |
|  | mAP |  | R@1 |  | Accuracy |  | Accuracy |  | Accuracy |  | mIoU |  |  |  |  |  |  |  |
| Specialist Models |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqTR [ 99 ] | - |  | - |  | - |  | 83.7 | 86.5 | 81.2 |  | 71.5 | 76.3 | 64.9 |  | 74.9 | 74.2 |  | - |
| PolyFormer [ 49 ] | - |  | - |  | - |  | 90.4 | 92.9 | 87.2 |  | 85.0 | 89.8 | 78.0 |  | 85.8 | 85.9 |  | 76.9 |
| UNINEXT [ 84 ] | 0.74B |  | 60.6 |  | - |  | 92.6 | 94.3 | 91.5 |  | 85.2 | 89.6 | 79.8 |  | 88.7 | 89.4 |  | - |
| Ferret [ 90 ] | 13B |  | - |  | - |  | 89.5 | 92.4 | 84.4 |  | 82.8 | 88.1 | 75.2 |  | 85.8 | 86.3 |  | - |
| Generalist Models |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| UniTAB [ 88 ] |  |  | - |  | - |  | 88.6 | 91.1 | 83.8 |  | 81.0 | 85.4 | 71.6 |  | 84.6 | 84.7 |  | - |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: Florence-2 Table 5

|  |  |  | Mask R-CNN |  | DINO |  |
| --- | --- | --- | --- | --- | --- | --- |
| Backbone | Pretrain |  | AP b | AP m |  | AP |
| ViT-B [ 46 ] | MAE, IN-1k |  | 51.6 | 45.9 |  | 55.0 |
| Swin-B [ 51 ] | Sup IN-1k |  | 50.2 | - |  | 53.4 |
| Swin-B [ 51 ] | SimMIM [ 83 ] |  | 52.3 | - |  | - |
| FocalAtt-B [ 86 ] | Sup IN-1k |  | 49.0 | 43.7 |  | - |
| FocalNet-B [ 85 ] | Sup IN-1k |  | 49.8 | 44.1 |  | 54.4 |
| ConvNeXt v1-B [ 52 ] | Sup IN-1k |  | 50.3 | 44.9 |  | 52.6 |
| ConvNeXt v2-B [ 81 ] | Sup IN-1k |  | 51.0 | 45.6 |  | - |
| ConvNeXt v2-B [ 81 ] | FCMAE |  | 52.9 | 46.6 |  | - |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: Florence-2 Table 6

| Pretrain | Frozen stages |  | Mask R-CNN |  | DINO |  | UperNet |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | AP b | AP m |  | AP |  | mIoU |  |  |
| Sup IN1k | n/a |  | 46.7 | 42.0 |  | 53.7 |  | 49 |
| UniCL [ 87 ] | n/a |  | 50.4 | 45.0 |  | 57.3 |  | 53.6 |
| Florence-2 | n/a |  | 53.6 | 46.4 |  | 59.2 |  | 54.9 |
| Florence-2 | [1] |  | 53.6 | 46.3 |  | 59.2 |  | 54.1 |
| Florence-2 | [1, 2] |  | 53.3 | 46.1 |  | 59.0 |  | 54.4 |
| Florence-2 | [1, 2, 3] |  | 49.5 | 42.9 |  | 56.7 |  | 49.6 |
| Florence-2 | [1, 2, 3, 4] |  | 48.3 | 44.5 |  | 56.1 |  | 45.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: Florence-2 Table 7

| Backbone | Pretrain | mIoU | ms-mIoU |
| --- | --- | --- | --- |
| ViT-B [ 24 ] | Sup IN-1k | 47.4 | - |
| ViT-B [ 24 ] | MAE IN-1k | 48.1 | - |
| ViT-B [ 4 ] | BEiT | 53.6 | 54.1 |
| ViT-B [ 59 ] | BEiTv2 IN-1k | 53.1 | - |
| ViT-B [ 59 ] | BEiTv2 IN-22k | 53.5 | - |
| Swin-B [ 51 ] | Sup IN-1k | 48.1 | 49.7 |
| Swin-B [ 51 ] | Sup IN-22k | - | 51.8 |
| Swin-B [ 51 ] | SimMIM [ 83 ] | - | 52.8 |
| FocalAtt-B [ 86 ] | Sup IN-1k | 49.0 | 50.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: Florence-2 Table 8

| Model |  | Caption |  | Detection |  | Grounding |  | RES |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | CIDEr |  | AP |  | Recall@1 |  | mIOU | oIOU |  |
| Base |  | 118.7 |  | 19.7 |  | 76.3 |  | 18.6 | 17.8 |
| Large |  | 124.4 |  | 22.6 |  | 78.2 |  | 21.5 | 19.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 9: Florence-2 Table 9

| Data |  | Caption |  | Detection |  | Grounding |  | RES |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| size |  | CIDEr |  | AP |  | Recall@1 |  | mIOU | oIOU |
| 0.12M |  | 102.8 |  | 16.1 |  | 74.0 |  | 15.9 | 16.6 |
| 0.36M |  | 114.3 |  | 18.7 |  | 75.8 |  | 16.6 | 16.4 |
| 1.2M |  | 118.1 |  | 18.9 |  | 76.3 |  | 19.3 | 18.4 |
| 12M |  | 118.7 |  | 19.7 |  | 76.3 |  | 18.6 | 17.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 10: Florence-2 Table 10

|  |  |  | Caption |  | Detection |  | Grounding |  | RES |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V Pre | L Pre |  | CIDEr |  | AP |  | Recall@1 |  | mIOU | oIOU |
| Freeze Vision Encoder |  |  |  |  |  |  |  |  |  |  |
| ✓ | ✓ |  | 120.0 |  | 6.9 |  | 66.3 |  | 9.9 | 13.6 |
| Unfreeze Vision Encoder |  |  |  |  |  |  |  |  |  |  |
|  | ✓ |  | 81.3 |  | 4.9 |  | 69.0 |  | 15.3 | 15.6 |
| ✓ |  |  | 117.4 |  | 19.6 |  | 75.2 |  | 21.5 | 19.3 |
| ✓ | ✓ |  | 118.7 |  | 19.7 |  | 76.3 |  | 18.6 | 17.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 11: Florence-2 Table 11

| Task | Annotation Type | Prompt Input | Output |
| --- | --- | --- | --- |
| Caption | Text | Image, text | Text |
| Detailed caption | Text | Image, text | Text |
| More detailed caption | Text | Image, text | Text |
| Region proposal | Region | Image, text | Region |
| Object detection | Region-Text | Image, text | Text, region |
| Dense region caption | Region-Text | Image, text | Text, region |
| Phrase grounding | Text-Phrase-Region | Image, text | Text, region |
| Referring expression comprehension | Region-Text | Image, text | Text, region |
| Open vocabulary detection | Region-Text | Image, text | Text, region |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 12: Florence-2 Table 12

| Task | Dataset |
| --- | --- |
| Caption | COCO [ 13 ] |
| Text Caption | TextCaps [ 72 ] |
| Paragraph caption | Standford Paragraph Caption [ 35 ] |
| Detailed caption | Localized Narratives [ 62 ] |
| Detection | COCO [ 47 ] , Object365 ∗ [ 70 ] , Open Images ∗ [ 39 ] |
| Phrase Grounding | Flickr30k, Object365 ∗ [ 70 ] , Open Images ∗ [ 39 ] |
| Referring expression | RefCOCO-mix (RefCOCO, RefCOCO+, RefCOCOg) [ 31 , 93 , 56 ] |
| Referring expression segmentation | RefCOCO-mix (RefCOCO, RefCOCO+, RefCOCOg) [ 31 , 93 , 56 ] |
| Region to category | COCO [ 47 ] , Object365 ∗ [ 70 ] , Open Images ∗ [ 39 ] |

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

- [[Grounding DINO]]
- [[Mask R-CNN]]
- [[YOLO-World]]
