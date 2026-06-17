---
title: "YOLO-World: Real-Time Open-Vocabulary Object Detection"
method_name: "YOLO-World"
authors: [Tianheng Cheng, et al.]
year: 2024
venue: CVPR
tags: [open-vocabulary-detection, yolo, real-time-detection, vision-language]
arxiv: https://arxiv.org/abs/2401.17270
created: 2026-06-11
image_source: online
---

# 论文笔记：YOLO-World: Real-Time Open-Vocabulary Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[YOLO-World]] |
| 作者 | [Tianheng Cheng, et al.] |
| 年份 | 2024 |
| 会议/来源 | CVPR |
| 链接 | [arXiv](https://arxiv.org/abs/2401.17270) / [PDF](https://arxiv.org/pdf/2401.17270.pdf) |
| 相关工作 | [[YOLOX]]、[[GLIP]]、[[Grounding DINO]]、[[OWLv2]] |

---

## 一句话总结

> YOLO-World 将开放词汇检测与 YOLO 的实时性结合，证明开放词汇检测也可以向高效部署靠近。

---

## 核心贡献

1. **范式贡献**：YOLO-World 将开放词汇检测与 YOLO 的实时性结合，证明开放词汇检测也可以向高效部署靠近。
2. **方法贡献**：把文本类别嵌入接入 YOLO-style detector，实现实时开放词汇检测。
3. **工程/研究影响**：该工作成为后续 [[YOLOX]]、[[GLIP]]、[[Grounding DINO]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

把文本类别嵌入接入 YOLO-style detector，实现实时开放词汇检测。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[YOLOX]]、[[GLIP]]。
- 后续影响：[[Grounding DINO]]、[[OWLv2]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
s_{i,c}=v_i^\top t_c
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2401.17270](https://ar5iv.labs.arxiv.org/html/2401.17270)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 8 张、Table 9 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1 : Speed-and-Accuracy Curve. We compare YOLO-World with recent open-vocabulary methods in terms of speed and accuracy. All models are evaluated on the LVIS minival and inference speeds are measured on one NVIDIA V100 w/o TensorRT. The size of the circle represents the model’s size.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2401.17270/assets/x1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看文本嵌入如何接入实时 YOLO head；用于评估开放词汇检测能否保持实时性。

### Figure 2: Figure 2 : Comparison with Detection Paradigms. (a) Traditional Object Detector : These object detectors can only detect objects within the fixed vocabulary pre-defined by the training datasets, e.g . , 80 categories of COCO dataset [ 26 ] . The fixed vocabulary limits the extension for open scenes. (b) Previous Open-Vocabulary Detectors: Previous methods tend to develop large and heavy detectors for open-vocabulary detection which intuitively have strong capacity. In addition, these detectors simultaneously encode images and texts as input for prediction, which is time-consuming for practical applications. (c) YOLO-World: We demonstrate the strong open-vocabulary performance of lightweight detectors, e.g . , YOLO detectors [ 42 , 20 ] , which is of great significance for real-world applications. Rather than using online vocabulary, we present a prompt-then-detect paradigm for efficient inference, in which the user generates a series of prompts according to the need and the prompts will be encoded into an offline vocabulary. Then it can be re-parameterized as the model weights for deployment and further acceleration.

![Figure 2](https://ar5iv.labs.arxiv.org/html/2401.17270/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看文本嵌入如何接入实时 YOLO head；用于评估开放词汇检测能否保持实时性。

### Figure 3: Figure 3 : Overall Architecture of YOLO-World. Compared to traditional YOLO detectors, YOLO-World as an open-vocabulary detector adopts text as input. The Text Encoder first encodes the input text input text embeddings. Then the Image Encoder encodes the input image into multi-scale image features and the proposed RepVL-PAN exploits the multi-level cross-modality fusion for both image and text features. Finally, YOLO-World predicts the regressed bounding boxes and the object embeddings for matching the categories or nouns that appeared in the input text.

![Figure 3](https://ar5iv.labs.arxiv.org/html/2401.17270/assets/x3.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看文本嵌入如何接入实时 YOLO head；用于评估开放词汇检测能否保持实时性。

### Figure 4: Figure 4 : Illustration of the RepVL-PAN. The proposed RepVL-PAN adopts the Text-guided CSPLayer (T-CSPLayer) for injecting language information into image features and the Image Pooling Attention (I-Pooling Attention) for enhancing image-aware text embeddings.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2401.17270/assets/x4.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看文本嵌入如何接入实时 YOLO head；用于评估开放词汇检测能否保持实时性。

### Figure 5: Figure 5 : Visualization Results on Zero-shot Inference on LVIS. We adopt the pre-trained YOLO-World-L and infer with the LVIS vocabulary (containing 1203 categories) on the COCO val2017 .

![Figure 5](https://ar5iv.labs.arxiv.org/html/2401.17270/assets/x5.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看文本嵌入如何接入实时 YOLO head；用于评估开放词汇检测能否保持实时性。

### Figure 6: Figure 6 : Visualization Results on User’s Vocabulary. We define the custom vocabulary for each input image and YOLO-World can detect the accurate regions according to the vocabulary. Images are obtained from COCO val2017 .

![Figure 6](https://ar5iv.labs.arxiv.org/html/2401.17270/assets/x6.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看文本嵌入如何接入实时 YOLO head；用于评估开放词汇检测能否保持实时性。

### Figure 7: Figure 7 : Visualization Results on Referring Object Detection. We explore the capability of the pre-trained YOLO-World to detect objects with descriptive noun phrases. Images are obtained from COCO val2017 .

![Figure 7](https://ar5iv.labs.arxiv.org/html/2401.17270/assets/x7.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看文本嵌入如何接入实时 YOLO head；用于评估开放词汇检测能否保持实时性。

### Figure 8: Figure 8 : Labeling Pipeline for Image-Text Data We first leverage the simple n-gram to extract object nouns from the captions. We adopt a pre-trained open-vocabulary detector to generate pseudo boxes given the object nouns, which forms the coarse region-text proposals. Then we use a pre-trained CLIP to rescore or relabel the boxes along with filtering.

![Figure 8](https://ar5iv.labs.arxiv.org/html/2401.17270/assets/x8.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看文本嵌入如何接入实时 YOLO head；用于评估开放词汇检测能否保持实时性。

### 关键表格

#### Table 1: YOLO-World Table 1

| Dataset | Type | Vocab. | Images | Anno. |
| --- | --- | --- | --- | --- |
| Objects365V1 [ 46 ] | Detection | 365 | 609k | 9,621k |
| GQA [ 17 ] | Grounding | - | 621k | 3,681k |
| Flickr [ 38 ] | Grounding | - | 149k | 641k |
| CC3M † † \dagger [ 47 ] | Image-Text | - | 246k | 821k |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: YOLO-World Table 2

| Method | Backbone | Params | Pre-trained Data | FPS | AP | AP r | AP c | AP f |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MDETR [ 21 ] | R-101 [ 15 ] | 169M | GoldG | - | 24.2 | 20.9 | 24.3 | 24.2 |
| GLIP-T [ 24 ] | Swin-T [ 32 ] | 232M | O365,GoldG | 0.12 | 24.9 | 17.7 | 19.5 | 31.0 |
| GLIP-T [ 24 ] | Swin-T [ 32 ] | 232M | O365,GoldG,Cap4M | 0.12 | 26.0 | 20.8 | 21.4 | 31.0 |
| GLIPv2-T [ 59 ] | Swin-T [ 32 ] | 232M | O365,GoldG | 0.12 | 26.9 | - | - | - |
| GLIPv2-T [ 59 ] | Swin-T [ 32 ] | 232M | O365,GoldG,Cap4M | 0.12 | 29.0 | - | - | - |
| Grounding DINO-T [ 30 ] | Swin-T [ 32 ] | 172M | O365,GoldG | 1.5 | 25.6 | 14.4 | 19.6 | 32.2 |
| Grounding DINO-T [ 30 ] | Swin-T [ 32 ] | 172M | O365,GoldG,Cap4M | 1.5 | 27.4 | 18.1 | 23.3 | 32.7 |
| DetCLIP-T [ 56 ] | Swin-T [ 32 ] | 155M | O365,GoldG | 2.3 | 34.4 | 26.9 | 33.9 | 36.3 |
| YOLO-World-S | YOLOv8-S | 13M (77M) | O365,GoldG | 74.1 (19.9) | 26.2 | 19.1 | 23.6 | 29.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: YOLO-World Table 3

| Pre-trained Data | AP | AP r | AP c | AP f |
| --- | --- | --- | --- | --- |
| O365 | 23.5 | 16.2 | 21.1 | 27.0 |
| O365,GQA | 31.9 | 22.5 | 29.9 | 35.4 |
| O365,GoldG | 32.5 | 22.3 | 30.6 | 36.0 |
| O365,GoldG,CC3M † | 33.0 | 23.6 | 32.0 | 35.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: YOLO-World Table 4

| GQA | T → → \rightarrow I | I → → \rightarrow T | AP | AP r | AP c | AP f |
| --- | --- | --- | --- | --- | --- | --- |
| ✗ | ✗ | ✗ | 22.4 | 14.5 | 20.1 | 26.0 |
| ✗ | ✓ | ✗ | 23.2 | 15.2 | 20.6 | 27.0 |
| ✗ | ✓ | ✓ | 23.5 | 16.2 | 21.1 | 27.0 |
| ✓ | ✗ | ✗ | 29.7 | 21.0 | 27.1 | 33.6 |
| ✓ | ✓ | ✓ | 31.9 | 22.5 | 29.9 | 35.4 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: YOLO-World Table 5

| Text Encoder | Frozen? | AP | AP r | AP c | AP f |
| --- | --- | --- | --- | --- | --- |
| BERT-base | Frozen | 14.6 | 3.4 | 10.7 | 20.0 |
| BERT-base | Fine-tune | 18.3 | 6.6 | 14.6 | 23.6 |
| CLIP-base | Frozen | 22.4 | 14.5 | 20.1 | 26.0 |
| CLIP-base | Fine-tune | 19.3 | 8.6 | 15.7 | 24.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: YOLO-World Table 6

| Method | Pre-train | AP | AP 50 | AP 75 | FPS |
| --- | --- | --- | --- | --- | --- |
| Training from scratch. |  |  |  |  |  |
| YOLOv6-S [ 23 ] | ✗ | 43.7 | 60.8 | 47.0 | 442 |
| YOLOv6-M [ 23 ] | ✗ | 48.4 | 65.7 | 52.7 | 277 |
| YOLOv6-L [ 23 ] | ✗ | 50.7 | 68.1 | 54.8 | 166 |
| YOLOv7-T [ 52 ] | ✗ | 37.5 | 55.8 | 40.2 | 404 |
| YOLOv7-L [ 52 ] | ✗ | 50.9 | 69.3 | 55.3 | 182 |
| YOLOv7-X [ 52 ] | ✗ | 52.6 | 70.6 | 57.3 | 131 |
| YOLOv8-S [ 20 ] | ✗ | 44.4 | 61.2 | 48.1 | 386 |
| YOLOv8-M [ 20 ] | ✗ | 50.5 | 67.3 | 55.0 | 238 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: YOLO-World Table 7

| Method | AP | AP r | AP c | AP f |
| --- | --- | --- | --- | --- |
| ViLD [ 13 ] | 27.8 | 16.7 | 26.5 | 34.2 |
| RegionCLIP [ 62 ] | 28.2 | 17.1 | - | - |
| Detic [ 63 ] | 26.8 | 17.8 | - | - |
| FVLM [ 22 ] | 24.2 | 18.6 | - | - |
| DetPro [ 8 ] | 28.4 | 20.8 | 27.8 | 32.4 |
| BARON [ 53 ] | 29.5 | 23.2 | 29.3 | 32.5 |
| YOLOv8-S | 19.4 | 7.4 | 17.4 | 27.0 |
| YOLOv8-M | 23.1 | 8.4 | 21.3 | 31.5 |
| YOLOv8-L | 26.9 | 10.2 | 25.4 | 35.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: YOLO-World Table 8

| Model | Fine-tune Data | Fine-tune Modules | AP | AP r | AP c | AP f | AP b | AP r b subscript superscript absent 𝑏 𝑟 {}^{b}_{r} |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO-World-M | COCO | Seg Head | 12.3 | 9.1 | 10.9 | 14.6 | 22.3 | 16.2 |
| YOLO-World-L | COCO | Seg Head | 16.2 | 12.4 | 15.0 | 19.2 | 25.3 | 18.0 |
| YOLO-World-M | LVIS-base | Seg Head | 16.7 | 12.6 | 14.6 | 20.8 | 22.3 | 16.2 |
| YOLO-World-L | LVIS-base | Seg Head | 19.1 | 14.2 | 17.2 | 23.5 | 25.3 | 18.0 |
| YOLO-World-M | LVIS-base | All | 25.9 | 13.4 | 24.9 | 32.6 | 32.6 | 15.8 |
| YOLO-World-L | LVIS-base | All | 28.7 | 15.0 | 28.3 | 35.2 | 36.2 | 17.4 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 9: YOLO-World Table 9

| Method | Pre-trained Data | Samples | AP | AP r | AP c | AP f |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO-World-S | O365 | 0.61M | 16.3 | 9.2 | 14.1 | 20.1 |
| YOLO-World-S | O365+GoldG | 1.38M | 24.2 | 16.4 | 21.7 | 27.8 |
| YOLO-World-S | O365+CC3M-245k | 0.85M | 16.5 | 10.8 | 14.8 | 19.1 |
| YOLO-World-S | O365+CC3M-520k | 1.13M | 19.2 | 10.7 | 17.4 | 22.4 |
| YOLO-World-S | O365+CC3M-750k | 1.36M | 18.2 | 11.2 | 16.0 | 21.1 |

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

- [[YOLOX]]
- [[GLIP]]
- [[Grounding DINO]]
- [[OWLv2]]
