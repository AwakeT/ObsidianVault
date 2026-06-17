---
title: "FCOS: Fully Convolutional One-Stage Object Detection"
method_name: "FCOS"
authors: [Zhi Tian, Chunhua Shen, Hao Chen, Tong He]
year: 2019
venue: ICCV
tags: [object-detection, anchor-free, dense-detection, fcos]
arxiv: https://arxiv.org/abs/1904.01355
created: 2026-06-11
image_source: online
---

# 论文笔记：FCOS: Fully Convolutional One-Stage Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[FCOS]] |
| 作者 | [Zhi Tian, Chunhua Shen, Hao Chen, Tong He] |
| 年份 | 2019 |
| 会议/来源 | ICCV |
| 链接 | [arXiv](https://arxiv.org/abs/1904.01355) / [PDF](https://arxiv.org/pdf/1904.01355.pdf) |
| 相关工作 | [[RetinaNet]]、[[ATSS]]、[[GFL]]、[[TOOD]]、[[YOLOX]] |

---

## 一句话总结

> FCOS 将目标检测转化为每个特征位置预测类别、四边距离和 centerness 的 anchor-free dense prediction。

---

## 核心贡献

1. **范式贡献**：FCOS 将目标检测转化为每个特征位置预测类别、四边距离和 centerness 的 anchor-free dense prediction。
2. **方法贡献**：每个特征位置直接预测类别、四边距离 l/t/r/b 和 centerness，去掉 anchor 设计。
3. **工程/研究影响**：该工作成为后续 [[RetinaNet]]、[[ATSS]]、[[GFL]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

每个特征位置直接预测类别、四边距离 l/t/r/b 和 centerness，去掉 anchor 设计。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[RetinaNet]]、[[ATSS]]。
- 后续影响：[[GFL]]、[[TOOD]]、[[YOLOX]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
centerness=\sqrt{\frac{\min(l,r)}{\max(l,r)}\frac{\min(t,b)}{\max(t,b)}}
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1904.01355](https://ar5iv.labs.arxiv.org/html/1904.01355)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 8 张、Table 7 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: As shown in the left image, FCOS works by predicting a 4D vector ( l , t , r , b ) 𝑙 𝑡 𝑟 𝑏 (l,t,r,b) encoding the location of a bounding box at each foreground pixel (supervised by ground-truth bounding box information during training). The right plot shows that when a location residing in multiple bounding boxes, it can be ambiguous in terms of which bounding box this location should regress.

![Figure 1](https://ar5iv.labs.arxiv.org/html/1904.01355/assets/x1.png)

**图表解读**：重点看 centerness 分支、l/t/r/b 四边距离预测和正样本区域定义；这些图直接对应当前 FCOS++ 的 objectness/assignment 改造。

### Figure 2: Figure 2: The network architecture of FCOS, where C3, C4, and C5 denote the feature maps of the backbone network and P3 to P7 are the feature levels used for the final prediction. H × W 𝐻 𝑊 H\times W is the height and width of feature maps. ‘/ s 𝑠 s ’ ( s = 8 , 16 , … , 128 𝑠 8 16 … 128 s=8,16,...,128 ) is the down-sampling ratio of the feature maps at the level to the input image. As an example, all the numbers are computed with an 800 × 1024 800 1024 800\times 1024 input.

![Figure 2](https://ar5iv.labs.arxiv.org/html/1904.01355/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 centerness 分支、l/t/r/b 四边距离预测和正样本区域定义；这些图直接对应当前 FCOS++ 的 objectness/assignment 改造。

### Figure 3: Figure 3: Center-ness. Red, blue, and other colors denote 1, 0 and the values between them, respectively. Center-ness is computed by Eq. ( 3 ) and decays from 1 to 0 as the location deviates from the center of the object. When testing, the center-ness predicted by the network is multiplied with the classification score thus can down-weight the low-quality bounding boxes predicted by a location far from the center of an object.

![Figure 3](https://ar5iv.labs.arxiv.org/html/1904.01355/assets/x3.png)

**图表解读**：重点看 centerness 分支、l/t/r/b 四边距离预测和正样本区域定义；这些图直接对应当前 FCOS++ 的 objectness/assignment 改造。

### Figure 4: Figure 4: Class-agnostic precision-recall curves at IOU = 0.50 absent 0.50 =0.50 .

![Figure 4](https://ar5iv.labs.arxiv.org/html/1904.01355/assets/x4.png)

**图表解读**：重点看 centerness 分支、l/t/r/b 四边距离预测和正样本区域定义；这些图直接对应当前 FCOS++ 的 objectness/assignment 改造。

### Figure 5: Figure 5: Class-agnostic precision-recall curves at IOU = 0.75 absent 0.75 =0.75 .

![Figure 5](https://ar5iv.labs.arxiv.org/html/1904.01355/assets/x5.png)

**图表解读**：重点看 centerness 分支、l/t/r/b 四边距离预测和正样本区域定义；这些图直接对应当前 FCOS++ 的 objectness/assignment 改造。

### Figure 6: Figure 6: Class-agnostic precision-recall curves at IOU = 0.90 absent 0.90 =0.90 .

![Figure 6](https://ar5iv.labs.arxiv.org/html/1904.01355/assets/x6.png)

**图表解读**：重点看 centerness 分支、l/t/r/b 四边距离预测和正样本区域定义；这些图直接对应当前 FCOS++ 的 objectness/assignment 改造。

### Figure 7: Figure 7: Without (left) or with (right) the proposed center-ness. A point in the figure denotes a detected bounding box. The dashed line is the line y = x 𝑦 𝑥 y=x . As shown in the figure (right), after multiplying the classification scores with the center-ness scores, the low-quality boxes (under the line y = x 𝑦 𝑥 y=x ) are pushed to the left side of the plot. It suggests that the scores of these boxes are reduced substantially.

![Figure 7](https://ar5iv.labs.arxiv.org/html/1904.01355/assets/x7.png)

**图表解读**：重点看 centerness 分支、l/t/r/b 四边距离预测和正样本区域定义；这些图直接对应当前 FCOS++ 的 objectness/assignment 改造。

### Figure 8: Figure 8: Some detection results on 𝚖𝚒𝚗𝚒𝚟𝚊𝚕 𝚖𝚒𝚗𝚒𝚟𝚊𝚕 \tt minival split. ResNet-50 is used as the backbone. As shown in the figure, FCOS works well with a wide range of objects including crowded, occluded, highly overlapped, extremely small and very large objects.

![Figure 8](https://ar5iv.labs.arxiv.org/html/1904.01355/assets/x8.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 centerness 分支、l/t/r/b 四边距离预测和正样本区域定义；这些图直接对应当前 FCOS++ 的 objectness/assignment 改造。

### 关键表格

#### Table 1: FCOS Table 1

| Method | w/ FPN | Low-quality matches | BPR (%) |
| --- | --- | --- | --- |
| RetinaNet | ✓ | None | 86.82 |
| RetinaNet | ✓ | ≥ 0.4 absent 0.4 \geq 0.4 | 90.92 |
| RetinaNet | ✓ | All | 99.23 |
| FCOS |  | - | 95.55 |
| FCOS | ✓ | - | 98.40 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: FCOS Table 2

| Method | C 5 subscript 𝐶 5 C_{5} / P 5 subscript 𝑃 5 P_{5} | w/ GN | nms thr. | AP | AP 50 | AP 75 | AP S | AP M | AP L | AR 1 | AR 10 | AR 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RetinaNet | C 5 subscript 𝐶 5 C_{5} |  | .50 | 35.9 | 56.0 | 38.2 | 20.0 | 39.8 | 47.4 | 31.0 | 49.4 | 52.5 |
| FCOS | C 5 subscript 𝐶 5 C_{5} |  | .50 | 36.3 | 54.8 | 38.7 | 20.5 | 39.8 | 47.8 | 31.5 | 50.6 | 53.5 |
| FCOS | P 5 subscript 𝑃 5 P_{5} |  | .50 | 36.4 | 54.9 | 38.8 | 19.7 | 39.7 | 48.8 | 31.4 | 50.6 | 53.4 |
| FCOS | P 5 subscript 𝑃 5 P_{5} |  | .60 | 36.5 | 54.5 | 39.2 | 19.8 | 40.0 | 48.9 | 31.3 | 51.2 | 54.5 |
| FCOS | P 5 subscript 𝑃 5 P_{5} | ✓ | .60 | 37.1 | 55.9 | 39.8 | 21.3 | 41.0 | 47.8 | 31.4 | 51.4 | 54.9 |
| Improvements |  |  |  |  |  |  |  |  |  |  |  |  |
| + ctr. on reg. | P 5 subscript 𝑃 5 P_{5} | ✓ | .60 | 37.4 | 56.1 | 40.3 | 21.8 | 41.2 | 48.8 | 31.5 | 51.7 | 55.2 |
| + ctr. sampling [ 1 ] | P 5 subscript 𝑃 5 P_{5} | ✓ | .60 | 38.1 | 56.7 | 41.4 | 22.6 | 41.6 | 50.4 | 32.1 | 52.8 | 56.3 |
| + GIoU [ 1 ] | P 5 subscript 𝑃 5 P_{5} | ✓ | .60 | 38.3 | 57.1 | 41.0 | 21.9 | 42.4 | 49.5 | 32.0 | 52.9 | 56.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: FCOS Table 3

|  | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- |
| None | 33.5 | 52.6 | 35.2 | 20.8 | 38.5 | 42.6 |
| center-ness † | 33.5 | 52.4 | 35.1 | 20.8 | 37.8 | 42.8 |
| center-ness | 37.1 | 55.9 | 39.8 | 21.3 | 41.0 | 47.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: FCOS Table 4

| Method | Backbone | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Two-stage methods: |  |  |  |  |  |  |  |
| Faster R-CNN w/ FPN [ 14 ] | ResNet-101-FPN | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2 |
| Faster R-CNN by G-RMI [ 11 ] | Inception-ResNet-v2 [ 27 ] | 34.7 | 55.5 | 36.7 | 13.5 | 38.1 | 52.0 |
| Faster R-CNN w/ TDM [ 25 ] | Inception-ResNet-v2-TDM | 36.8 | 57.7 | 39.2 | 16.2 | 39.8 | 52.1 |
| One-stage methods: |  |  |  |  |  |  |  |
| YOLOv2 [ 22 ] | DarkNet-19 [ 22 ] | 21.6 | 44.0 | 19.2 | 5.0 | 22.4 | 35.5 |
| SSD513 [ 18 ] | ResNet-101-SSD | 31.2 | 50.4 | 33.3 | 10.2 | 34.5 | 49.8 |
| DSSD513 [ 5 ] | ResNet-101-DSSD | 33.2 | 53.3 | 35.2 | 13.0 | 35.4 | 51.1 |
| RetinaNet [ 15 ] | ResNet-101-FPN | 39.1 | 59.1 | 42.3 | 21.8 | 42.7 | 50.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: FCOS Table 5

| Method | # samples | AR 100 | AR 1k |
| --- | --- | --- | --- |
| RPN w/ FPN & GN (ReImpl.) | ∼ similar-to \sim 200K | 44.7 | 56.9 |
| FCOS w/ GN w/o center-ness | ∼ similar-to \sim 66K | 48.0 | 59.3 |
| FCOS w/ GN | ∼ similar-to \sim 66K | 52.8 | 60.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: FCOS Table 6

| Method | AP | AP 50 | AP 75 | AP 90 |
| --- | --- | --- | --- | --- |
| Orginal RetinaNet [ 15 ] | 39.5 | 63.6 | 41.8 | 10.6 |
| RetinaNet w/ GN [ 29 ] | 40.0 | 64.5 | 42.2 | 10.4 |
| FCOS | 40.5 | 64.7 | 42.6 | 13.1 |
|  |  | +0.2 | +0.4 | +2.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: FCOS Table 7

| Method | C 5 / P 5 subscript 𝐶 5 subscript 𝑃 5 C_{5}/P_{5} | GN | Scalar | IoU | AP |
| --- | --- | --- | --- | --- | --- |
| RetinaNet (#A=1) | C 5 subscript 𝐶 5 C_{5} |  |  |  | 32.5 |
| RetinaNet (#A=9) | C 5 subscript 𝐶 5 C_{5} |  |  |  | 35.7 |
| FCOS (pure) | C 5 subscript 𝐶 5 C_{5} |  |  |  | 35.7 |
| FCOS | P 5 subscript 𝑃 5 P_{5} |  |  |  | 35.8 |
| FCOS | P 5 subscript 𝑃 5 P_{5} | ✓ |  |  | 36.3 |
| FCOS | P 5 subscript 𝑃 5 P_{5} | ✓ | ✓ |  | 36.4 |
| FCOS | P 5 subscript 𝑃 5 P_{5} | ✓ | ✓ | ✓ | 36.6 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation. Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3, and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor box free, as well as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxe

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

这是当前生产头基础，下一步不是换掉 FCOS，而是升级为 FCOS++：objectness、ATSS/TAL、hard negative 和 ROI verifier。

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
- [[ATSS]]
- [[GFL]]
- [[TOOD]]
- [[YOLOX]]
