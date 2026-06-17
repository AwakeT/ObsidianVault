---
title: "Feature Pyramid Networks for Object Detection"
method_name: "FPN"
authors: [Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie]
year: 2016
venue: CVPR
tags: [object-detection, feature-pyramid, multi-scale, cnn]
arxiv: https://arxiv.org/abs/1612.03144
created: 2026-06-11
image_source: online
---

# 论文笔记：Feature Pyramid Networks for Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[FPN]] |
| 作者 | [Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie] |
| 年份 | 2016 |
| 会议/来源 | CVPR |
| 链接 | [arXiv](https://arxiv.org/abs/1612.03144) / [PDF](https://arxiv.org/pdf/1612.03144.pdf) |
| 相关工作 | [[Faster R-CNN]]、[[RetinaNet]]、[[FCOS]]、[[SSD]] |

---

## 一句话总结

> FPN 将深层语义和浅层空间细节融合成多尺度特征金字塔，成为现代检测器的基础组件。

---

## 核心贡献

1. **范式贡献**：FPN 将深层语义和浅层空间细节融合成多尺度特征金字塔，成为现代检测器的基础组件。
2. **方法贡献**：用 top-down pathway 和 lateral connection 把深层语义传播到高分辨率层，构建 P2-P5 多尺度特征金字塔。
3. **工程/研究影响**：该工作成为后续 [[Faster R-CNN]]、[[RetinaNet]]、[[FCOS]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

用 top-down pathway 和 lateral connection 把深层语义传播到高分辨率层，构建 P2-P5 多尺度特征金字塔。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[Faster R-CNN]]、[[RetinaNet]]。
- 后续影响：[[FCOS]]、[[SSD]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
P_l = Conv_{3\times3}(C_l^{1\times1}+Up(P_{l+1}))
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1612.03144](https://ar5iv.labs.arxiv.org/html/1612.03144)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 4 张、Table 4 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: (a) Using an image pyramid to build a feature pyramid. Features are computed on each of the image scales independently, which is slow. (b) Recent detection systems have opted to use only single scale features for faster detection. (c) An alternative is to reuse the pyramidal feature hierarchy computed by a ConvNet as if it were a featurized image pyramid. (d) Our proposed Feature Pyramid Network (FPN) is fast like (b) and (c), but more accurate. In this figure, feature maps are indicate by blue outlines and thicker outlines denote semantically stronger features.

![Figure 1](https://ar5iv.labs.arxiv.org/html/1612.03144/assets/x1.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 2: Figure 2: Top: a top-down architecture with skip connections, where predictions are made on the finest level ( e.g . , [ 28 ] ). Bottom: our model that has a similar structure but leverages it as a feature pyramid , with predictions made independently at all levels.

![Figure 2](https://ar5iv.labs.arxiv.org/html/1612.03144/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 3: Figure 3: A building block illustrating the lateral connection and the top-down pathway, merged by addition.

![Figure 3](https://ar5iv.labs.arxiv.org/html/1612.03144/assets/x3.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 4: Figure 4: FPN for object segment proposals. The feature pyramid is constructed with identical structure as for object detection. We apply a small MLP on 5 × \times 5 windows to generate dense object segments with output dimension of 14 × \times 14. Shown in orange are the size of the image regions the mask corresponds to for each pyramid level (levels P 3 − 5 subscript 𝑃 3 5 P_{3-5} are shown here). Both the corresponding image region size (light orange) and canonical object size (dark orange) are shown. Half octaves are handled by an MLP on 7x7 windows ( 7 ≈ 5 ​ 2 7 5 2 7\approx 5\sqrt{2} ), not shown here. Details are in the appendix.

![Figure 4](https://ar5iv.labs.arxiv.org/html/1612.03144/assets/x4.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### 关键表格

#### Table 1: FPN Table 1

| Fast R-CNN | proposals | feature | head | lateral? | top-down? | AP@0.5 | AP | AP s | AP m | AP l |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) baseline on conv4 | RPN, { P k } subscript 𝑃 𝑘 \{P_{k}\} | C 4 subscript 𝐶 4 C_{4} | conv5 |  |  | 54.7 | 31.9 | 15.7 | 36.5 | 45.5 |
| (b) baseline on conv5 | RPN, { P k } subscript 𝑃 𝑘 \{P_{k}\} | C 5 subscript 𝐶 5 C_{5} | 2 fc |  |  | 52.9 | 28.8 | 11.9 | 32.4 | 43.4 |
| (c) FPN | RPN, { P k } subscript 𝑃 𝑘 \{P_{k}\} | { P k } subscript 𝑃 𝑘 \{P_{k}\} | 2 fc | ✓ | ✓ | 56.9 | 33.9 | 17.8 | 37.7 | 45.8 |
| Ablation experiments follow: |  |  |  |  |  |  |  |  |  |  |
| (d) bottom-up pyramid | RPN, { P k } subscript 𝑃 𝑘 \{P_{k}\} | { P k } subscript 𝑃 𝑘 \{P_{k}\} | 2 fc | ✓ |  | 44.9 | 24.9 | 10.9 | 24.4 | 38.5 |
| (e) top-down pyramid, w/o lateral | RPN, { P k } subscript 𝑃 𝑘 \{P_{k}\} | { P k } subscript 𝑃 𝑘 \{P_{k}\} | 2 fc |  | ✓ | 54.0 | 31.3 | 13.3 | 35.2 | 45.3 |
| (f) only finest level | RPN, { P k } subscript 𝑃 𝑘 \{P_{k}\} | P 2 subscript 𝑃 2 P_{2} | 2 fc | ✓ | ✓ | 56.3 | 33.4 | 17.3 | 37.3 | 45.6 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: FPN Table 2

| Faster R-CNN | proposals | feature | head | lateral? | top-down? | AP@0.5 | AP | AP s | AP m | AP l |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (*) baseline from He et al . [ 16 ] † | RPN, C 4 subscript 𝐶 4 C_{4} | C 4 subscript 𝐶 4 C_{4} | conv5 |  |  | 47.3 | 26.3 | - | - | - |
| (a) baseline on conv4 | RPN, C 4 subscript 𝐶 4 C_{4} | C 4 subscript 𝐶 4 C_{4} | conv5 |  |  | 53.1 | 31.6 | 13.2 | 35.6 | 47.1 |
| (b) baseline on conv5 | RPN, C 5 subscript 𝐶 5 C_{5} | C 5 subscript 𝐶 5 C_{5} | 2 fc |  |  | 51.7 | 28.0 | 9.6 | 31.9 | 43.1 |
| (c) FPN | RPN, { P k } subscript 𝑃 𝑘 \{P_{k}\} | { P k } subscript 𝑃 𝑘 \{P_{k}\} | 2 fc | ✓ | ✓ | 56.9 | 33.9 | 17.8 | 37.7 | 45.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: FPN Table 3

|  |  |  | image | test-dev | test-std |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| method | backbone | competition | pyramid | AP @.5 | AP | AP s | AP m | AP l | AP @.5 | AP | AP s | AP m | AP l |
| ours, Faster R-CNN on FPN | ResNet-101 | - |  | 59.1 | 36.2 | 18.2 | 39.0 | 48.2 | 58.5 | 35.8 | 17.5 | 38.7 | 47.8 |
| Competition-winning single-model results follow: |  |  |  |  |  |  |  |  |  |  |  |  |  |
| G-RMI † | Inception-ResNet | 2016 |  | - | 34.7 | - | - | - | - | - | - | - | - |
| AttractioNet ‡ [ 10 ] | VGG16 + Wide ResNet § | 2016 | ✓ | 53.4 | 35.7 | 15.6 | 38.0 | 52.7 | 52.9 | 35.3 | 14.7 | 37.6 | 51.9 |
| Faster R-CNN +++ [ 16 ] | ResNet-101 | 2015 | ✓ | 55.7 | 34.9 | 15.6 | 38.7 | 50.9 | - | - | - | - | - |
| Multipath [ 40 ] (on minival ) | VGG-16 | 2015 |  | 49.6 | 31.5 | - | - | - | - | - | - | - | - |
| ION ‡ [ 2 ] | VGG-16 | 2015 |  | 53.4 | 31.2 | 12.8 | 32.9 | 45.2 | 52.9 | 30.7 | 11.8 | 32.8 | 44.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: FPN Table 4

|  | ResNet-50 | ResNet-101 |  |  |
| --- | --- | --- | --- | --- |
| share features? | AP @0.5 | AP | AP @0.5 | AP |
| no | 56.9 | 33.9 | 58.0 | 35.0 |
| yes | 57.2 | 34.3 | 58.2 | 35.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

Feature pyramids are a basic component in recognition systems for detecting objects at different scales. But recent deep learning object detectors have avoided pyramid representations, in part because they are compute and memory intensive. In this paper, we exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. A top-down architecture with lateral connections is developed for building high-level semantic featur

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

需要按 P2/P3/P4/P5 分析 FCOS 的 FP_loc 和小物体召回，确认 adapter 金字塔是否把 DINOv2 token 组织成了检测友好的尺度层级。

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
- [[RetinaNet]]
- [[FCOS]]
- [[SSD]]
