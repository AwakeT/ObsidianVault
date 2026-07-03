---
title: "Le-DETR: Revisiting Real-Time Detection Transformer with Efficient Encoder Design"
method_name: "Le-DETR"
authors: [Jiannan Huang, Aditya Kane, Fengzhe Zhou, Yunchao Wei, Humphrey Shi]
year: 2026
venue: arXiv
tags: [object-detection, detr, real-time-detection, efficient-encoder]
arxiv: https://arxiv.org/abs/2602.21010
created: 2026-06-11
image_source: online
---

# 论文笔记：Le-DETR: Revisiting Real-Time Detection Transformer with Efficient Encoder Design

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[Le-DETR]] |
| 作者 | [Jiannan Huang, Aditya Kane, Fengzhe Zhou, Yunchao Wei, Humphrey Shi] |
| 年份 | 2026 |
| 会议/来源 | arXiv |
| 链接 | 无 arXiv 链接 / 经典论文 |
| 相关工作 | [[RT-DETR]]、[[D-FINE]]、[[RF-DETR]] |

---

## 一句话总结

> Le-DETR 重新设计实时 DETR 的高效 encoder，强调低预训练成本下也能获得强实时检测性能。

---

## 核心贡献

1. **范式贡献**：Le-DETR 重新设计实时 DETR 的高效 encoder，强调低预训练成本下也能获得强实时检测性能。
2. **方法贡献**：重新设计实时 DETR 的高效 encoder，强调低预训练成本和部署友好性。
3. **工程/研究影响**：该工作成为后续 [[RT-DETR]]、[[D-FINE]]、[[RF-DETR]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

重新设计实时 DETR 的高效 encoder，强调低预训练成本和部署友好性。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[RT-DETR]]、[[D-FINE]]。
- 后续影响：[[RF-DETR]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\text{Latency}=T_{backbone}+T_{encoder}+T_{decoder}
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2602.21010](https://ar5iv.labs.arxiv.org/html/2602.21010)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 7 张、Table 5 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 2 : Left: Comparison of Le-DETR and Real-Time DETR series in training overheads, While saving lots of training overheads offered by efficient encoder design, our model outperforms the previous SOTA model, D-FINE and DEIM-D-FINE. Right: Unlike previous works such as DEIM and D-FINE mainly focus on decoder architecture and training objects, we focus on efficient encoder design.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2602.21010/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 2: Figure 3 : The proposed Le-DETR is structured in three distinct stages: backbone, encoder, and decoder. Each first block of stages in this backbone serves as the downsampling block. In this figure, we also illustrate the encoder, which incorporates both the NAIFI and Feature Fusion components.

![Figure 2](https://ar5iv.labs.arxiv.org/html/2602.21010/assets/x3.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 3: Figure 4 : Overview of EfficientNATBlock and Neighborhood Attention-based Improved Feature Inference(NAIFI).

![Figure 3](https://ar5iv.labs.arxiv.org/html/2602.21010/assets/x4.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 4: Table 5 : We conducted experiments to explore different patterns on the backbone. As described in Section 3.4 , we designed our backbone scale study using three distinct patterns. These are referred to as P A , P B , and P C , corresponding to balanced distribution, late-stage heavy distribution and an early-stage heavy distribution patterns, respectively. The results demonstrate that, for the L-scale, P A yields the best performance, whereas P C for the X-scale. The images clearly show the preference of different patterns at different scales.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2602.21010/assets/x5.png)

**图表解读**：这是消融/分析图，重点看每个模块独立贡献，避免把数据增强、backbone 或训练轮数带来的收益误认为核心方法收益。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 5: Figure 5 : Visualization of applying Le-DETR-L into hard cases of object detection.

![Figure 5](https://ar5iv.labs.arxiv.org/html/2602.21010/assets/x6.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 6: Figure 6 : Visualization of applying Le-DETR-X into hard cases of object detection.

![Figure 6](https://ar5iv.labs.arxiv.org/html/2602.21010/assets/x7.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 7: Figure 1 1

![Figure 7](https://ar5iv.labs.arxiv.org/html/2602.21010/assets/x1.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### 关键表格

#### Table 1: Le-DETR Table 1

| Model | KD & ATI | Variant | Pretraining Image Nums | Latency(ms) | AP val |
| --- | --- | --- | --- | --- | --- |
| RT-DETRv2-L | ✓ | ✓ | 5M | 5.46 | 53.4 |
| ✗ | ✓ | 1M | 5.46 | 51.6 ( ↓ \downarrow 1.8 ) |  |
| ✗ | ✗ | 1M | 4.91 | 51.6 ( ↓ \downarrow 1.8 ) |  |
| Le-DETR-L | ✗ | ✗ | 1M | 5.01 | 54.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: Le-DETR Table 2

| Models | Params (M) | GFLOPs | Latency(ms) | AP val | AP 50 v ​ a ​ l {}^{val}_{50} | AP 75 v ​ a ​ l {}^{val}_{75} | AP S v ​ a ​ l {}^{val}_{S} | AP M v ​ a ​ l {}^{val}_{M} | AP L v ​ a ​ l {}^{val}_{L} |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO Series Models |  |  |  |  |  |  |  |  |  |
| YOLOv5-M [ 36 ] | 25.1 | 64.2 | 3.57 | 49.1 | 66.0 | 53.8 | 31.2 | 54.2 | 65.4 |
| YOLOv5-L [ 36 ] | 46.5 | 109.1 | 5.57 | 49.0 | 67.3 | - | - | - | - |
| YOLOv5-X [ 36 ] | 86.7 | 205.7 | 8.55 | 50.7 | 68.9 | - | - | - | - |
| YOLOv8-M [ 37 ] | 25.9 | 78.9 | 4.06 | 50.3 | 67.3 | 54.8 | 32.3 | 55.9 | 66.5 |
| YOLOv8-L [ 37 ] | 43.7 | 165.2 | 6.52 | 52.9 | 69.8 | 57.7 | 35.5 | 58.5 | 69.8 |
| YOLOv8-X [ 37 ] | 68.2 | 257.8 | 8.51 | 53.9 | 71.1 | 58.9 | 36.0 | 59.4 | 70.9 |
| YOLOv9-M [ 43 ] | 20.1 | 76.8 | 4.78 | 51.5 | 68.4 | 54.0 | 33.8 | 57.2 | 67.3 |
| YOLOv9-C [ 43 ] | 25.5 | 102.8 | 5.06 | 52.9 | 69.8 | 57.7 | 35.6 | 58.2 | 69.0 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: Le-DETR Table 3

| Models | Params (M) | GFLOPs | Latency(ms) | AP val | AP 50 v ​ a ​ l {}^{val}_{50} | AP 75 v ​ a ​ l {}^{val}_{75} | AP S v ​ a ​ l {}^{val}_{S} | AP M v ​ a ​ l {}^{val}_{M} | AP L v ​ a ​ l {}^{val}_{L} |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Le-DETR w/ ResNet50_vd_ssld | 43.8 | 128.3 | 5.80 | 53.6 | 71.4 | 58.0 | 35.5 | 58.3 | 70.9 |
| Le-DETR w/ EfficientViT | 64.7 | 135.7 | 6.60 | 53.8 | 71.5 | 58.6 | 36.8 | 58.1 | 71.3 |
| Le-DETR w/o NAIFI | 41.5 | 124.3 | 5.18 | 54.1 | 71.2 | 58.9 | 36.3 | 58.6 | 70.4 |
| Le-DETR (Ours Full Method) | 41.5 | 124.3 | 5.01 | 54.3 | 71.7 | 58.9 | 37.1 | 58.7 | 71.4 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: Le-DETR Table 4

| Model Name | Params(M) | GFLOPs | Latency(ms) | AP val | AP 50 v ​ a ​ l {}^{val}_{50} |
| --- | --- | --- | --- | --- | --- |
| Le-DETR-M 4 | 31.4 | 114.1 | 4.19 | 52.5 | 69.3 |
| Le-DETR-M 5 | 32.7 | 115.0 | 4.45 | 52.9 | 70.0 |
| Le-DETR-M 6 | 34.0 | 115.6 | 4.73 | 52.9 | 70.0 |
| Le-DETR-L 4 | 38.8 | 122.6 | 4.53 | 53.7 | 70.9 |
| Le-DETR-L 5 | 40.1 | 123.5 | 4.79 | 54.2 | 71.6 |
| Le-DETR-L 6 | 41.5 | 124.3 | 5.01 | 54.3 | 71.7 |
| Le-DETR-X 4 | 42.3 | 195.2 | 6.10 | 54.6 | 71.8 |
| Le-DETR-X 5 | 43.6 | 196.0 | 6.42 | 54.9 | 72.3 |
| Le-DETR-X 6 | 44.9 | 196.9 | 6.68 | 55.1 | 72.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: Le-DETR Table 5

| COCO Training Settings | Le-DETR-M | Le-DETR-L | Le-DETR-X |
| --- | --- | --- | --- |
| Backbone | EfficientNAT-M | EfficientNAT-L | EfficientNAT-X |
| Backbone Freezing Layer | (0, 1) | (0, 1) | (0, 1) |
| Embedding Dimension | 256 | 256 | 256 |
| Feedforward Dimension | 1024 | 1024 | 1024 |
| Encoder Layer Number | 1 | 1 | 1 |
| NAAIFI Kernel Size | 63 | 63 | 63 |
| Decoder Hidden Dimension | 256 | 256 | 256 |
| Training Decoder Layer Number | 6 | 6 | 6 |
| Inference Decoder Layer Number | 4 | 6 | 6 |

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

- [[RT-DETR]]
- [[D-FINE]]
- [[RF-DETR]]
