---
title: "SSD: Single Shot MultiBox Detector"
method_name: "SSD"
authors: [Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander Berg]
year: 2015
venue: ECCV
tags: [object-detection, one-stage-detection, multi-scale, real-time-detection]
arxiv: https://arxiv.org/abs/1512.02325
created: 2026-06-11
image_source: online
---

# 论文笔记：SSD: Single Shot MultiBox Detector

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[SSD]] |
| 作者 | [Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander Berg] |
| 年份 | 2015 |
| 会议/来源 | ECCV |
| 链接 | [arXiv](https://arxiv.org/abs/1512.02325) / [PDF](https://arxiv.org/pdf/1512.02325.pdf) |
| 相关工作 | [[YOLOv1]]、[[FPN]]、[[RetinaNet]] |

---

## 一句话总结

> SSD 在多个尺度 feature map 上直接预测 default boxes，是早期高效一阶段检测器代表。

---

## 核心贡献

1. **范式贡献**：SSD 在多个尺度 feature map 上直接预测 default boxes，是早期高效一阶段检测器代表。
2. **方法贡献**：在多个 feature map 的每个位置设置 default boxes，直接预测类别分数和 box offsets，兼顾不同尺度目标。
3. **工程/研究影响**：该工作成为后续 [[YOLOv1]]、[[FPN]]、[[RetinaNet]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

在多个 feature map 的每个位置设置 default boxes，直接预测类别分数和 box offsets，兼顾不同尺度目标。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[YOLOv1]]、[[FPN]]。
- 后续影响：[[RetinaNet]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
b = d + \Delta(d)
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1512.02325](https://ar5iv.labs.arxiv.org/html/1512.02325)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 49 张、Table 6 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: SSD framework. (a) SSD only needs an input image and ground truth boxes for each object during training. In a convolutional fashion, we evaluate a small set (e.g. 4) of default boxes of different aspect ratios at each location in several feature maps with different scales (e.g. 8 × 8 8 8 8\times 8 and 4 × 4 4 4 4\times 4 in (b) and (c)). For each default box, we predict both the shape offsets and the confidences for all object categories ( ( c 1 , c 2 , ⋯ , c p ) subscript 𝑐 1 subscript 𝑐 2 ⋯ subscript 𝑐 𝑝 (c_{1},c_{2},\cdots,c_{p}) ). At training time, we first match these default boxes to the ground truth boxes. For example, we have matched two default boxes with the cat and one with the dog, which are treated as positives and the rest as negatives. The model loss is a weighted sum between localization loss (e.g. Smooth L1 [ 6 ] ) and confidence loss (e.g. Softmax).

![Figure 1](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 2: Figure 2: A comparison between two single shot detection models: SSD and YOLO [ 5 ] . Our SSD model adds several feature layers to the end of a base network, which predict the offsets to default boxes of different scales and aspect ratios and their associated confidences. SSD with a 300 × 300 300 300 300\times 300 input size significantly outperforms its 448 × 448 448 448 448\times 448 YOLO counterpart in accuracy on VOC2007 test while also improving the speed.

![Figure 2](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 3: Figure 3: Visualization of performance for SSD512 on animals, vehicles, and furniture from VOC2007 test . The top row shows the cumulative fraction of detections that are correct (Cor) or false positive due to poor localization (Loc), confusion with similar categories (Sim), with others (Oth), or with background (BG). The solid red line reflects the change of recall with âstrongâ criteria (0.5 jaccard overlap) as the num- ber of detections increases. The dashed red line is using the âweakâ criteria (0.1 jaccard overlap). The bottom row shows the distribution of top-ranked false positive types.

![Figure 3](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x3.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 4: Figure 3: Visualization of performance for SSD512 on animals, vehicles, and furniture from VOC2007 test . The top row shows the cumulative fraction of detections that are correct (Cor) or false positive due to poor localization (Loc), confusion with similar categories (Sim), with others (Oth), or with background (BG). The solid red line reflects the change of recall with âstrongâ criteria (0.5 jaccard overlap) as the num- ber of detections increases. The dashed red line is using the âweakâ criteria (0.1 jaccard overlap). The bottom row shows the distribution of top-ranked false positive types.

![Figure 4](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x4.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 5: Figure 3: Visualization of performance for SSD512 on animals, vehicles, and furniture from VOC2007 test . The top row shows the cumulative fraction of detections that are correct (Cor) or false positive due to poor localization (Loc), confusion with similar categories (Sim), with others (Oth), or with background (BG). The solid red line reflects the change of recall with âstrongâ criteria (0.5 jaccard overlap) as the num- ber of detections increases. The dashed red line is using the âweakâ criteria (0.1 jaccard overlap). The bottom row shows the distribution of top-ranked false positive types.

![Figure 5](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x5.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 6: Figure 3: Visualization of performance for SSD512 on animals, vehicles, and furniture from VOC2007 test . The top row shows the cumulative fraction of detections that are correct (Cor) or false positive due to poor localization (Loc), confusion with similar categories (Sim), with others (Oth), or with background (BG). The solid red line reflects the change of recall with âstrongâ criteria (0.5 jaccard overlap) as the num- ber of detections increases. The dashed red line is using the âweakâ criteria (0.1 jaccard overlap). The bottom row shows the distribution of top-ranked false positive types.

![Figure 6](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x6.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 7: Figure 3: Visualization of performance for SSD512 on animals, vehicles, and furniture from VOC2007 test . The top row shows the cumulative fraction of detections that are correct (Cor) or false positive due to poor localization (Loc), confusion with similar categories (Sim), with others (Oth), or with background (BG). The solid red line reflects the change of recall with âstrongâ criteria (0.5 jaccard overlap) as the num- ber of detections increases. The dashed red line is using the âweakâ criteria (0.1 jaccard overlap). The bottom row shows the distribution of top-ranked false positive types.

![Figure 7](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x7.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 8: Figure 3: Visualization of performance for SSD512 on animals, vehicles, and furniture from VOC2007 test . The top row shows the cumulative fraction of detections that are correct (Cor) or false positive due to poor localization (Loc), confusion with similar categories (Sim), with others (Oth), or with background (BG). The solid red line reflects the change of recall with âstrongâ criteria (0.5 jaccard overlap) as the num- ber of detections increases. The dashed red line is using the âweakâ criteria (0.1 jaccard overlap). The bottom row shows the distribution of top-ranked false positive types.

![Figure 8](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x8.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 9: Figure 4: Sensitivity and impact of different object characteristics on VOC2007 test set using [ 21 ] . The plot on the left shows the effects of BBox Area per category, and the right plot shows the effect of Aspect Ratio. Key: BBox Area: XS=extra-small; S=small; M=medium; L=large; XL =extra-large. Aspect Ratio: XT=extra-tall/narrow; T=tall; M=medium; W=wide; XW =extra-wide.

![Figure 9](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x9.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 10: Figure 4: Sensitivity and impact of different object characteristics on VOC2007 test set using [ 21 ] . The plot on the left shows the effects of BBox Area per category, and the right plot shows the effect of Aspect Ratio. Key: BBox Area: XS=extra-small; S=small; M=medium; L=large; XL =extra-large. Aspect Ratio: XT=extra-tall/narrow; T=tall; M=medium; W=wide; XW =extra-wide.

![Figure 10](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x10.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 11: Figure 4: Sensitivity and impact of different object characteristics on VOC2007 test set using [ 21 ] . The plot on the left shows the effects of BBox Area per category, and the right plot shows the effect of Aspect Ratio. Key: BBox Area: XS=extra-small; S=small; M=medium; L=large; XL =extra-large. Aspect Ratio: XT=extra-tall/narrow; T=tall; M=medium; W=wide; XW =extra-wide.

![Figure 11](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x11.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 12: Figure 4: Sensitivity and impact of different object characteristics on VOC2007 test set using [ 21 ] . The plot on the left shows the effects of BBox Area per category, and the right plot shows the effect of Aspect Ratio. Key: BBox Area: XS=extra-small; S=small; M=medium; L=large; XL =extra-large. Aspect Ratio: XT=extra-tall/narrow; T=tall; M=medium; W=wide; XW =extra-wide.

![Figure 12](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x12.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 13: [Uncaptioned image]

![Figure 13](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x13.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 14: [Uncaptioned image]

![Figure 14](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x14.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 15: [Uncaptioned image]

![Figure 15](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x15.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 16: [Uncaptioned image]

![Figure 16](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x16.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 17: [Uncaptioned image]

![Figure 17](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x47.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 18: SSD Figure 18

![Figure 18](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x17.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 19: SSD Figure 19

![Figure 19](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x18.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 20: SSD Figure 20

![Figure 20](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x19.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 21: SSD Figure 21

![Figure 21](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x20.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 22: SSD Figure 22

![Figure 22](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x21.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 23: SSD Figure 23

![Figure 23](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x22.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 24: SSD Figure 24

![Figure 24](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x23.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 25: SSD Figure 25

![Figure 25](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x24.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 26: SSD Figure 26

![Figure 26](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x25.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 27: SSD Figure 27

![Figure 27](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x26.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 28: SSD Figure 28

![Figure 28](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x27.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 29: SSD Figure 29

![Figure 29](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x28.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 30: SSD Figure 30

![Figure 30](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x29.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 31: SSD Figure 31

![Figure 31](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x30.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 32: SSD Figure 32

![Figure 32](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x31.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 33: SSD Figure 33

![Figure 33](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x32.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 34: SSD Figure 34

![Figure 34](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x33.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 35: SSD Figure 35

![Figure 35](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x34.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 36: SSD Figure 36

![Figure 36](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x35.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 37: Figure 5:

![Figure 37](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x36.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 38: Figure 5:

![Figure 38](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x37.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 39: Figure 5:

![Figure 39](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x38.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 40: Figure 5: 3.6

![Figure 40](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x39.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 41: Figure 5: 3.6

![Figure 41](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x40.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 42: Figure 5: 3.6

![Figure 42](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x41.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 43: Figure 5: 3.6

![Figure 43](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x42.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 44: Figure 5: 3.6

![Figure 44](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x43.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 45: Figure 5: 3.6

![Figure 45](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x44.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 46: Figure 5: 3.6

![Figure 46](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x45.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 47: Figure 5: 3.6

![Figure 47](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x46.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 48: Table 6: Figure 6:

![Figure 48](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x48.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 49: Table 6: Figure 6:

![Figure 49](https://ar5iv.labs.arxiv.org/html/1512.02325/assets/x49.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### 关键表格

#### Table 1: SSD Table 1

| Method | data | mAP | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fast [ 6 ] | 07 | 66.9 | 74.5 | 78.3 | 69.2 | 53.2 | 36.6 | 77.3 | 78.2 | 82.0 | 40.7 | 72.7 | 67.9 | 79.6 | 79.2 | 73.0 | 69.0 | 30.1 | 65.4 | 70.2 | 75.8 | 65.8 |
| Fast [ 6 ] | 07+12 | 70.0 | 77.0 | 78.1 | 69.3 | 59.4 | 38.3 | 81.6 | 78.6 | 86.7 | 42.8 | 78.8 | 68.9 | 84.7 | 82.0 | 76.6 | 69.9 | 31.8 | 70.1 | 74.8 | 80.4 | 70.4 |
| Faster [ 2 ] | 07 | 69.9 | 70.0 | 80.6 | 70.1 | 57.3 | 49.9 | 78.2 | 80.4 | 82.0 | 52.2 | 75.3 | 67.2 | 80.3 | 79.8 | 75.0 | 76.3 | 39.1 | 68.3 | 67.3 | 81.1 | 67.6 |
| Faster [ 2 ] | 07+12 | 73.2 | 76.5 | 79.0 | 70.9 | 65.5 | 52.1 | 83.1 | 84.7 | 86.4 | 52.0 | 81.9 | 65.7 | 84.8 | 84.6 | 77.5 | 76.7 | 38.8 | 73.6 | 73.9 | 83.0 | 72.6 |
| Faster [ 2 ] | 07+12+COCO | 78.8 | 84.3 | 82.0 | 77.7 | 68.9 | 65.7 | 88.1 | 88.4 | 88.9 | 63.6 | 86.3 | 70.8 | 85.9 | 87.6 | 80.1 | 82.3 | 53.6 | 80.4 | 75.8 | 86.6 | 78.9 |
| SSD300 | 07 | 68.0 | 73.4 | 77.5 | 64.1 | 59.0 | 38.9 | 75.2 | 80.8 | 78.5 | 46.0 | 67.8 | 69.2 | 76.6 | 82.1 | 77.0 | 72.5 | 41.2 | 64.2 | 69.1 | 78.0 | 68.5 |
| SSD300 | 07+12 | 74.3 | 75.5 | 80.2 | 72.3 | 66.3 | 47.6 | 83.0 | 84.2 | 86.1 | 54.7 | 78.3 | 73.9 | 84.5 | 85.3 | 82.6 | 76.2 | 48.6 | 73.9 | 76.0 | 83.4 | 74.0 |
| SSD300 | 07+12+COCO | 79.6 | 80.9 | 86.3 | 79.0 | 76.2 | 57.6 | 87.3 | 88.2 | 88.6 | 60.5 | 85.4 | 76.7 | 87.5 | 89.2 | 84.5 | 81.4 | 55.0 | 81.9 | 81.5 | 85.9 | 78.9 |
| SSD512 | 07 | 71.6 | 75.1 | 81.4 | 69.8 | 60.8 | 46.3 | 82.6 | 84.7 | 84.1 | 48.5 | 75.0 | 67.4 | 82.3 | 83.9 | 79.4 | 76.6 | 44.9 | 69.9 | 69.1 | 78.1 | 71.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: SSD Table 2

| Prediction source layers from: | mAP |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| use boundary boxes? | # Boxes |  |  |  |  |  |  |  |
| conv4_3 | conv7 | conv8_2 | conv9_2 | conv10_2 | conv11_2 | Yes | No |  |
| ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | 74.3 | 63.4 | 8732 |
| ✔ | ✔ | ✔ | ✔ | ✔ |  | 74.6 | 63.1 | 8764 |
| ✔ | ✔ | ✔ | ✔ |  |  | 73.8 | 68.4 | 8942 |
| ✔ | ✔ | ✔ |  |  |  | 70.7 | 69.2 | 9864 |
| ✔ | ✔ |  |  |  |  | 64.2 | 64.4 | 9025 |
|  | ✔ |  |  |  |  | 62.4 | 64.0 | 8664 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: SSD Table 3

| Method | data | mAP | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fast [ 6 ] | 07++12 | 68.4 | 82.3 | 78.4 | 70.8 | 52.3 | 38.7 | 77.8 | 71.6 | 89.3 | 44.2 | 73.0 | 55.0 | 87.5 | 80.5 | 80.8 | 72.0 | 35.1 | 68.3 | 65.7 | 80.4 | 64.2 |
| Faster [ 2 ] | 07++12 | 70.4 | 84.9 | 79.8 | 74.3 | 53.9 | 49.8 | 77.5 | 75.9 | 88.5 | 45.6 | 77.1 | 55.3 | 86.9 | 81.7 | 80.9 | 79.6 | 40.1 | 72.6 | 60.9 | 81.2 | 61.5 |
| Faster [ 2 ] | 07++12+COCO | 75.9 | 87.4 | 83.6 | 76.8 | 62.9 | 59.6 | 81.9 | 82.0 | 91.3 | 54.9 | 82.6 | 59.0 | 89.0 | 85.5 | 84.7 | 84.1 | 52.2 | 78.9 | 65.5 | 85.4 | 70.2 |
| YOLO [ 5 ] | 07++12 | 57.9 | 77.0 | 67.2 | 57.7 | 38.3 | 22.7 | 68.3 | 55.9 | 81.4 | 36.2 | 60.8 | 48.5 | 77.2 | 72.3 | 71.3 | 63.5 | 28.9 | 52.2 | 54.8 | 73.9 | 50.8 |
| SSD300 | 07++12 | 72.4 | 85.6 | 80.1 | 70.5 | 57.6 | 46.2 | 79.4 | 76.1 | 89.2 | 53.0 | 77.0 | 60.8 | 87.0 | 83.1 | 82.3 | 79.4 | 45.9 | 75.9 | 69.5 | 81.9 | 67.5 |
| SSD300 | 07++12+COCO | 77.5 | 90.2 | 83.3 | 76.3 | 63.0 | 53.6 | 83.8 | 82.8 | 92.0 | 59.7 | 82.7 | 63.5 | 89.3 | 87.6 | 85.9 | 84.3 | 52.6 | 82.5 | 74.1 | 88.4 | 74.2 |
| SSD512 | 07++12 | 74.9 | 87.4 | 82.3 | 75.8 | 59.0 | 52.6 | 81.7 | 81.5 | 90.0 | 55.4 | 79.0 | 59.8 | 88.4 | 84.3 | 84.7 | 83.3 | 50.2 | 78.0 | 66.3 | 86.3 | 72.0 |
| SSD512 | 07++12+COCO | 80.0 | 90.7 | 86.8 | 80.5 | 67.8 | 60.8 | 86.3 | 85.5 | 93.5 | 63.2 | 85.7 | 64.4 | 90.9 | 89.0 | 88.9 | 86.8 | 57.2 | 85.1 | 72.8 | 88.4 | 75.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: SSD Table 4

| Method | data | Avg. Precision, IoU: | Avg. Precision, Area: | Avg. Recall, #Dets: | Avg. Recall, Area: |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5:0.95 | 0.5 | 0.75 | S | M | L | 1 | 10 | 100 | S | M | L |  |  |
| Fast [ 6 ] | train | 19.7 | 35.9 | - | - | - | - | - | - | - | - | - | - |
| Fast [ 24 ] | train | 20.5 | 39.9 | 19.4 | 4.1 | 20.0 | 35.8 | 21.3 | 29.5 | 30.1 | 7.3 | 32.1 | 52.0 |
| Faster [ 2 ] | trainval | 21.9 | 42.7 | - | - | - | - | - | - | - | - | - | - |
| ION [ 24 ] | train | 23.6 | 43.2 | 23.6 | 6.4 | 24.1 | 38.3 | 23.2 | 32.7 | 33.5 | 10.1 | 37.7 | 53.6 |
| Faster [ 25 ] | trainval | 24.2 | 45.3 | 23.5 | 7.7 | 26.4 | 37.1 | 23.8 | 34.0 | 34.6 | 12.0 | 38.5 | 54.4 |
| SSD300 | trainval35k | 23.2 | 41.2 | 23.4 | 5.3 | 23.2 | 39.6 | 22.5 | 33.2 | 35.3 | 9.6 | 37.6 | 56.5 |
| SSD512 | trainval35k | 26.8 | 46.5 | 27.8 | 9.0 | 28.9 | 41.9 | 24.8 | 37.5 | 39.8 | 14.0 | 43.5 | 59.0 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: SSD Table 5

| Method | VOC2007 test | VOC2012 test | COCO test-dev2015 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 07+12 | 07+12+COCO | 07++12 | 07++12+COCO | trainval35k |  |  |
|  | 0.5 | 0.5 | 0.5 | 0.5 | 0.5:0.95 | 0.5 | 0.75 |
| SSD300 | 74.3 | 79.6 | 72.4 | 77.5 | 23.2 | 41.2 | 23.4 |
| SSD512 | 76.8 | 81.6 | 74.9 | 80.0 | 26.8 | 46.5 | 27.8 |
| SSD300* | 77.2 | 81.2 | 75.8 | 79.3 | 25.1 | 43.1 | 25.8 |
| SSD512* | 79.8 | 83.2 | 78.5 | 82.2 | 28.8 | 48.5 | 30.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: SSD Table 6

| Method | mAP | FPS | batch size | # Boxes | Input resolution |
| --- | --- | --- | --- | --- | --- |
| Faster R-CNN (VGG16) | 73.2 | 7 | 1 | ∼ 6000 similar-to absent 6000 \sim 6000 | ∼ 1000 × 600 similar-to absent 1000 600 \sim 1000\times 600 |
| Fast YOLO | 52.7 | 155 | 1 | 98 | 448 × 448 448 448 448\times 448 |
| YOLO (VGG16) | 66.4 | 21 | 1 | 98 | 448 × 448 448 448 448\times 448 |
| SSD300 | 74.3 | 46 | 1 | 8732 | 300 × 300 300 300 300\times 300 |
| SSD512 | 76.8 | 19 | 1 | 24564 | 512 × 512 512 512 512\times 512 |
| SSD300 | 74.3 | 59 | 8 | 8732 | 300 × 300 300 300 300\times 300 |
| SSD512 | 76.8 | 22 | 8 | 24564 | 512 × 512 512 512 512\times 512 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple featu

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

- [[YOLOv1]]
- [[FPN]]
- [[RetinaNet]]
