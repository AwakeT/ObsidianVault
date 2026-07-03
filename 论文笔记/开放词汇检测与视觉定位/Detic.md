---
title: "Detecting Twenty-thousand Classes using Image-level Supervision"
method_name: "Detic"
authors: [Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, Ishan Misra]
year: 2022
venue: ECCV
tags: [open-vocabulary-detection, long-tail, weak-supervision, lvis]
arxiv: https://arxiv.org/abs/2201.02605
created: 2026-06-11
image_source: online
---

# 论文笔记：Detecting Twenty-thousand Classes using Image-level Supervision

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[Detic]] |
| 作者 | [Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, Ishan Misra] |
| 年份 | 2022 |
| 会议/来源 | ECCV |
| 链接 | [arXiv](https://arxiv.org/abs/2201.02605) / [PDF](https://arxiv.org/pdf/2201.02605.pdf) |
| 相关工作 | [[GLIP]]、[[LVIS]]、[[Grounding DINO]]、[[OWLv2]] |

---

## 一句话总结

> Detic 利用 image-level labels 扩展检测类别到大词表，是长尾和开放词汇检测的重要方法。

---

## 核心贡献

1. **范式贡献**：Detic 利用 image-level labels 扩展检测类别到大词表，是长尾和开放词汇检测的重要方法。
2. **方法贡献**：结合 box-level detection supervision 和 image-level classification supervision，把检测类别扩展到大词表。
3. **工程/研究影响**：该工作成为后续 [[GLIP]]、[[LVIS]]、[[Grounding DINO]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

结合 box-level detection supervision 和 image-level classification supervision，把检测类别扩展到大词表。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[GLIP]]、[[LVIS]]。
- 后续影响：[[Grounding DINO]]、[[OWLv2]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\mathcal{L}=\mathcal{L}_{det}+\lambda\mathcal{L}_{image-cls}
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2201.02605](https://ar5iv.labs.arxiv.org/html/2201.02605)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 27 张、Table 12 张；下方按论文 HTML 顺序保留。

### Figure 1: Refer to caption

![Figure 1](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x1.png)

**图表解读**：重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 2: Refer to caption

![Figure 2](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x2.png)

**图表解读**：重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 3: (a) Standard detection

![Figure 3](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x3.png)

**图表解读**：重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 4: (b) Prediction-based label assignment

![Figure 4](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x4.png)

**图表解读**：重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 5: (c) Our non-prediction-based loss

![Figure 5](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x5.png)

**图表解读**：重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 6: (a) Detection data

![Figure 6](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x6.png)

**图表解读**：重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 7: (b) Image-labeled data

![Figure 7](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x7.png)

**图表解读**：重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 8: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 8](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/dynamic/1_sc_30k.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 9: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 9](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/dynamic/1_sc_60k.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 10: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 10](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/dynamic/1_sc_90k.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 11: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 11](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x8.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 12: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 12](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x9.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 13: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 13](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x10.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 14: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 14](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/dynamic/1_sz_30k.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 15: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 15](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/dynamic/1_sz_60k.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 16: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 16](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/dynamic/1_sz_90k.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 17: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 17](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x11.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 18: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 18](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x12.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 19: Figure 4 : Visualization of the assigned boxes during training. We show all boxes with score > 0.5 absent 0.5 >0.5 in blue and the assigned (selected) box in red . Top : The prediction-based method selects different boxes across training, and the selected box may not cover the objects in the image. Bottom : Our simpler max-size variant selects a box that covers the objects and is more consistent across training.

![Figure 19](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x13.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 20: Figure 5 : Qualitative results of our 21k-class detector. We show random samples from images containing novel classes in OpenImages (top) and Objects365 (bottom) validation sets. We use the CLIP embedding of the corresponding vocabularies. We show LVIS classes in purple and novel classes in green . We use a score threshold of 0.5 0.5 0.5 and show the most confident class for each box. Best viewed on screen.

![Figure 20](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x14.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 21: Figure 5 : Qualitative results of our 21k-class detector. We show random samples from images containing novel classes in OpenImages (top) and Objects365 (bottom) validation sets. We use the CLIP embedding of the corresponding vocabularies. We show LVIS classes in purple and novel classes in green . We use a score threshold of 0.5 0.5 0.5 and show the most confident class for each box. Best viewed on screen.

![Figure 21](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x15.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 22: Figure 5 : Qualitative results of our 21k-class detector. We show random samples from images containing novel classes in OpenImages (top) and Objects365 (bottom) validation sets. We use the CLIP embedding of the corresponding vocabularies. We show LVIS classes in purple and novel classes in green . We use a score threshold of 0.5 0.5 0.5 and show the most confident class for each box. Best viewed on screen.

![Figure 22](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x16.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 23: Figure 5 : Qualitative results of our 21k-class detector. We show random samples from images containing novel classes in OpenImages (top) and Objects365 (bottom) validation sets. We use the CLIP embedding of the corresponding vocabularies. We show LVIS classes in purple and novel classes in green . We use a score threshold of 0.5 0.5 0.5 and show the most confident class for each box. Best viewed on screen.

![Figure 23](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x17.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 24: Figure 5 : Qualitative results of our 21k-class detector. We show random samples from images containing novel classes in OpenImages (top) and Objects365 (bottom) validation sets. We use the CLIP embedding of the corresponding vocabularies. We show LVIS classes in purple and novel classes in green . We use a score threshold of 0.5 0.5 0.5 and show the most confident class for each box. Best viewed on screen.

![Figure 24](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/qualitative/o365/160.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 25: Figure 5 : Qualitative results of our 21k-class detector. We show random samples from images containing novel classes in OpenImages (top) and Objects365 (bottom) validation sets. We use the CLIP embedding of the corresponding vocabularies. We show LVIS classes in purple and novel classes in green . We use a score threshold of 0.5 0.5 0.5 and show the most confident class for each box. Best viewed on screen.

![Figure 25](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x18.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 26: Figure 5 : Qualitative results of our 21k-class detector. We show random samples from images containing novel classes in OpenImages (top) and Objects365 (bottom) validation sets. We use the CLIP embedding of the corresponding vocabularies. We show LVIS classes in purple and novel classes in green . We use a score threshold of 0.5 0.5 0.5 and show the most confident class for each box. Best viewed on screen.

![Figure 26](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/qualitative/o365/1450.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### Figure 27: Figure 5 : Qualitative results of our 21k-class detector. We show random samples from images containing novel classes in OpenImages (top) and Objects365 (bottom) validation sets. We use the CLIP embedding of the corresponding vocabularies. We show LVIS classes in purple and novel classes in green . We use a score threshold of 0.5 0.5 0.5 and show the most confident class for each box. Best viewed on screen.

![Figure 27](https://ar5iv.labs.arxiv.org/html/2201.02605/assets/x19.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 image-level supervision 如何扩展 detection vocabulary；对 LVIS/Objects365 长尾类别有借鉴意义。

### 关键表格

#### Table 1: Detic Table 1

| Notation | Definition | #Images | #Classes |
| --- | --- | --- | --- |
| LVIS-all | The original LVIS dataset [ 18 ] | 100K | 1203 |
| LVIS-base | LVIS without rare-class annotations | 100K | 866 |
| IN-21K | ​​​The original ImageNet-21K dataset [ 10 ] | 14M | 21k |
| IN-L | ​​​​​​​997 overlapping IN-21K classes with LVIS | 1.2M | 997 |
| CC | ​​​​​​​Conceptual Captions [ 50 ] with LVIS classes | 1.5M | 992 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: Detic Table 2

|  |  | ​​​​IN-L (object-centric) | ​​​​CC (non object-centric) |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  | mAP mask mask {}^{\text{mask}} | mAP novel mask subscript superscript absent mask novel {}^{\text{mask}}_{\text{n | mAP mask mask {}^{\text{mask}} | mAP novel mask subscript superscript absent mask novel {}^{\text{mask}}_{\text{n |
|  | Box-Supervised (baseline) | 30.0 ± 0.4 plus-or-minus 0.4 \pm 0.4 | 16.3 ± 0.7 plus-or-minus 0.7 \pm 0.7 | 30.0 ± 0.4 plus-or-minus 0.4 \pm 0.4 | 16.3 ± 0.7 plus-or-minus 0.7 \pm 0.7 |
| Prediction-based methods |  |  |  |  |  |
|  | Self-training [ 54 ] | 30.3 ± 0.0 plus-or-minus 0.0 \pm 0.0 | 15.6 ± 0.1 plus-or-minus 0.1 \pm 0.1 | 30.1 ± 0.2 plus-or-minus 0.2 \pm 0.2 | 15.9 ± 0.8 plus-or-minus 0.8 \pm 0.8 |
|  | WSDDN [ 3 ] | 29.8 ± 0.2 plus-or-minus 0.2 \pm 0.2 | 15.6 ± 0.3 plus-or-minus 0.3 \pm 0.3 | 30.0 ± 0.1 plus-or-minus 0.1 \pm 0.1 | 16.5 ± 0.8 plus-or-minus 0.8 \pm 0.8 |
|  | DLWL* [ 44 ] | 30.6 ± 0.1 plus-or-minus 0.1 \pm 0.1 | 18.2 ± 0.2 plus-or-minus 0.2 \pm 0.2 | 29.7 ± 0.3 plus-or-minus 0.3 \pm 0.3 | 16.9 ± 0.6 plus-or-minus 0.6 \pm 0.6 |
|  | YOLO9000 [ 45 ] | 31.2 ± 0.3 plus-or-minus 0.3 \pm 0.3 | 20.4 ± 0.9 plus-or-minus 0.9 \pm 0.9 | 29.4 ± 0.1 plus-or-minus 0.1 \pm 0.1 | 15.9 ± 0.6 plus-or-minus 0.6 \pm 0.6 |
| Non-prediction-based methods |  |  |  |  |  |
|  | Detic (Max-object-score) | 32.2 ± 0.1 plus-or-minus 0.1 \pm 0.1 | 24.4 ± 0.3 plus-or-minus 0.3 \pm 0.3 | 29.8 ± 0.1 plus-or-minus 0.1 \pm 0.1 | 18.2 ± 0.6 plus-or-minus 0.6 \pm 0.6 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: Detic Table 3

|  | mAP mask mask {}^{\text{mask}} | mAP novel mask subscript superscript absent mask novel {}^{\text{mask}}_{\text{n | mAP c mask subscript superscript absent mask c {}^{\text{mask}}_{\text{c}} | mAP f mask subscript superscript absent mask f {}^{\text{mask}}_{\text{f}} |
| --- | --- | --- | --- | --- |
| ViLD-text [ 17 ] | 24.9 | 10.1 | 23.9 | 32.5 |
| ViLD [ 17 ] | 22.5 | 16.1 | 20.0 | 28.3 |
| ViLD-ensemble [ 17 ] | 25.5 | 16.6 | 24.6 | 30.3 |
| Detic | 26.8 | 17.8 | 26.3 | 31.6 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: Detic Table 4

|  | mAP50 all box subscript superscript absent box all {}^{\text{box}}_{\text{all}} | mAP50 novel box subscript superscript absent box novel {}^{\text{box}}_{\text{no | mAP50 base box subscript superscript absent box base {}^{\text{box}}_{\text{base |
| --- | --- | --- | --- |
| Base-only† | 39.9 | 0 | 49.9 |
| Base-only (CLIP) | 39.3 | 1.3 | 48.7 |
| WSDDN [ 3 ] † | 24.6 | 20.5 | 23.4 |
| Cap2Det [ 71 ] † | 20.1 | 20.3 | 20.1 |
| SB [ 2 ] ‡ | 24.9 | 0.31 | 29.2 |
| DELO [ 78 ] ‡ | 13.0 | 3.41 | 13.8 |
| PL [ 43 ] ‡ | 27.9 | 4.12 | 35.9 |
| OVR-CNN [ 72 ] † | 39.9 | 22.8 | 46.0 |
| Detic | 45.0 | 27.8 | 47.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: Detic Table 5

|  | Objects365 [ 49 ] | OpenImages [ 28 ] |  |  |
| --- | --- | --- | --- | --- |
|  | mAP box box {}^{\text{box}} | mAP rare box subscript superscript absent box rare {}^{\text{box}}_{\text{rare}} | mAP50 box box {}^{\text{box}} | mAP50 rare box subscript superscript absent box rare {}^{\text{box}}_{\text{rare |
| Box-Supervised | 19.1 | 14.0 | 46.2 | 61.7 |
| Detic w. IN-L | 21.2 | 17.8 | 53.0 | 67.1 |
| Detic w. IN-21k | 21.5 | 20.0 | 55.2 | 68.8 |
| Dataset-specific oracles | 31.2 | 22.5 | 69.9 | 81.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: Detic Table 6

|  | Box-Supervised | Detic |  |  |
| --- | --- | --- | --- | --- |
| Classifier | mAP mask mask {}^{\text{mask}} | mAP novel mask subscript superscript absent mask novel {}^{\text{mask}}_{\text{n | mAP mask mask {}^{\text{mask}} | mAP novel mask subscript superscript absent mask novel {}^{\text{mask}}_{\text{n |
| *CLIP [ 42 ] | 30.2 | 16.4 | 32.4 | 24.9 |
| Trained | 27.4 | 0 | 31.7 | 17.4 |
| FastText [ 24 ] | 27.5 | 9.0 | 30.9 | 19.2 |
| OpenCLIP [ 23 ] | 27.1 | 8.9 | 30.7 | 19.4 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: Detic Table 7

|  | Pretrain data | mAP mask mask {}^{\text{mask}} | mAP novel mask subscript superscript absent mask novel {}^{\text{mask}}_{\text{n |
| --- | --- | --- | --- |
| Box-Supervised | IN-1K | 26.1 | 13.6 |
| Detic | IN-1K | 28.8 (+2.7) | 21.7 (+8.1) |
| Box-Supervised | IN-21K | 30.2 | 16.4 |
| Detic | IN-21K | 32.4 (+2.2) | 24.9 (+8.5) |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: Detic Table 8

|  | Backbone | mAP mask mask {}^{\text{mask}} | mAP r mask subscript superscript absent mask r {}^{\text{mask}}_{\text{r}} | mAP c mask subscript superscript absent mask c {}^{\text{mask}}_{\text{c}} | mAP f mask subscript superscript absent mask f {}^{\text{mask}}_{\text{f}} |
| --- | --- | --- | --- | --- | --- |
| MosaicOS † † \dagger [ 73 ] | ResNeXt-101 | 28.3 | 21.7 | 27.3 | 32.4 |
| CenterNet2 [ 76 ] | ResNeXt-101 | 34.9 | 24.6 | 34.7 | 42.5 |
| AsyncSLL † † \dagger [ 19 ] | ResNeSt-269 | 36.0 | 27.8 | 36.7 | 39.6 |
| SeesawLoss [ 64 ] ​ | ResNeSt-200 | 37.3 | 26.4 | 36.3 | 43.1 |
| Copy-paste [ 15 ] | EfficientNet-B7​​​​ | 38.1 | 32.1 | 37.1 | 41.9 |
| Tan et al. [ 57 ] | ResNeSt-269 | 38.8 | 28.5 | 39.5 | 42.7 |
| Baseline | Swin-B | 40.7 | 35.9 | 40.5 | 43.1 |
| Detic † † \dagger | Swin-B | 41.7 | 41.7 | 40.8 | 42.6 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 9: Detic Table 9

|  | AR r 50@100 | AR r 50@300 | AR r 50@1k | AR50@1k |
| --- | --- | --- | --- | --- |
| LVIS-all | 63.3 | 76.3 | 79.7 | 80.9 |
| LVIS-base | 62.2 | 76.2 | 78.5 | 81.0 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 10: Detic Table 10

|  | AR half-1st half-1st {}_{\text{half-1st}} 50@1k | AR half-2nd half-2nd {}_{\text{half-2nd}} 50@1k |
| --- | --- | --- |
| LVIS-half-1st | 80.8 | 69.6 |
| LVIS-half-2nd | 62.9 | 82.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 11: Detic Table 11

|  | Supervision | mAP mask mask {}^{\text{mask}} | mAP novel mask subscript superscript absent mask novel {}^{\text{mask}}_{\text{n |
| --- | --- | --- | --- |
| Box-Supervised | - | 30.2 | 16.4 |
| Detic w. CC | Image label | 31.0 | 19.8 |
| Detic w. CC | Caption | 30.4 | 17.4 |
| Detic w. CC | Both | 31.0 | 21.3 |
|  |  | mAP50 all box subscript superscript absent box all {}^{\text{box}}_{\text{all}} | mAP50 novel box subscript superscript absent box novel {}^{\text{box}}_{\text{no |
| Box-Supervised | - | 39.3 | 1.3 |
| Detic w. COCO-cap. | Image label | 44.7 | 24.1 |
| Detic w. COCO-cap. | Caption | 43.8 | 21.0 |
| Detic w. COCO-cap. | Both | 45.0 | 27.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 12: Detic Table 12

|  | mAP box box {}^{\text{box}} | mAP r box subscript superscript absent box r {}^{\text{box}}_{\text{r}} | mAP mask mask {}^{\text{mask}} | mAP r mask subscript superscript absent mask r {}^{\text{mask}}_{\text{r}} | T |
| --- | --- | --- | --- | --- | --- |
| D2 baseline [ 66 ] | 22.9 | 11.3 | 22.4 | 11.6 | 12h |
| +Class-agnostic box&mask | 22.3 | 10.1 | 21.2 | 10.1 | 12h |
| +Federated loss [ 76 ] | 27.0 | 20.2 | 24.6 | 18.2 | 12h |
| +CenterNet2 [ 76 ] | 30.7 | 22.9 | 26.8 | 19.4 | 13h |
| +LSJ 640 × 640 640 640 640\!\times\!640 , 4 × 4\!\times\! sched.​ [ 15 ] ​​​ | 31.0 | 21.6 | 27.2 | 20.1 | 17h |
| +CLIP classifier [ 42 ] | 31.5 | 24.2 | 28 | 22.5 | 17h |
| +Adam optimizer, lr 2 ​ e 2 𝑒 2e - 4 4 4 [ 26 ] | 30.4 | 23.6 | 26.9 | 21.4 | 17h |
| +IN-21k pretrain [ 48 ] * | 35.3 | 28.2 | 31.5 | 25.6 | 17h |
| +Input size 896 × 896 896 896 896\!\times\!896 | 37.1 | 29.5 | 33.2 | 26.9 | 25h |

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

- [[GLIP]]
- [[LVIS]]
- [[Grounding DINO]]
- [[OWLv2]]
