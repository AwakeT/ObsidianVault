---
title: "Cascade R-CNN: Delving into High Quality Object Detection"
method_name: "Cascade R-CNN"
authors: [Zhaowei Cai, Nuno Vasconcelos]
year: 2017
venue: CVPR
tags: [object-detection, cascade, high-quality-detection, roi-head]
arxiv: https://arxiv.org/abs/1712.00726
created: 2026-06-11
image_source: online
---

# 论文笔记：Cascade R-CNN: Delving into High Quality Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[Cascade R-CNN]] |
| 作者 | [Zhaowei Cai, Nuno Vasconcelos] |
| 年份 | 2017 |
| 会议/来源 | CVPR |
| 链接 | [arXiv](https://arxiv.org/abs/1712.00726) / [PDF](https://arxiv.org/pdf/1712.00726.pdf) |
| 相关工作 | [[Faster R-CNN]]、[[Mask R-CNN]]、[[GFL]]、[[D-FINE]] |

---

## 一句话总结

> Cascade R-CNN 用多个递进 IoU 阈值的检测头逐步提升框质量，是高质量检测的重要范式。

---

## 核心贡献

1. **范式贡献**：Cascade R-CNN 用多个递进 IoU 阈值的检测头逐步提升框质量，是高质量检测的重要范式。
2. **方法贡献**：串联多个 ROI head，每一级使用更高 IoU 阈值训练，并用前一级 refined boxes 作为后一级输入。
3. **工程/研究影响**：该工作成为后续 [[Faster R-CNN]]、[[Mask R-CNN]]、[[GFL]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

串联多个 ROI head，每一级使用更高 IoU 阈值训练，并用前一级 refined boxes 作为后一级输入。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[Faster R-CNN]]、[[Mask R-CNN]]。
- 后续影响：[[GFL]]、[[D-FINE]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
h_t(x_t)=f_t(x_t),\quad IoU_t<IoU_{t+1}
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1712.00726](https://ar5iv.labs.arxiv.org/html/1712.00726)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 24 张、Table 6 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: The detection outputs, localization and detection performance of object detectors of increasing IoU threshold u 𝑢 u .

![Figure 1](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x1.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 2: Figure 1: The detection outputs, localization and detection performance of object detectors of increasing IoU threshold u 𝑢 u .

![Figure 2](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x2.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 3: Figure 1: The detection outputs, localization and detection performance of object detectors of increasing IoU threshold u 𝑢 u .

![Figure 3](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x3.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 4: Figure 1: The detection outputs, localization and detection performance of object detectors of increasing IoU threshold u 𝑢 u .

![Figure 4](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x4.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 5: Figure 2: Sequential Δ Δ \Delta distribution (without normalization) at different cascade stage. Red dots are outliers when using increasing IoU thresholds, and the statistics are obtained after outlier removal.

![Figure 5](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x5.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 6: Figure 2: Sequential Δ Δ \Delta distribution (without normalization) at different cascade stage. Red dots are outliers when using increasing IoU thresholds, and the statistics are obtained after outlier removal.

![Figure 6](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x6.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 7: Figure 2: Sequential Δ Δ \Delta distribution (without normalization) at different cascade stage. Red dots are outliers when using increasing IoU thresholds, and the statistics are obtained after outlier removal.

![Figure 7](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x7.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 8: Figure 2: Sequential Δ Δ \Delta distribution (without normalization) at different cascade stage. Red dots are outliers when using increasing IoU thresholds, and the statistics are obtained after outlier removal.

![Figure 8](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x8.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 9: Figure 2: Sequential Δ Δ \Delta distribution (without normalization) at different cascade stage. Red dots are outliers when using increasing IoU thresholds, and the statistics are obtained after outlier removal.

![Figure 9](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x9.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 10: Figure 2: Sequential Δ Δ \Delta distribution (without normalization) at different cascade stage. Red dots are outliers when using increasing IoU thresholds, and the statistics are obtained after outlier removal.

![Figure 10](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x10.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 11: Figure 3: The architectures of different frameworks. “I” is input image, “conv” backbone convolutions, “pool” region-wise feature extraction, “H” network head, “B” bounding box, and “C” classification. “B0” is proposals in all architectures.

![Figure 11](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x11.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 12: Figure 3: The architectures of different frameworks. “I” is input image, “conv” backbone convolutions, “pool” region-wise feature extraction, “H” network head, “B” bounding box, and “C” classification. “B0” is proposals in all architectures.

![Figure 12](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x12.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 13: Figure 3: The architectures of different frameworks. “I” is input image, “conv” backbone convolutions, “pool” region-wise feature extraction, “H” network head, “B” bounding box, and “C” classification. “B0” is proposals in all architectures.

![Figure 13](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x13.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 14: Figure 3: The architectures of different frameworks. “I” is input image, “conv” backbone convolutions, “pool” region-wise feature extraction, “H” network head, “B” bounding box, and “C” classification. “B0” is proposals in all architectures.

![Figure 14](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x14.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 15: Figure 4: The IoU histogram of training samples. The distribution at 1st stage is the output of RPN. The red numbers are the positive percentage higher than the corresponding IoU threshold.

![Figure 15](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x15.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 16: Figure 4: The IoU histogram of training samples. The distribution at 1st stage is the output of RPN. The red numbers are the positive percentage higher than the corresponding IoU threshold.

![Figure 16](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x16.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 17: Figure 4: The IoU histogram of training samples. The distribution at 1st stage is the output of RPN. The red numbers are the positive percentage higher than the corresponding IoU threshold.

![Figure 17](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x17.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 18: Figure 5: (a) is detection performance of individually trained detectors, with their own proposals (solid curves) or Cascade R-CNN stage proposals (dashed curves), and (b) is by adding ground truth to the proposal set.

![Figure 18](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x18.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 19: Figure 5: (a) is detection performance of individually trained detectors, with their own proposals (solid curves) or Cascade R-CNN stage proposals (dashed curves), and (b) is by adding ground truth to the proposal set.

![Figure 19](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x19.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 20: Figure 6: The detection performance of all Cascade R-CNN detectors at all cascade stages.

![Figure 20](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x20.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 21: Figure 6: The detection performance of all Cascade R-CNN detectors at all cascade stages.

![Figure 21](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x21.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 22: Figure 6: The detection performance of all Cascade R-CNN detectors at all cascade stages.

![Figure 22](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x22.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 23: Figure 7: (a) is the localization comparison, and (b) is the detection performance of individual classifiers in the integral loss detector.

![Figure 23](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x23.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 24: Figure 7: (a) is the localization comparison, and (b) is the detection performance of individual classifiers in the integral loss detector.

![Figure 24](https://ar5iv.labs.arxiv.org/html/1712.00726/assets/x24.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### 关键表格

#### Table 1: Cascade R-CNN Table 1

|  | AP | AP 50 | AP 60 | AP 70 | AP 80 | AP 90 |
| --- | --- | --- | --- | --- | --- | --- |
| FPN+ baseline | 34.9 | 57.0 | 51.9 | 43.6 | 29.7 | 7.1 |
| Iterative BBox | 35.4 | 57.2 | 52.1 | 44.2 | 30.4 | 8.1 |
| Integral Loss | 35.4 | 57.3 | 52.5 | 44.4 | 29.9 | 6.9 |
| Cascade R-CNN | 38.9 | 57.8 | 53.4 | 46.9 | 35.8 | 15.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: Cascade R-CNN Table 2

| test stage | AP | AP 50 | AP 60 | AP 70 | AP 80 | AP 90 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 35.5 | 57.2 | 52.4 | 44.1 | 30.5 | 8.1 |
| 2 | 38.3 | 57.9 | 53.4 | 46.4 | 35.2 | 14.2 |
| 3 | 38.3 | 56.6 | 52.2 | 46.3 | 35.7 | 15.9 |
| 1 ∼ 2 ¯ ¯ similar-to 1 2 \overline{1\sim{2}} | 38.5 | 58.2 | 53.8 | 46.7 | 35.0 | 14.0 |
| 1 ∼ 3 ¯ ¯ similar-to 1 3 \overline{1\sim{3}} | 38.9 | 57.8 | 53.4 | 46.9 | 35.8 | 15.8 |
| FPN+ baseline | 34.9 | 57.0 | 51.9 | 43.6 | 29.7 | 7.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: Cascade R-CNN Table 3

| IoU ↑ ↑ \uparrow | stat | AP | AP 50 | AP 60 | AP 70 | AP 80 | AP 90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 36.8 | 57.8 | 52.9 | 45.4 | 32.0 | 10.7 |
| ✓ |  | 38.5 | 58.4 | 54.1 | 47.1 | 35.0 | 13.1 |
|  | ✓ | 37.5 | 57.8 | 53.1 | 45.5 | 33.3 | 13.1 |
| ✓ | ✓ | 38.9 | 57.8 | 53.4 | 46.9 | 35.8 | 15.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: Cascade R-CNN Table 4

| # stages | test stage | AP | AP 50 | AP 60 | AP 70 | AP 80 | AP 90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 34.9 | 57.0 | 51.9 | 43.6 | 29.7 | 7.1 |
| 2 | 1 ∼ 2 ¯ ¯ similar-to 1 2 \overline{1\sim{2}} | 38.2 | 58.0 | 53.6 | 46.7 | 34.6 | 13.6 |
| 3 | 1 ∼ 3 ¯ ¯ similar-to 1 3 \overline{1\sim{3}} | 38.9 | 57.8 | 53.4 | 46.9 | 35.8 | 15.8 |
| 4 | 1 ∼ 3 ¯ ¯ similar-to 1 3 \overline{1\sim{3}} | 38.9 | 57.4 | 53.2 | 46.8 | 36.0 | 16.0 |
| 4 | 1 ∼ 4 ¯ ¯ similar-to 1 4 \overline{1\sim{4}} | 38.6 | 57.2 | 52.8 | 46.2 | 35.5 | 16.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: Cascade R-CNN Table 5

|  | backbone | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv2 [ 26 ] | DarkNet-19 | 21.6 | 44.0 | 19.2 | 5.0 | 22.4 | 35.5 |
| SSD513 [ 23 ] | ResNet-101 | 31.2 | 50.4 | 33.3 | 10.2 | 34.5 | 49.8 |
| RetinaNet [ 22 ] | ResNet-101 | 39.1 | 59.1 | 42.3 | 21.8 | 42.7 | 50.2 |
| Faster R-CNN+++ [ 16 ] * | ResNet-101 | 34.9 | 55.7 | 37.4 | 15.6 | 38.7 | 50.9 |
| Faster R-CNN w FPN [ 21 ] | ResNet-101 | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2 |
| Faster R-CNN w FPN+ (ours) | ResNet-101 | 38.8 | 61.1 | 41.9 | 21.3 | 41.8 | 49.8 |
| Faster R-CNN by G-RMI [ 17 ] | Inception-ResNet-v2 | 34.7 | 55.5 | 36.7 | 13.5 | 38.1 | 52.0 |
| Deformable R-FCN [ 5 ] * | Aligned-Inception-ResNet | 37.5 | 58.0 | 40.8 | 19.4 | 40.1 | 52.5 |
| Mask R-CNN [ 14 ] | ResNet-101 | 38.2 | 60.3 | 41.7 | 20.1 | 41.1 | 50.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: Cascade R-CNN Table 6

|  | backbone | cascade | train | test | param | val (5k) | test-dev (20k) |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | speed | speed | AP | AP 50 | AP 75 | AP S | AP M | AP L | AP | AP 50 | AP 75 | AP S | AP M | AP L |  |  |  |
| Faster R-CNN | VGG | ✗ | 0.12s | 0.075s | 278M | 23.6 | 43.9 | 23.0 | 8.0 | 26.2 | 35.5 | 23.5 | 43.9 | 22.6 | 8.1 | 25.1 | 34.7 |
| ✓ | 0.14s | 0.115s | 704M | 27.0 | 44.2 | 27.7 | 8.6 | 29.1 | 42.2 | 26.9 | 44.3 | 27.8 | 8.3 | 28.2 | 41.1 |  |  |
| R-FCN | ResNet-50 | ✗ | 0.19s | 0.07s | 133M | 27.0 | 48.7 | 26.9 | 9.8 | 30.9 | 40.3 | 27.1 | 49.0 | 26.9 | 10.4 | 29.7 | 39.2 |
| ✓ | 0.24s | 0.075s | 184M | 31.1 | 49.8 | 32.8 | 10.4 | 34.4 | 48.5 | 30.9 | 49.9 | 32.6 | 10.5 | 33.1 | 46.9 |  |  |
| R-FCN | ResNet-101 | ✗ | 0.23s | 0.075s | 206M | 30.3 | 52.2 | 30.8 | 12.0 | 34.7 | 44.3 | 30.5 | 52.9 | 31.2 | 12.0 | 33.9 | 43.8 |
| ✓ | 0.29s | 0.083s | 256M | 33.3 | 52.0 | 35.2 | 11.8 | 37.2 | 51.1 | 33.3 | 52.6 | 35.2 | 12.1 | 36.2 | 49.3 |  |  |
| FPN+ | ResNet-50 | ✗ | 0.30s | 0.095s | 165M | 36.5 | 58.6 | 39.2 | 20.8 | 40.0 | 47.8 | 36.5 | 59.0 | 39.2 | 20.3 | 38.8 | 46.4 |
| ✓ | 0.33s | 0.115s | 272M | 40.3 | 59.4 | 43.7 | 22.9 | 43.7 | 54.1 | 40.6 | 59.9 | 44.0 | 22.6 | 42.7 | 52.1 |  |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

In object detection, an intersection over union (IoU) threshold is required to define positives and negatives. An object detector, trained with low IoU threshold, e.g. 0.5, usually produces noisy detections. However, detection performance tends to degrade with increasing the IoU thresholds. Two main factors are responsible for this: 1) overfitting during training, due to exponentially vanishing positive samples, and 2) inference-time mismatch between the IoUs for which the detector is optimal an

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

- [[Faster R-CNN]]
- [[Mask R-CNN]]
- [[GFL]]
- [[D-FINE]]
