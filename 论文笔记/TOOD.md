---
title: "TOOD: Task-aligned One-stage Object Detection"
method_name: "TOOD"
authors: [Chengjian Feng, Yujie Zhong, Yu Gao, Matthew Scott, Weilin Huang]
year: 2021
venue: ICCV
tags: [object-detection, task-alignment, sample-assignment, one-stage-detection]
arxiv: https://arxiv.org/abs/2108.07755
created: 2026-06-11
image_source: online
---

# 论文笔记：TOOD: Task-aligned One-stage Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[TOOD]] |
| 作者 | [Chengjian Feng, Yujie Zhong, Yu Gao, Matthew Scott, Weilin Huang] |
| 年份 | 2021 |
| 会议/来源 | ICCV |
| 链接 | [arXiv](https://arxiv.org/abs/2108.07755) / [PDF](https://arxiv.org/pdf/2108.07755.pdf) |
| 相关工作 | [[GFL]]、[[ATSS]]、[[YOLOX]]、[[FCOS]] |

---

## 一句话总结

> TOOD 解决一阶段检测中分类和定位任务不对齐的问题，通过 task-aligned learning 让高分类分数对应高质量框。

---

## 核心贡献

1. **范式贡献**：TOOD 解决一阶段检测中分类和定位任务不对齐的问题，通过 task-aligned learning 让高分类分数对应高质量框。
2. **方法贡献**：用 task-aligned learning 同时考虑分类分数和 IoU 质量，缓解分类最优点与定位最优点不一致。
3. **工程/研究影响**：该工作成为后续 [[GFL]]、[[ATSS]]、[[YOLOX]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

用 task-aligned learning 同时考虑分类分数和 IoU 质量，缓解分类最优点与定位最优点不一致。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[GFL]]、[[ATSS]]。
- 后续影响：[[YOLOX]]、[[FCOS]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
t=s^\alpha u^\beta
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2108.07755](https://ar5iv.labs.arxiv.org/html/2108.07755)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 29 张、Table 6 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: Illustration of detection results (‘Result’) and spatial distributions of classification scores (‘Score’) and localization scores (‘IoU’) predicted by ATSS [ 31 ] (top row) and the proposed TOOD (bottom row). Ground-truth is indicated by yellow boxes, and a white arrow means the main direction of the best anchor away from the center of an object. In the ‘Result’ column, a red/green patch is the location of the best anchor for classification/localization, while a red/green box means an object bounding box predicted from the anchor in the red/green patch (if they coincide, we only show the red patches and boxes).

![Figure 1](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/x1.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 2: Figure 2: Overall learning mechanism of TOOD. First, predictions are made by T-head on the FPN features. Second, the predictions are used to compute a task alignment metric at each anchor point, based on which TAL produces learning signals for T-head . Lastly, T-head adjusts the distributions of classification and localization accordingly. Specifically, the most aligned anchor obtains a higher classification score through ‘prob’ (probability map), and acquires a more accurate bounding box prediction via a learned ‘offset’.

![Figure 2](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/x2.png)

**图表解读**：重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 3: (a) Parallel head

![Figure 3](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/x3.png)

**图表解读**：重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 4: (b) Task-aligned head (T-Head)

![Figure 4](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/x4.png)

**图表解读**：重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 5: (c) Task-aligned predictor (TAP)

![Figure 5](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/x5.png)

**图表解读**：重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 6: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 6](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000146363_Umbrella_atss.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 7: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 7](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000317433_Horse_atss.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 8: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 8](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000337987_Bird_atss.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 9: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 9](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000364297_Cat_atss.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 10: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 10](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000134112_Dog_atss.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 11: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 11](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000521956_Person_atss.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 12: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 12](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000146363_Umbrella_tah.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 13: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 13](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000317433_Horse_tah.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 14: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 14](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000337987_Bird_tah.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 15: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 15](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000364297_Cat_tah.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 16: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 16](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000134112_Dog_tah.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 17: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 17](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000521956_Person_tah.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 18: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 18](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000146363_Umbrella_tsa.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 19: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 19](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000317433_Horse_tsa.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 20: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 20](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000337987_Bird_tsa.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 21: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 21](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000364297_Cat_tsa.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 22: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 22](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000134112_Dog_tsa.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 23: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 23](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000521956_Person_tsa.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 24: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 24](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000146363_Umbrella_tood.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 25: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 25](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000317433_Horse_tood.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 26: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 26](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000337987_Bird_tood.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 27: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 27](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000364297_Cat_tood.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 28: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 28](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000134112_Dog_tood.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### Figure 29: Figure 4: Illustration of several detection results predicted from the best anchors for classification (in red) and localization (in green). The illustrated patches and bounding boxes correspond to that in Figure 1 .

![Figure 29](https://ar5iv.labs.arxiv.org/html/2108.07755/assets/demo_image/iccv_single_image_resize/000000521956_Person_tood.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 task-aligned predictor 和 TAL 分配；它解释了分类高分点与定位高质量点为什么需要重新对齐。

### 关键表格

#### Table 1: TOOD Table 1

| Method | Head | Head/full Params (M) | Head/full FLOPs (G) | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- | --- | --- |
| FoveaBox [ 10 ] | Parallel head | 4.92/36.20 | 104.87/206.28 | 37.3 | 56.2 | 39.7 |
|  | T-head | 4.82/36.10 | 100.79/202.20 | 38.0 | 56.8 | 40.5 |
| FCOS w/ imprv [ 27 ] | Parallel head | 4.92/32.02 | 104.91/200.50 | 38.6 | 57.2 | 41.7 |
|  | T-head | 4.82/31.92 | 100.79/196.38 | 40.5 | 58.5 | 43.8 |
| ATSS (anchor-based) [ 31 ] | Parallel head | 4.92/32.07 | 104.87/205.21 | 39.3 | 57.5 | 42.8 |
|  | T-head | 4.82/31.98 | 100.79/201.13 | 41.1 | 58.6 | 44.5 |
| ATSS (anchor-free) [ 31 ] | Parallel head | 4.92/32.07 | 104.87/205.21 | 39.2 | 57.4 | 42.2 |
|  | T-head | 4.82/31.98 | 100.79/201.13 | 41.1 | 58.4 | 44.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: TOOD Table 2

| Anchor assignment | Pos/neg | Weight | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- | --- |
| IoU-based [ 16 ] | fixed | fixed | 36.5 | 55.5 | 38.7 |
| Center sampling [ 10 ] | fixed | fixed | 37.3 | 56.2 | 39.3 |
| Centerness [ 27 ] | fixed | fixed | 37.4 | 56.1 | 40.3 |
| ATSS [ 31 ] | fixed | fixed | 39.2 | 57.4 | 42.2 |
| PISA [ 1 ] | fixed | ada | 37.3 | 56.5 | 40.3 |
| NoisyAnchor [ 12 ] | fixed | ada | 38.0 | 56.9 | 40.6 |
| ATSS+QFL [ 14 ] | fixed | ada | 39.9 | 58.5 | 43.0 |
| FreeAnchor [ 32 ] | ada | fixed | 39.1 | 58.2 | 42.1 |
| MAL [ 8 ] | ada | fixed | 39.2 | 58.0 | 42.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: TOOD Table 3

| Type | Method | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- |
| Anchor-free | ATSS [ 31 ] | 39.2 | 57.4 | 42.2 |
|  | TOOD | 42.5 | 59.8 | 46.4 |
| Anchor-based | ATSS [ 31 ] | 39.3 | 57.5 | 42.8 |
|  | TOOD | 42.4 | 59.8 | 46.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: TOOD Table 4

| α 𝛼 \alpha | β 𝛽 \beta | AP | AP 50 | AP 75 |
| --- | --- | --- | --- | --- |
| 0.5 | 2 | 42.4 | 60.0 | 46.1 |
| 0.5 | 4 | 42.3 | 59.3 | 45.8 |
| 0.5 | 6 | 41.7 | 58.1 | 45.1 |
| 1.0 | 6 | 42.5 | 59.8 | 46.4 |
| 1.0 | 8 | 42.2 | 59.0 | 46.0 |
| 1.5 | 8 | 41.5 | 59.4 | 44.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: TOOD Table 5

| Method | Reference | Backbone | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RetinaNet [ 16 ] | ICCV17 | ResNet-101 | 39.1 | 59.1 | 42.3 | 21.9 | 42.7 | 50.2 |
| FoveaBox [ 10 ] | - | ResNet-101 | 40.6 | 60.1 | 43.5 | 23.3 | 45.2 | 54.5 |
| FCOS w/ imprv [ 27 ] | ICCV19 | ResNet-101 | 43.0 | 61.7 | 46.3 | 26.0 | 46.8 | 55.0 |
| Noisy Anchor [ 12 ] | CVPR20 | ResNet-101 | 41.8 | 61.1 | 44.9 | 23.4 | 44.9 | 52.9 |
| MAL [ 8 ] | CVPR20 | ResNet-101 | 43.6 | 62.8 | 47.1 | 25.0 | 46.9 | 55.8 |
| SAPD [ 35 ] | CVPR20 | ResNet-101 | 43.5 | 63.6 | 46.5 | 24.9 | 46.8 | 54.6 |
| ATSS [ 31 ] | CVPR20 | ResNet-101 | 43.6 | 62.1 | 47.4 | 26.1 | 47.0 | 53.6 |
| PAA [ 9 ] | ECCV20 | ResNet-101 | 44.8 | 63.3 | 48.7 | 26.5 | 48.8 | 56.3 |
| GFL [ 14 ] | NeurIPS20 | ResNet-101 | 45.0 | 63.7 | 48.9 | 27.2 | 48.8 | 54.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: TOOD Table 6

| Method | AP | PCC (top-50) | IoU (top-10) | #Correct boxes | #Redundant boxes | #Error boxes |
| --- | --- | --- | --- | --- | --- | --- |
| Parallel head + ATSS [ 31 ] | 39.2 | 0.408 | 0.637 | 30,261 | 25,428 | 92,677 |
| T-head + ATSS [ 31 ] | 41.1 | 0.440 | 0.644 | 30,601 | 21,838 | 79,189 |
| Parallel head + TAL | 40.3 | 0.415 | 0.643 | 30,506 | 15,927 | 72,320 |
| T-head + TAL | 42.5 | 0.452 | 0.661 | 30,734 | 15,242 | 69,013 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

One-stage object detection is commonly implemented by optimizing two sub-tasks: object classification and localization, using heads with two parallel branches, which might lead to a certain level of spatial misalignment in predictions between the two tasks. In this work, we propose a Task-aligned One-stage Object Detection (TOOD) that explicitly aligns the two tasks in a learning-based manner. First, we design a novel Task-aligned Head (T-Head) which offers a better balance between learning task

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

若 TP 分数与 IoU 质量仍错位，TAL 可作为高置信召回提升的核心改造。

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

- [[GFL]]
- [[ATSS]]
- [[YOLOX]]
- [[FCOS]]
