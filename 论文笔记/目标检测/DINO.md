---
title: "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection"
method_name: "DINO"
authors: [Hao Zhang, Feng Li, Shilong Liu, Lei Zhang]
year: 2022
venue: ICLR
tags: [object-detection, detr, denoising, query-selection]
arxiv: https://arxiv.org/abs/2203.03605
created: 2026-06-11
image_source: online
---

# 论文笔记：DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[DINO]] |
| 作者 | [Hao Zhang, Feng Li, Shilong Liu, Lei Zhang] |
| 年份 | 2022 |
| 会议/来源 | ICLR |
| 链接 | [arXiv](https://arxiv.org/abs/2203.03605) / [PDF](https://arxiv.org/pdf/2203.03605.pdf) |
| 相关工作 | [[DN-DETR]]、[[Deformable DETR]]、[[Grounding DINO]]、[[RF-DETR]] |

---

## 一句话总结

> DINO 集成 denoising、anchor refinement 和 mixed query selection，是现代高性能 DETR 的关键代表。

---

## 核心贡献

1. **范式贡献**：DINO 集成 denoising、anchor refinement 和 mixed query selection，是现代高性能 DETR 的关键代表。
2. **方法贡献**：集成 denoising、mixed query selection、anchor refinement 等机制，形成强 DETR 训练范式。
3. **工程/研究影响**：该工作成为后续 [[DN-DETR]]、[[Deformable DETR]]、[[Grounding DINO]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

集成 denoising、mixed query selection、anchor refinement 等机制，形成强 DETR 训练范式。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[DN-DETR]]、[[Deformable DETR]]。
- 后续影响：[[Grounding DINO]]、[[RF-DETR]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\mathcal{L}_{DINO}=\mathcal{L}_{match}+\mathcal{L}_{dn}+\mathcal{L}_{aux}
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2203.03605](https://ar5iv.labs.arxiv.org/html/2203.03605)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 12 张、Table 8 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: AP on COCO compared with other detection models. (a) Comparison to models with a ResNet-50 backbone w.r.t. training epochs. Models marked with DC5 use a dilated larger resolution feature map. Other models use multi-scale features. (b) Comparison to SOTA models w.r.t. pre-training data size and model size. SOTA models are from the COCO test-dev leaderboard. In the legend we list the backbone pre-training data size (first number) and detection pre-training data size (second number). ∗ * means the data size is not disclosed.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/images/ap_r50_alllllll.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 2: Figure 1: AP on COCO compared with other detection models. (a) Comparison to models with a ResNet-50 backbone w.r.t. training epochs. Models marked with DC5 use a dilated larger resolution feature map. Other models use multi-scale features. (b) Comparison to SOTA models w.r.t. pre-training data size and model size. SOTA models are from the COCO test-dev leaderboard. In the legend we list the backbone pre-training data size (first number) and detection pre-training data size (second number). ∗ * means the data size is not disclosed.

![Figure 2](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/x1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 3: Figure 2: The framework of our proposed DINO model. Our improvements are mainly in the Transformer encoder and decoder. The top-K encoder features in the last layer are selected to initialize the positional queries for the Transformer decoder, whereas the content queries are kept as learnable parameters. Our decoder also contains a Contrastive DeNoising (CDN) part with both positive and negative samples.

![Figure 3](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 4: Figure 3: The structure of CDN group and a demonstration of positive and negative examples. Although both positive and negative examples are 4 4 4 D anchors that can be represented as points in 4 4 4 D space, we illustrate them as points in 2 2 2 D space on concentric squares for simplicity. Assuming the square center is a GT box, points inside the inner square are regarded as a positive example and points between the inner square and the outer square are viewed as negative examples.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/x3.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 5: Figure 4: (a) and (b) A ​ T ​ D ​ ( 100 ) 𝐴 𝑇 𝐷 100 ATD(100) on all objects and small objects respectively. (c) The AP on small objects.

![Figure 5](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/x4.png)

**图表解读**：重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 6: Figure 4: (a) and (b) A ​ T ​ D ​ ( 100 ) 𝐴 𝑇 𝐷 100 ATD(100) on all objects and small objects respectively. (c) The AP on small objects.

![Figure 6](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/x5.png)

**图表解读**：重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 7: Figure 4: (a) and (b) A ​ T ​ D ​ ( 100 ) 𝐴 𝑇 𝐷 100 ATD(100) on all objects and small objects respectively. (c) The AP on small objects.

![Figure 7](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/x6.png)

**图表解读**：重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 8: Figure 5: Comparison of three different query initialization methods. The term “static” means that they will keep the same for different images in inference. A common implementation for these static queries is to make them learnable.

![Figure 8](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/x7.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 9: Figure 6: Comparison of box update in Deformable DETR and our method.

![Figure 9](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/x8.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 10: Figure 7: Training convergence curves evaluated on COCO val2017 for DINO and two previous state-of-the-art models with ResNet-50 using multi-scale features.

![Figure 10](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/images/convergence_comp_eccv2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 11: Figure 8: The left figure is the detection result using a model trained with DN queries and the right is the result of our method. In the left image, the boy pointed by the arrow has 3 3 3 duplicate bounding boxes. For clarity, we only show boxes of class “person”.

![Figure 11](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/images/img1000_0.3_dn.jpg)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### Figure 12: Figure 8: The left figure is the detection result using a model trained with DN queries and the right is the result of our method. In the left image, the boy pointed by the arrow has 3 3 3 duplicate bounding boxes. For clarity, we only show boxes of class “person”.

![Figure 12](https://ar5iv.labs.arxiv.org/html/2203.03605/assets/images/img1000_0.3_exc.jpg)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 mixed query selection、denoising 和 iterative box refinement 的组合；这是现代 DETR head 的强训练范式。

### 关键表格

#### Table 1: DINO Table 1

| Model | Epochs | AP | AP 50 | AP 75 | AP S | AP M | AP L | GFLOPS | Params | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Faster-RCNN(5scale) [ 30 ] | 12 12 12 | 37.9 37.9 37.9 | 58.8 58.8 58.8 | 41.1 41.1 41.1 | 22.4 22.4 22.4 | 41.1 41.1 41.1 | 49.1 49.1 49.1 | 207 207 207 | 40 40 40 M | 21 ∗ superscript 21 21^{*} |
| DETR(DC5) [ 3 ] | 12 12 12 | 15.5 15.5 15.5 | 29.4 29.4 29.4 | 14.5 14.5 14.5 | 4.3 4.3 4.3 | 15.1 15.1 15.1 | 26.7 26.7 26.7 | 225 225 225 | 41 41 41 M | 20 20 20 |
| Deformable DETR(4scale) [ 41 ] | 12 12 12 | 41.1 41.1 41.1 | − - | − - | − - | − - |  | 196 196 196 | 40 40 40 M | 24 |
| DAB-DETR(DC5) † [ 21 ] | 12 12 12 | 38.0 38.0 38.0 | 60.3 60.3 60.3 | 39.8 39.8 39.8 | 19.2 19.2 19.2 | 40.9 40.9 40.9 | 55.4 55.4 55.4 | 256 256 256 | 44 44 44 M | 17 17 17 |
| Dynamic DETR(5scale) [ 8 ] | 12 12 12 | 42.9 42.9 42.9 | 61.0 61.0 61.0 | 46.3 46.3 46.3 | 24.6 24.6 24.6 | 44.9 44.9 44.9 | 54.4 54.4 54.4 | − - | 58 58 58 M | − - |
| Dynamic Head(5scale) [ 7 ] | 12 12 12 | 43.0 43.0 43.0 | 60.7 60.7 60.7 | 46.8 46.8 46.8 | 24.7 24.7 24.7 | 46.4 46.4 46.4 | 53.9 53.9 53.9 | − - | − - | − - |
| HTC(5scale) [ 4 ] | 12 12 12 | 42.3 42.3 42.3 | − - | − - | − - | − - | − - | 441 441 441 | 80 80 80 M | 5 ∗ superscript 5 5^{*} |
| DN-Deformable-DETR(4scale) † [ 17 ] | 12 12 12 | 43.4 43.4 43.4 | 61.9 61.9 61.9 | 47.2 47.2 47.2 | 24.8 24.8 24.8 | 46.8 46.8 46.8 | 59.4 59.4 59.4 | 265 265 265 | 48 48 48 M | 23 23 23 |
| DINO-4scale † | 12 | 49.0 ( + 5.6 ) 5.6 {(+5.6)} | 66.6 | 53.5 | 32.0 ( + 7.2 ) 7.2 {(+7.2)} | 52.3 | 63.0 | 279 279 279 | 47 47 47 M | 24 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: DINO Table 2

| Model | Epochs | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Faster-RCNN [ 30 ] | 108 108 108 | 42.0 42.0 42.0 | 62.4 62.4 62.4 | 44.2 44.2 44.2 | 20.5 20.5 20.5 | 45.8 45.8 45.8 | 61.1 61.1 61.1 |
| DETR(DC5) [ 41 ] | 500 500 500 | 43.3 43.3 43.3 | 63.1 63.1 63.1 | 45.9 45.9 45.9 | 22.5 22.5 22.5 | 47.3 47.3 47.3 | 61.1 61.1 61.1 |
| Deformable DETR [ 41 ] | 50 50 50 | 46.2 46.2 46.2 | 65.2 65.2 65.2 | 50.0 50.0 50.0 | 28.8 28.8 28.8 | 49.2 49.2 49.2 | 61.7 61.7 61.7 |
| SMCA-R [ 11 ] | 50 50 50 | 43.7 43.7 43.7 | 63.6 63.6 63.6 | 47.2 47.2 47.2 | 24.2 24.2 24.2 | 47.0 47.0 47.0 | 60.4 60.4 60.4 |
| TSP-RCNN-R [ 34 ] | 96 96 96 | 45.0 45.0 45.0 | 64.5 64.5 64.5 | 49.6 49.6 49.6 | 29.7 29.7 29.7 | 47.7 47.7 47.7 | 58.0 58.0 58.0 |
| Dynamic DETR(5scale) [ 7 ] | 50 50 50 | 47.2 47.2 {47.2} | 65.9 65.9 65.9 | 51.1 51.1 51.1 | 28.6 28.6 28.6 | 49.3 49.3 49.3 | 59.1 59.1 59.1 |
| DAB-Deformable-DETR [ 21 ] | 50 50 50 | 46.9 46.9 46.9 | 66.0 66.0 66.0 | 50.8 50.8 50.8 | 30.1 30.1 30.1 | 50.4 50.4 50.4 | 62.5 62.5 62.5 |
| DN-Deformable-DETR [ 17 ] | 50 | 48.6 | 67.4 67.4 67.4 | 52.7 52.7 52.7 | 31.0 31.0 31.0 | 52.0 52.0 52.0 | 63.7 63.7 63.7 |
| DINO-4scale | 24 | 50.4 (+1.8) | 68.3 68.3 68.3 | 54.8 54.8 54.8 | 33.3 33.3 33.3 | 53.7 53.7 53.7 | 64.8 64.8 64.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: DINO Table 3

| Method | Params | Backbone Pre-training Dataset | Detection Pre-training Dataset | Use Mask | End-to-end | val2017 (AP) | test-dev (AP) |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w/o TTA | w/ TTA | w/o TTA | w/ TTA |  |  |  |  |  |  |
| SwinL [ 23 ] | 284 284 284 M | IN-22K-14M | O365 | ✓ |  | 57.1 57.1 57.1 | 58.0 58.0 58.0 | 57.7 57.7 57.7 | 58.7 58.7 58.7 |
| DyHead [ 7 ] | ≥ 284 absent 284 \geq 284 M | IN-22K-14M | Unknown* |  |  | − - | 58.4 58.4 58.4 | − - | 60.6 60.6 60.6 |
| Soft Teacher+SwinL [ 38 ] | 284 284 284 M | IN-22K-14M | O365 | ✓ |  | 60.1 60.1 60.1 | 60.7 60.7 60.7 | − - | 61.3 61.3 61.3 |
| GLIP [ 18 ] | ≥ 284 absent 284 \geq 284 M | IN-22K-14M | FourODs [ 18 ] ,GoldG+ [ 18 , 15 ] |  |  | − - | 60.8 60.8 60.8 | − - | 61.5 61.5 61.5 |
| Florence-CoSwin-H [ 40 ] | ≥ 637 absent 637 \geq 637 M | FLD-900M [ 40 ] | FLD-9M [ 40 ] |  |  | − - | 62.0 62.0 62.0 | − - | 62.4 62.4 62.4 |
| SwinV2-G [ 22 ] | 3.0 3.0 3.0 B | IN-22K-ext-70M [ 22 ] | O365 | ✓ |  | 61.9 61.9 61.9 | 62.5 62.5 62.5 | − - | 63.1 63.1 63.1 |
| DINO-SwinL(Ours) | 218 M | IN-22K-14M | O365 |  | ✓ | 63.1 | 63.2 | 63.2 | 63.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: DINO Table 4

| #Row | QS | CDN | LFT | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. DN-DETR [ 17 ] | No |  |  | 43.4 43.4 43.4 | 61.9 61.9 61.9 | 47.2 47.2 47.2 | 24.8 24.8 24.8 | 46.8 46.8 46.8 | 59.4 59.4 59.4 |
| 2. Optimized DN-DETR | No |  |  | 44.9 44.9 44.9 | 62.8 62.8 62.8 | 48.6 48.6 48.6 | 26.9 26.9 26.9 | 48.2 48.2 48.2 | 60.0 60.0 60.0 |
| 3. Strong baseline (Row2+pure query selection) | Pure |  |  | 46.5 46.5 46.5 | 64.2 64.2 64.2 | 50.4 50.4 50.4 | 29.6 29.6 29.6 | 49.8 49.8 49.8 | 61.0 61.0 61.0 |
| 4. Row3+mixed query selection | Mixed |  |  | 47.0 47.0 47.0 | 64.2 64.2 64.2 | 51.0 51.0 51.0 | 31.1 31.1 31.1 | 50.1 50.1 50.1 | 61.5 61.5 61.5 |
| 5. Row4+look forward twice | Mixed |  | ✓ | 47.4 47.4 47.4 | 64.8 64.8 64.8 | 51.6 51.6 51.6 | 29.9 29.9 29.9 | 50.8 50.8 50.8 | 61.9 61.9 61.9 |
| 6. DINO (ours, Row5+contrastive DN) | Mixed | ✓ | ✓ | 47.9 | 65.3 | 52.1 | 31.2 | 50.9 | 61.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: DINO Table 5

| Model | Batch Size per GPU | Traning Time | GPU Mem. | Epoch | AP |
| --- | --- | --- | --- | --- | --- |
| Faster RCNN [ 30 ] * | 8 | ∼ 60 similar-to absent 60 \sim 60 min/ep | 13 13 13 GB | 108 108 108 | 42.0 42.0 42.0 |
| DETR [ 3 ] | 8 | ∼ 16 similar-to absent 16 \sim 16 min/ep | 26 26 26 GB | 300 300 300 | 41.2 41.2 41.2 |
| Deformable DETR [ 41 ] ⋆ | 2 | ∼ 55 similar-to absent 55 \sim 55 min/ep | 16 16 16 GB | 50 50 50 | 45.4 45.4 45.4 |
| DINO(Ours) | 2 | ∼ 55 similar-to absent 55 \sim 55 min/ep | 16 16 16 GB | 12 | 47.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: DINO Table 6

| # Encoder/Decoder | 6 / 6 6 6 6/6 | 4 / 6 4 6 4/6 | 3 / 6 3 6 3/6 | 2 / 6 2 6 2/6 | 6/4 | 6/2 | 2 / 4 2 4 2/4 | 2 / 2 2 2 2/2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP | 47.4 47.4 47.4 | 46.2 46.2 46.2 | 45.8 45.8 45.8 | 45.4 45.4 45.4 | 46.0 46.0 46.0 | 44.4 44.4 44.4 | 44.1 44.1 44.1 | 41.2 41.2 41.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: DINO Table 7

| # Denoising query | 100 CDN | 1000 DN | 200 DN | 100 DN | 50 DN | 10 DN | No DN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP | 47.9 47.9 47.9 | 47.6 47.6 47.6 | 47.4 47.4 47.4 | 47.4 47.4 47.4 | 46.7 46.7 46.7 | 46.0 46.0 46.0 | 45.1 45.1 45.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: DINO Table 8

| Item | Value |
| --- | --- |
| lr | 0.0001 |
| lr_backbone | 1e-05 |
| weight_decay | 0.0001 |
| clip_max_norm | 0.1 |
| pe_temperature | 20 |
| enc_layers | 6 |
| dec_layers | 6 |
| dim_feedforward | 2048 |
| hidden_dim | 256 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

We present DINO (\textbf{D}ETR with \textbf{I}mproved de\textbf{N}oising anch\textbf{O}r boxes), a state-of-the-art end-to-end object detector. % in this paper. DINO improves over previous DETR-like models in performance and efficiency by using a contrastive way for denoising training, a mixed query selection method for anchor initialization, and a look forward twice scheme for box prediction. DINO achieves $49.4$AP in $12$ epochs and $51.3$AP in $24$ epochs on COCO with a ResNet-50 backbone and

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

- [[DN-DETR]]
- [[Deformable DETR]]
- [[Grounding DINO]]
- [[RF-DETR]]
