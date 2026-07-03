---
title: "End-to-End Object Detection with Transformers"
method_name: "DETR"
authors: [Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko]
year: 2020
venue: ECCV
tags: [object-detection, transformer, detr, set-prediction]
arxiv: https://arxiv.org/abs/2005.12872
created: 2026-06-11
image_source: online
---

# 论文笔记：End-to-End Object Detection with Transformers

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[DETR]] |
| 作者 | [Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko] |
| 年份 | 2020 |
| 会议/来源 | ECCV |
| 链接 | [arXiv](https://arxiv.org/abs/2005.12872) / [PDF](https://arxiv.org/pdf/2005.12872.pdf) |
| 相关工作 | [[Deformable DETR]]、[[DN-DETR]]、[[DINO]]、[[RT-DETR]] |

---

## 一句话总结

> DETR 将目标检测表述为 set prediction，用 object queries 和 Hungarian matching 实现端到端检测，去掉 anchor 和 NMS。

---

## 核心贡献

1. **范式贡献**：DETR 将目标检测表述为 set prediction，用 object queries 和 Hungarian matching 实现端到端检测，去掉 anchor 和 NMS。
2. **方法贡献**：用 object queries 和 Transformer decoder 直接输出固定大小预测集合，再用 Hungarian matching 做一对一监督。
3. **工程/研究影响**：该工作成为后续 [[Deformable DETR]]、[[DN-DETR]]、[[DINO]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

用 object queries 和 Transformer decoder 直接输出固定大小预测集合，再用 Hungarian matching 做一对一监督。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[Deformable DETR]]、[[DN-DETR]]。
- 后续影响：[[DINO]]、[[RT-DETR]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\hat{\sigma}=\arg\min_\sigma\sum_i\mathcal{L}_{match}(y_i,\hat{y}_{\sigma(i)})
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2005.12872](https://ar5iv.labs.arxiv.org/html/2005.12872)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 20 张、Table 5 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: DETR directly predicts (in parallel) the final set of detections by combining a common CNN with a transformer architecture. During training, bipartite matching uniquely assigns predictions with ground truth boxes. Prediction with no match should yield a “ no object ” ( ∅ \varnothing ) class prediction.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/x1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 2: Figure 2: DETR uses a conventional CNN backbone to learn a 2D representation of an input image. The model flattens it and supplements it with a positional encoding before passing it into a transformer encoder. A transformer decoder then takes as input a small fixed number of learned positional embeddings, which we call object queries , and additionally attends to the encoder output. We pass each output embedding of the decoder to a shared feed forward network (FFN) that predicts either a detection (class and bounding box) or a “ no object ” class.

![Figure 2](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 3: Figure 3: Encoder self-attention for a set of reference points. The encoder is able to separate individual instances. Predictions are made with baseline DETR model on a validation set image.

![Figure 3](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/x3.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 4: Figure 4: AP and AP 50 performance after each decoder layer. A single long schedule baseline model is evaluated. DETR does not need NMS by design, which is validated by this figure. NMS lowers AP in the final layers, removing TP predictions, but improves AP in the first decoder layers, removing double predictions, as there is no communication in the first layer, and slightly improves AP 50 .

![Figure 4](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/x4.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 5: Figure 5: Out of distribution generalization for rare classes. Even though no image in the training set has more than 13 giraffes, DETR has no difficulty generalizing to 24 and more instances of the same class.

![Figure 5](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/giraffe_collage2.jpg)

**图表解读**：重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 6: Figure 6: Visualizing decoder attention for every predicted object (images from COCO val set). Predictions are made with DETR-DC5 model. Attention scores are coded with different colors for different objects. Decoder typically attends to object extremities, such as legs and heads. Best viewed in color.

![Figure 6](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/elephants.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 7: Figure 6: Visualizing decoder attention for every predicted object (images from COCO val set). Predictions are made with DETR-DC5 model. Attention scores are coded with different colors for different objects. Decoder typically attends to object extremities, such as legs and heads. Best viewed in color.

![Figure 7](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/zebras.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 8: Figure 7 : Visualization of all box predictions on all images from COCO 2017 val set for 20 out of total N = 100 𝑁 100 N=100 prediction slots in DETR decoder. Each box prediction is represented as a point with the coordinates of its center in the 1-by-1 square normalized by each image size. The points are color-coded so that green color corresponds to small boxes, red to large horizontal boxes and blue to large vertical boxes. We observe that each slot learns to specialize on certain areas and box sizes with several operating modes. We note that almost all slots have a mode of predicting large image-wide boxes that are common in COCO dataset.

![Figure 8](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/query_distr.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 9: Figure 8: Illustration of the panoptic head. A binary mask is generated in parallel for each detected object, then the masks are merged using pixel-wise argmax.

![Figure 9](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/x5.png)

**图表解读**：重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 10: Figure 9: Qualitative results for panoptic segmentation generated by DETR-R101. DETR produces aligned mask predictions in a unified manner for things and stuff.

![Figure 10](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/Panoptic/0784_detr_R101.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 11: Figure 9: Qualitative results for panoptic segmentation generated by DETR-R101. DETR produces aligned mask predictions in a unified manner for things and stuff.

![Figure 11](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/Panoptic/3499_detr_R101.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 12: Figure 9: Qualitative results for panoptic segmentation generated by DETR-R101. DETR produces aligned mask predictions in a unified manner for things and stuff.

![Figure 12](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/Panoptic/0673_detr_R101.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 13: Figure 10: Architecture of DETR’s transformer. Please, see Section 0.A.3 for details.

![Figure 13](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/x6.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 14: (a) Failure case with overlapping objects. PanopticFPN misses one plane entirely, while DETR fails to accurately segment 3 of them.

![Figure 14](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/Panoptic/1113_gt.jpg)

**图表解读**：重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 15: (a) Failure case with overlapping objects. PanopticFPN misses one plane entirely, while DETR fails to accurately segment 3 of them.

![Figure 15](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/Panoptic/1113_panopticfpn.jpg)

**图表解读**：重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 16: (a) Failure case with overlapping objects. PanopticFPN misses one plane entirely, while DETR fails to accurately segment 3 of them.

![Figure 16](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/Panoptic/1113_detr_R101.jpg)

**图表解读**：重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 17: (b) Things masks are predicted at full resolution, which allows sharper boundaries than PanopticFPN

![Figure 17](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/Panoptic/3485_gt.jpg)

**图表解读**：重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 18: (b) Things masks are predicted at full resolution, which allows sharper boundaries than PanopticFPN

![Figure 18](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/Panoptic/3485_panopticfpn.jpg)

**图表解读**：重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 19: (b) Things masks are predicted at full resolution, which allows sharper boundaries than PanopticFPN

![Figure 19](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/figures/Panoptic/3485_detr_R101.jpg)

**图表解读**：重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### Figure 20: Figure 12: Analysis of the number of instances of various classes missed by DETR depending on how many are present in the image. We report the mean and the standard deviation. As the number of instances gets close to 100, DETR starts saturating and misses more and more objects

![Figure 20](https://ar5iv.labs.arxiv.org/html/2005.12872/assets/x7.png)

**图表解读**：这是消融/分析图，重点看每个模块独立贡献，避免把数据增强、backbone 或训练轮数带来的收益误认为核心方法收益。重点看 object queries、encoder-decoder 和 Hungarian matching；对比旧 DETR 失败实验时要关注 query 初始化与收敛问题。

### 关键表格

#### Table 1: DETR Table 1

| Model | GFLOPS/FPS | #params | AP | AP 50 | AP 75 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Faster RCNN-DC5 | 320/16 | 166M | 39.0 | 60.5 | 42.3 | 21.4 | 43.5 | 52.5 |
| Faster RCNN-FPN | 180/26 | 42M | 40.2 | 61.0 | 43.8 | 24.2 | 43.5 | 52.0 |
| Faster RCNN-R101-FPN | 246/20 | 60M | 42.0 | 62.5 | 45.9 | 25.2 | 45.6 | 54.6 |
| Faster RCNN-DC5+ | 320/16 | 166M | 41.1 | 61.4 | 44.3 | 22.9 | 45.9 | 55.0 |
| Faster RCNN-FPN+ | 180/26 | 42M | 42.0 | 62.1 | 45.5 | 26.6 | 45.4 | 53.4 |
| Faster RCNN-R101-FPN+ | 246/20 | 60M | 44.0 | 63.9 | 47.8 | 27.2 | 48.1 | 56.0 |
| DETR | 86/28 | 41M | 42.0 | 62.4 | 44.2 | 20.5 | 45.8 | 61.1 |
| DETR-DC5 | 187/12 | 41M | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1 |
| DETR-R101 | 152/20 | 60M | 43.5 | 63.8 | 46.4 | 21.9 | 48.0 | 61.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: DETR Table 2

| #layers | GFLOPS/FPS | #params | AP | AP 50 | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 76/28 | 33.4M | 36.7 | 57.4 | 16.8 | 39.6 | 54.2 |
| 3 | 81/25 | 37.4M | 40.1 | 60.6 | 18.5 | 43.8 | 58.6 |
| 6 | 86/23 | 41.3M | 40.6 | 61.6 | 19.9 | 44.3 | 60.2 |
| 12 | 95/20 | 49.2M | 41.6 | 62.1 | 19.8 | 44.9 | 61.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: DETR Table 3

| spatial pos. enc. | output pos. enc. |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| encoder | decoder | decoder | AP | Δ Δ \Delta | AP 50 | Δ Δ \Delta |
| none | none | learned at input | 32.8 | -7.8 | 55.2 | -6.5 |
| sine at input | sine at input | learned at input | 39.2 | -1.4 | 60.0 | -1.6 |
| learned at attn. | learned at attn. | learned at attn. | 39.6 | -1.0 | 60.7 | -0.9 |
| none | sine at attn. | learned at attn. | 39.3 | -1.3 | 60.3 | -1.4 |
| sine at attn. | sine at attn. | learned at attn. | 40.6 | - | 61.6 | - |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: DETR Table 4

| class | ℓ 1 subscript ℓ 1 \ell_{1} | GIoU | AP | Δ Δ \Delta | AP 50 | Δ Δ \Delta | AP S | AP M | AP L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ |  | 35.8 | -4.8 | 57.3 | -4.4 | 13.7 | 39.8 | 57.9 |
| ✓ |  | ✓ | 39.9 | -0.7 | 61.6 | 0 | 19.9 | 43.2 | 57.9 |
| ✓ | ✓ | ✓ | 40.6 | - | 61.6 | - | 19.9 | 44.3 | 60.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: DETR Table 5

| Model | Backbone | PQ | SQ | RQ | PQ th superscript PQ th \text{PQ}^{\text{th}} | SQ th superscript SQ th \text{SQ}^{\text{th}} | RQ th superscript RQ th \text{RQ}^{\text{th}} | PQ st superscript PQ st \text{PQ}^{\text{st}} | SQ st superscript SQ st \text{SQ}^{\text{st}} | RQ st superscript RQ st \text{RQ}^{\text{st}} | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PanopticFPN++ | R50 | 42.4 | 79.3 | 51.6 | 49.2 | 82.4 | 58.8 | 32.3 | 74.8 | 40.6 | 37.7 |
| UPSnet | R50 | 42.5 | 78.0 | 52.5 | 48.6 | 79.4 | 59.6 | 33.4 | 75.9 | 41.7 | 34.3 |
| UPSnet-M | R50 | 43.0 | 79.1 | 52.8 | 48.9 | 79.7 | 59.7 | 34.1 | 78.2 | 42.3 | 34.3 |
| PanopticFPN++ | R101 | 44.1 | 79.5 | 53.3 | 51.0 | 83.2 | 60.6 | 33.6 | 74.0 | 42.1 | 39.7 |
| DETR | R50 | 43.4 | 79.3 | 53.8 | 48.2 | 79.8 | 59.5 | 36.3 | 78.5 | 45.3 | 31.1 |
| DETR-DC5 | R50 | 44.6 | 79.8 | 55.0 | 49.4 | 80.5 | 60.6 | 37.3 | 78.7 | 46.5 | 31.9 |
| DETR-R101 | R101 | 45.1 | 79.9 | 55.5 | 50.5 | 80.9 | 61.7 | 37.0 | 78.5 | 46.0 | 33.0 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transfor

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

旧 DETR 头失败不是偶然；只有引入现代 query selection/denoising/deformable attention 才值得重试。

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

- [[Deformable DETR]]
- [[DN-DETR]]
- [[DINO]]
- [[RT-DETR]]
