---
title: "Scaling Open-Vocabulary Object Detection"
method_name: "OWLv2"
authors: [Matthias Minderer, et al.]
year: 2023
venue: NeurIPS
tags: [open-vocabulary-detection, self-training, vision-language, owl-vit]
arxiv: https://arxiv.org/abs/2306.09683
created: 2026-06-11
image_source: online
---

# 论文笔记：Scaling Open-Vocabulary Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[OWLv2]] |
| 作者 | [Matthias Minderer, et al.] |
| 年份 | 2023 |
| 会议/来源 | NeurIPS |
| 链接 | 无 arXiv 链接 / 经典论文 |
| 相关工作 | [[GLIP]]、[[Detic]]、[[YOLO-World]] |

---

## 一句话总结

> OWLv2 通过 web-scale self-training 扩展开放词汇检测能力，是 OWL-ViT 系列的重要升级。

---

## 核心贡献

1. **范式贡献**：OWLv2 通过 web-scale self-training 扩展开放词汇检测能力，是 OWL-ViT 系列的重要升级。
2. **方法贡献**：用 web-scale self-training 扩展 OWL-ViT 开放词汇检测，依赖 teacher 生成伪框并训练 student。
3. **工程/研究影响**：该工作成为后续 [[GLIP]]、[[Detic]]、[[YOLO-World]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

用 web-scale self-training 扩展 OWL-ViT 开放词汇检测，依赖 teacher 生成伪框并训练 student。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[GLIP]]、[[Detic]]。
- 后续影响：[[YOLO-World]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\mathcal{D}_{pseudo}=\{(x,\hat{b},\hat{y})\mid score>\tau\}
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2306.09683](https://ar5iv.labs.arxiv.org/html/2306.09683)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 15 张、Table 4 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: Overview of our method. Left: Our method consists of three steps: (1) Generate pseudo-box annotations on WebLI with OWL-ViT L/14, queried with caption N-grams. (2) Train new models on pseudo-annotations. (3) Optionally, fine-tune on human annotations. Right: Zero-shot detection performance on LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} after fine-tuning on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} . Neither the annotator nor our models have seen any human-generated box annotations for LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} classes. Our self-training approach improves over other methods even at moderate amounts of training (e.g. the OWL-L/14 model we use as annotator; black × \times ), and continues to improve as training is scaled up. Horizontal black lines indicate previous state-of-the-art open-vocabulary detectors which did not see LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} classes during training.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 2: Figure 1: Overview of our method. Left: Our method consists of three steps: (1) Generate pseudo-box annotations on WebLI with OWL-ViT L/14, queried with caption N-grams. (2) Train new models on pseudo-annotations. (3) Optionally, fine-tune on human annotations. Right: Zero-shot detection performance on LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} after fine-tuning on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} . Neither the annotator nor our models have seen any human-generated box annotations for LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} classes. Our self-training approach improves over other methods even at moderate amounts of training (e.g. the OWL-L/14 model we use as annotator; black × \times ), and continues to improve as training is scaled up. Horizontal black lines indicate previous state-of-the-art open-vocabulary detectors which did not see LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} classes during training.

![Figure 2](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x2.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 3: Figure 2: Comparison of pseudo-label spaces. Self-training on a human-curated list of classes yields good downstream performance on these classes, but generalizes poorly to unseen classes and datasets. Open-vocabulary generalization can be improved by obtaining weak but diverse supervision from image-associated text. WebLI image-text data was pseudo-annotated using OWL-ViT CLIP-L/14 with one of three label spaces: Curated vocabulary (the union of label spaces from LVIS, Objects365, OpenImagesv4, and Visual Genome), N-grams (lightly filtered N-grams from the text associated with each image), or a combination of both ( N-grams + curated ). OWLv2-B/16 models were then self-trained on the pseudo-annotations and fine-tuned on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} . Each point represents a separate fine-tuning run. “Examples seen” refers to the number of images after creating mosaics; the total number of raw images seen is 13.2 × 13.2\times that number ( Section 3.2 ).

![Figure 3](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x3.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 4: Figure 3: Impact of pseudo-annotation filtering by detection confidence on self-training effectiveness. Pseudo-labels (N-gram label space) were filtered using different confidence thresholds. Number of remaining images for each threshold: 0.1: 5B, 0.3: 2B, 0.5: 782M, 0.7: 224M. OWLv2-B/16 detectors were self-trained on the filtered pseudo-annotations and fine-tuned on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} . Each point represents a different fine-tuning run. “Examples seen” refers to the number of images after creating mosaics; the total number of raw images seen is 13.2 × 13.2\times that number ( Section 3.2 ).

![Figure 4](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x4.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 5: Figure 4: Scaling of detection performance with model size and training compute. Models show classic scaling behavior [ 36 ] : Performance increases monotonically with training compute, with larger models being necessary to benefit from larger amounts of compute/data. Models were self-trained on N-gram pseudo-annotations and fine-tuned on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} .

![Figure 5](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x5.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 6: Figure 5: Trade-off between fine-tuned and open-world performance. Self-training yields continued improvements on a suite of diverse datasets (ODinW13; x 𝑥 x -axis), but performance on any given dataset (e.g. LVIS; y 𝑦 y -axis) may saturate ( red circles). Fine-tuning on a target dataset improves performance on that dataset, but reduces the open-world generalization ability in proportion to the finetuning duration ( light blue squares; numbers indicate finetuning steps). This trade-off can be improved through weight-space ensembling (averaging) of the pretrained and fine-tuned checkpoints [ 31 ] ( purple diamonds; numbers indicate the mixing coefficient for the fine-tuned weights). The plot shows B/16 models self-trained on N-gram pseudo-annotations and evaluated either directly after self-training or after fine-tuning on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} . Ensembles were created between the longest-self-trained checkpoint and the weights obtained after finetuning that checkpoint for 20k steps. Note that there is significant variability in ODinW13 performance between checkpoints towards the end of self-training.

![Figure 6](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x6.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 7: Figure A1: Trade-off between fine-tuned and open-world performance. Similar to Figure 5 , but for OWLv2 L/14.

![Figure 7](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x7.png)

**图表解读**：重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 8: Figure A2: Example pseudo-annotations on WebLI [ 3 ] . Image-associated text (from the HTML alt_text tag) is shown above the images. If the text is not in English, an automatically generated translation is used. N-grams are extracted from these texts to generate queries for the annotator model. Pseudo-annotations were filtered as for our main experiments: To be included, boxes must have a score of at least 0.1, and images must have at least one box with a score above 0.3. All images from Wikimedia Commons.

![Figure 8](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x8.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 9: Figure A3: Training inputs after pre-processing. Top: A 4 × 4 4 4 4\times 4 mosaic of randomly resized and padded images as used for self-training. Bottom: The same mosaic after dropping the 50% of patches with lowest pixel variance (image size: 1008 × 1008 1008 1008 1008\times 1008 ; patch size: 14 × 14 14 14 14\times 14 ). Most dropped patches belong to padding areas or uniform image backgrounds. All images from Wikimedia Commons.

![Figure 9](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x9.png)

**图表解读**：重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 10: Figure A4: Qualitative example for OWLv2 L/14 from the LVIS val set. For the visualization, all LVIS classes were used as prompts. LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} classes are labeled in black. Top: OWL-ST self-trained on N-grams, not fine-tuned ( Table 1 row 12). Bottom: OWL-ST+FT self-trained on N-grams and fine-tuned on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} ( Table 1 row 15). Boxes above score 0.08 (top) or 0.3 (bottom) are shown.

![Figure 10](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x10.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 11: Figure A4: Qualitative example for OWLv2 L/14 from the LVIS val set. For the visualization, all LVIS classes were used as prompts. LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} classes are labeled in black. Top: OWL-ST self-trained on N-grams, not fine-tuned ( Table 1 row 12). Bottom: OWL-ST+FT self-trained on N-grams and fine-tuned on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} ( Table 1 row 15). Boxes above score 0.08 (top) or 0.3 (bottom) are shown.

![Figure 11](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x11.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 12: Figure A5: Qualitative example for OWLv2 L/14 from the LVIS val set. For the visualization, all LVIS classes were used as prompts. LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} classes are labeled in black. Top: OWL-ST self-trained on N-grams, not fine-tuned ( Table 1 row 12). Bottom: OWL-ST+FT self-trained on N-grams and fine-tuned on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} ( Table 1 row 15). Boxes above score 0.08 (top) or 0.3 (bottom) are shown.

![Figure 12](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x12.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 13: Figure A5: Qualitative example for OWLv2 L/14 from the LVIS val set. For the visualization, all LVIS classes were used as prompts. LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} classes are labeled in black. Top: OWL-ST self-trained on N-grams, not fine-tuned ( Table 1 row 12). Bottom: OWL-ST+FT self-trained on N-grams and fine-tuned on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} ( Table 1 row 15). Boxes above score 0.08 (top) or 0.3 (bottom) are shown.

![Figure 13](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x13.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 14: Figure A6: Qualitative example for OWLv2 L/14 from the LVIS val set. For the visualization, all LVIS classes were used as prompts. LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} classes are labeled in black. Top: OWL-ST self-trained on N-grams, not fine-tuned ( Table 1 row 12). Bottom: OWL-ST+FT self-trained on N-grams and fine-tuned on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} ( Table 1 row 15). Boxes above score 0.08 (top) or 0.3 (bottom) are shown.

![Figure 14](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x14.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### Figure 15: Figure A6: Qualitative example for OWLv2 L/14 from the LVIS val set. For the visualization, all LVIS classes were used as prompts. LVIS rare subscript LVIS rare \text{LVIS}_{\text{rare}} classes are labeled in black. Top: OWL-ST self-trained on N-grams, not fine-tuned ( Table 1 row 12). Bottom: OWL-ST+FT self-trained on N-grams and fine-tuned on LVIS base subscript LVIS base \text{LVIS}_{\text{base}} ( Table 1 row 15). Boxes above score 0.08 (top) or 0.3 (bottom) are shown.

![Figure 15](https://ar5iv.labs.arxiv.org/html/2306.09683/assets/x15.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 self-training pipeline 和 open-vocabulary 扩展；它启发用 teacher 生成 OOD 类别框。

### 关键表格

#### Table 1: OWLv2 Table 1

|  | Method | Backbone | Self-training data | Self-training vocabulary | Human box annotations | ODinW 13 | LVIS AP all mini subscript superscript AP mini all \text{AP}^{\text{mini}}_{\tex | LVIS AP rare mini subscript superscript AP mini rare \text{AP}^{\text{mini}}_{\t | LVIS AP all val subscript superscript AP val all \text{AP}^{\text{val}}_{\text{a | LVIS AP rare val subscript superscript AP val rare \text{{AP}}^{\text{{val}}}_{\ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Open vocabulary (evaluation vocabulary is not available at training time): |  |  |  |  |  |  |  |  |  |  |
| 1 | RegionCLIP [ 40 ] | R50x4 | CC3M | 6k concepts | LVIS base subscript LVIS base \text{LVIS}_{\text{base}} | – | – | – | 32.3 | +0.0 22.0 +0.0 |
| 2 | OWL [ 21 ] | CLIP B/16 | – | – | O365+VG | – | – | – | 27.2 | +0.0 20.6 +0.0 |
| 3 | OWL [ 21 ] | CLIP L/14 | – | – | O365+VG | 48.4 | – | – | 34.6 | +0.0 31.2 +0.0 |
| 4 | GLIPv2 [ 39 ] | Swin-T | Cap4M | tokens | O365+GoldG | 48.5 | 29.0 | – | – | – |
| 5 | GLIPv2 [ 39 ] | Swin-B | CC15M | tokens | FiveODs+GoldG | 54.2 | 48.5 | – | – | – |
| 6 | GLIPv2 [ 39 ] | Swin-H | CC15M | tokens | FiveODs+GoldG | 55.5 | 50.1 | – | – | – |
| 7 | F-VLM [ 14 ] | R50x4 | – | – | LVIS base subscript LVIS base \text{LVIS}_{\text{base}} | – | – | – | 28.5 | +0.0 26.3 +0.0 |
| 8 | F-VLM [ 14 ] | R50x64 | – | – | LVIS base subscript LVIS base \text{LVIS}_{\text{base}} | – | – | – | 34.9 | +0.0 32.8 +0.0 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: OWLv2 Table 2

|  | Token drop rate |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Metric | 0.00 | 0.25 | 0.33 | 0.50 | 0.70 |
| LVIS AP all val subscript superscript AP val all \text{AP}^{\text{val}}_{\text{a | 33.3 ± 0.33 plus-or-minus 0.33 \pm 0.33 | 33.1 | 33.6 | 32.9 | 30.4 |
| LVIS AP rare val subscript superscript AP val rare \text{AP}^{\text{val}}_{\text | 31.8 ± 1.16 plus-or-minus 1.16 \pm 1.16 | 31.0 | 32.6 | 30.8 | 28.2 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: OWLv2 Table 3

|  | Method | Backbone | Image size | Learning rate | Dropout rate | Droplayer rate | Instance top k 𝑘 k | Batch size (ST) | Batch size (FT) | Examples seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Open vocabulary: |  |  |  |  |  |  |  |  |  |  |
| 11 | OWL-ST | CLIP B/16 | 960 | 5 × 10 − 05 5E-05 5\text{\times}{10}^{-05} | .0/.0 | .2/.1 | 256 | 256 | – | 365 260 288 365260288 365\,260\,288 |
| 12 | OWL-ST | CLIP L/14 | 1008 | 2 × 10 − 05 2E-05 2\text{\times}{10}^{-05} | .0/.0 | .2/.1 | 512 | 256 | – | 225 893 376 225893376 225\,893\,376 |
| 13 | OWL-ST | SigLIP G/14 | 1008 | 2 × 10 − 05 2E-05 2\text{\times}{10}^{-05} | .0/.1 | .2/.4 | 512 | 128 | – | 160 396 032 160396032 160\,396\,032 |
| 14 | OWL-ST+FT | CLIP B/16 | 960 | 5 × 10 − 05 5E-05 5\text{\times}{10}^{-05} | .0/.0 | .2/.1 | 256 | 256 | 256 | 361 573 632 361573632 361\,573\,632 |
| 15 | OWL-ST+FT | CLIP L/14 | 1008 | 2 × 10 − 05 2E-05 2\text{\times}{10}^{-05} | .0/.0 | .2/.1 | 512 | 256 | 128 | 228 197 248 228197248 228\,197\,248 |
| 16 | OWL-ST+FT | SigLIP G/14 | 1008 | 2 × 10 − 05 2E-05 2\text{\times}{10}^{-05} | .0/.1 | .2/.4 | 512 | 128 | 128 | 162 904 704 162904704 162\,904\,704 |
| Human-curated vocabulary: |  |  |  |  |  |  |  |  |  |  |
| 20 | OWL-ST+FT | CLIP B/16 | 960 | 5 × 10 − 05 5E-05 5\text{\times}{10}^{-05} | .0/.0 | .2/.1 | 256 | 256 | 256 | 816 945 920 816945920 816\,945\,920 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: OWLv2 Table 4

|  | Method | Backbone | AP all mini subscript superscript AP mini all \text{AP}^{\text{mini}}_{\text{all | AP rare mini subscript superscript AP mini rare \text{AP}^{\text{mini}}_{\text{r | AP all val subscript superscript AP val all \text{AP}^{\text{val}}_{\text{all}} | AP rare val subscript superscript AP val rare \text{{AP}}^{\text{{val}}}_{\text{ |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | old | fixed | old | fixed | old | fixed | old | fixed |  |  |
| Open vocabulary: |  |  |  |  |  |  |  |  |  |  |
| 1 | RegionCLIP [ 40 ] | R50x4 | – | – | – | – | 32.3 | – | 22.0 | – |
| 2 | OWL [ 21 ] | CLIP B/16 | – | – | – | – | 27.2 | – | 20.6 | – |
| 3 | OWL [ 21 ] | CLIP L/14 | – | – | – | – | 34.6 | – | 31.2 | – |
| 4 | GLIPv2 [ 39 ] | Swin-T | 29.0 | – | – | – | – | – | – | – |
| 5 | GLIPv2 [ 39 ] | Swin-B | 48.5 | – | – | – | – | – | – | – |
| 6 | GLIPv2 [ 39 ] | Swin-H | 50.1 | – | – | – | – | – | – | – |
| 7 | F-VLM [ 14 ] | R50x4 | – | – | – | – | 28.5 | – | 26.3 | – |

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
- [[Detic]]
- [[YOLO-World]]
