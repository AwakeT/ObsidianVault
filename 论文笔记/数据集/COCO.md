---
title: "Microsoft COCO: Common Objects in Context"
method_name: "COCO"
authors: [Tsung-Yi Lin, et al.]
year: 2014
venue: ECCV
tags: [dataset, object-detection, instance-segmentation, benchmark]
arxiv: https://arxiv.org/abs/1405.0312
created: 2026-06-11
image_source: online
---

# 论文笔记：Microsoft COCO: Common Objects in Context

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[COCO]] |
| 作者 | [Tsung-Yi Lin, et al.] |
| 年份 | 2014 |
| 会议/来源 | ECCV |
| 链接 | 无 arXiv 链接 / 经典论文 |
| 相关工作 | [[R-CNN]]、[[Faster R-CNN]]、[[RetinaNet]]、[[DETR]] |

---

## 一句话总结

> COCO 是现代目标检测最核心的数据集和评测基准之一，强调复杂场景、多实例和上下文中的常见物体。

---

## 核心贡献

1. **范式贡献**：COCO 是现代目标检测最核心的数据集和评测基准之一，强调复杂场景、多实例和上下文中的常见物体。
2. **方法贡献**：构建 Common Objects in Context 数据集，提供 bbox、mask、caption 等标注，并推动 AP@[0.5:0.95] 成为标准评测。
3. **工程/研究影响**：该工作成为后续 [[R-CNN]]、[[Faster R-CNN]]、[[RetinaNet]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

构建 Common Objects in Context 数据集，提供 bbox、mask、caption 等标注，并推动 AP@[0.5:0.95] 成为标准评测。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[R-CNN]]、[[Faster R-CNN]]。
- 后续影响：[[RetinaNet]]、[[DETR]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
AP=\frac{1}{10}\sum_{t\in\{.50,\ldots,.95\}}AP_t
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1405.0312](https://ar5iv.labs.arxiv.org/html/1405.0312)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 22 张、Table 1 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1 : While previous object recognition datasets have focused on (a) image classification, (b) object bounding box localization or (c) semantic pixel-level segmentation, we focus on (d) segmenting individual object instances. We introduce a large, richly-annotated dataset comprised of images depicting complex everyday scenes of common objects in their natural context.

![Figure 1](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/teaser.jpg)

**图表解读**：重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 2: Figure 2 : Example of (a) iconic object images, (b) iconic scene images, and (c) non-iconic images.

![Figure 2](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/iconic.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 3: Figure 3 : Our annotation pipeline is split into 3 primary tasks: (a) labeling the categories present in the image (§ 4.1 ), (b) locating and marking all instances of the labeled categories (§ 4.2 ), and (c) segmenting each object instance (§ 4.3 ).

![Figure 3](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/ui_pipeline_summary.jpg)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 4: (a)

![Figure 4](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/expert-worker.png)

**图表解读**：重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 5: (b)

![Figure 5](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/x1.png)

**图表解读**：重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 6: Figure 5 : (a) Number of annotated instances per category for MS COCO and PASCAL VOC. (b,c) Number of annotated categories and annotated instances, respectively, per image for MS COCO, ImageNet Detection, PASCAL VOC and SUN (average number of categories and instances are shown in parentheses). (d) Number of categories vs. the number of instances per category for a number of popular object recognition datasets. (e) The distribution of instance sizes for the MS COCO, ImageNet Detection, PASCAL VOC and SUN datasets.

![Figure 6](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/dataanalysis.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 7: Figure 6 : Samples of annotated images in the MS COCO dataset.

![Figure 7](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/fantastic_fig.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 8: Figure 7 : We visualize our mixture-specific shape masks. We paste thresholded shape masks on each candidate detection to generate candidate segments.

![Figure 8](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/x2.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 9: Figure 8 : Evaluating instance detections with segmentation masks versus bounding boxes. Bounding boxes are a particularly crude approximation for articulated objects; in this case, the majority of the pixels in the ( blue ) tight-fitting bounding-box do not lie on the object. Our ( green ) instance-level segmentation masks allows for a more accurate measure of object detection and localization.

![Figure 9](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/x3.png)

**图表解读**：重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 10: Figure 9 : A predicted segmentation might not recover object detail even though detection and ground truth bounding boxes overlap well (left). Sampling from the person category illustrates that predicting segmentations from top-down projection of DPM part masks is difficult even for correct detections (center). Average segmentation overlap measured on MS COCO for the 20 PASCAL VOC categories demonstrates the difficulty of the problem (right).

![Figure 10](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/ped_box.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 11: Figure 9 : A predicted segmentation might not recover object detail even though detection and ground truth bounding boxes overlap well (left). Sampling from the person category illustrates that predicting segmentations from top-down projection of DPM part masks is difficult even for correct detections (center). Average segmentation overlap measured on MS COCO for the 20 PASCAL VOC categories demonstrates the difficulty of the problem (right).

![Figure 11](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/ped_mask_gt.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 12: Figure 9 : A predicted segmentation might not recover object detail even though detection and ground truth bounding boxes overlap well (left). Sampling from the person category illustrates that predicting segmentations from top-down projection of DPM part masks is difficult even for correct detections (center). Average segmentation overlap measured on MS COCO for the 20 PASCAL VOC categories demonstrates the difficulty of the problem (right).

![Figure 12](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/ped_mask_det.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 13: Figure 9 : A predicted segmentation might not recover object detail even though detection and ground truth bounding boxes overlap well (left). Sampling from the person category illustrates that predicting segmentations from top-down projection of DPM part masks is difficult even for correct detections (center). Average segmentation overlap measured on MS COCO for the 20 PASCAL VOC categories demonstrates the difficulty of the problem (right).

![Figure 13](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/x4.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 14: Figure 9 : A predicted segmentation might not recover object detail even though detection and ground truth bounding boxes overlap well (left). Sampling from the person category illustrates that predicting segmentations from top-down projection of DPM part masks is difficult even for correct detections (center). Average segmentation overlap measured on MS COCO for the 20 PASCAL VOC categories demonstrates the difficulty of the problem (right).

![Figure 14](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/x5.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 15: Figure 10 : User interfaces for non-iconic image collection. (a) Interface for selecting non-iconic images containing pairs of objects. (b) Interface for selecting non-iconic images for categories that rarely co-occurred with others.

![Figure 15](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/ui_collection.jpg)

**图表解读**：重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 16: Figure 11 : Icons of 91 categories in the MS COCO dataset grouped by 11 super-categories. We use these icons in our annotation pipeline to help workers quickly reference the indicated object category.

![Figure 16](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/icons.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 17: Figure 12 : User interfaces for collecting instance annotations, see text for details.

![Figure 17](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/ui_pipeline.jpg)

**图表解读**：重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 18: (a) PASCAL VOC.

![Figure 18](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/ex_pascal_person.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 19: (b) MS COCO.

![Figure 19](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/ex_coco_person.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 20: (a) PASCAL VOC.

![Figure 20](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/ex_pascal_chair.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 21: (b) MS COCO.

![Figure 21](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/ex_coco_chair.jpg)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### Figure 22: Figure 15 : Examples of borderline segmentations that passed (top) or were rejected (bottom) in the verification stage.

![Figure 22](https://ar5iv.labs.arxiv.org/html/1405.0312/assets/figs/verification.jpg)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看数据集标注类型、类别分布和 AP 指标；它是基础评测，但室内导航长尾覆盖不足。

### 关键表格

#### Table 1: COCO Table 1

| person | bicycle | car | motorcycle | bird | cat | dog | horse | sheep | bottle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chair | couch | potted plant | tv | cow | airplane | hat ∗ | license plate | bed | laptop |
| fridge | microwave | sink | oven | toaster | bus | train | mirror ∗ | dining table | elephant |
| banana | bread | toilet | book | boat | plate ∗ | cell phone | mouse | remote | clock |
| face | hand | apple | keyboard | backpack | steering wheel | wine glass | chicken | zebra | shoe ∗ |
| eye | mouth | scissors | truck | traffic light | eyeglasses ∗ | cup | blender ∗ | hair drier | wheel |
| street sign ∗ | umbrella | door ∗ | fire hydrant | bowl | teapot | fork | knife | spoon | bear |
| headlights | window ∗ | desk ∗ | computer | refrigerator | pizza | squirrel | duck | frisbee | guitar |
| nose | teddy bear | tie | stop sign | surfboard | sandwich | pen/pencil | kite | orange | toothbrush |
| printer | pans | head | sports ball | broccoli | suitcase | carrot | chandelier | parking meter | fish |

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

COCO 是干净基础集，但室内类覆盖不足，必须与 Objects365/in-domain 数据混合。

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

- [[R-CNN]]
- [[Faster R-CNN]]
- [[RetinaNet]]
- [[DETR]]
