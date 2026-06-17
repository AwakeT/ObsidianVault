---
title: "Rich feature hierarchies for accurate object detection and semantic segmentation"
method_name: "R-CNN"
authors: [Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik]
year: 2013
venue: CVPR
tags: [object-detection, cnn, two-stage-detection, region-proposal]
arxiv: https://arxiv.org/abs/1311.2524
created: 2026-06-11
image_source: online
---

# 论文笔记：Rich feature hierarchies for accurate object detection and semantic segmentation

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[R-CNN]] |
| 作者 | [Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik] |
| 年份 | 2013 |
| 会议/来源 | CVPR |
| 链接 | [arXiv](https://arxiv.org/abs/1311.2524) / [PDF](https://arxiv.org/pdf/1311.2524.pdf) |
| 相关工作 | [[DPM]]、[[Fast R-CNN]]、[[Faster R-CNN]]、[[FPN]] |

---

## 一句话总结

> R-CNN 首次系统证明 CNN 特征能大幅提升目标检测，是深度检测时代的起点。

---

## 核心贡献

1. **范式贡献**：R-CNN 首次系统证明 CNN 特征能大幅提升目标检测，是深度检测时代的起点。
2. **方法贡献**：用 Selective Search 生成候选区域，对每个 proposal 裁剪后送入 CNN 提取特征，再用 SVM 分类并用 bbox regression 精修。核心贡献是把 CNN 迁移到 region proposal 检测框架。
3. **工程/研究影响**：该工作成为后续 [[DPM]]、[[Fast R-CNN]]、[[Faster R-CNN]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

用 Selective Search 生成候选区域，对每个 proposal 裁剪后送入 CNN 提取特征，再用 SVM 分类并用 bbox regression 精修。核心贡献是把 CNN 迁移到 region proposal 检测框架。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[DPM]]、[[Fast R-CNN]]。
- 后续影响：[[Faster R-CNN]]、[[FPN]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\text{score}(r,c)=w_c^\top\phi_{\text{CNN}}(r)
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1311.2524](https://ar5iv.labs.arxiv.org/html/1311.2524)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 214 张、Table 6 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: Object detection system overview. Our system (1) takes an input image, (2) extracts around 2000 bottom-up region proposals, (3) computes features for each proposal using a large convolutional neural network (CNN), and then (4) classifies each region using class-specific linear SVMs. R-CNN achieves a mean average precision (mAP) of 53.7% on PASCAL VOC 2010 . For comparison, [ 39 ] reports 35.1% mAP using the same region proposals, but with a spatial pyramid and bag-of-visual-words approach. The popular deformable part models perform at 33.4%. On the 200-class ILSVRC2013 detection dataset, R-CNN’s mAP is 31.4% , a large improvement over OverFeat [ 34 ] , which had the previous best result at 24.3%.

![Figure 1](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 2: Figure 2: Warped training samples from VOC 2007 train.

![Figure 2](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x2.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 3: Figure 3: (Left) Mean average precision on the ILSVRC2013 detection test set. Methods preceeded by * use outside training data (images and labels from the ILSVRC classification dataset in all cases). (Right) Box plots for the 200 average precision values per method. A box plot for the post-competition OverFeat result is not shown because per-class APs are not yet available (per-class APs for R-CNN are in Table 8 and also included in the tech report source uploaded to arXiv.org; see R-CNN-ILSVRC2013-APs.txt ). The red line marks the median AP, the box bottom and top are the 25th and 75th percentiles. The whiskers extend to the min and max AP of each method. Each AP is plotted as a green dot over the whiskers (best viewed digitally with zoom).

![Figure 3](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x3.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 4: Figure 3: (Left) Mean average precision on the ILSVRC2013 detection test set. Methods preceeded by * use outside training data (images and labels from the ILSVRC classification dataset in all cases). (Right) Box plots for the 200 average precision values per method. A box plot for the post-competition OverFeat result is not shown because per-class APs are not yet available (per-class APs for R-CNN are in Table 8 and also included in the tech report source uploaded to arXiv.org; see R-CNN-ILSVRC2013-APs.txt ). The red line marks the median AP, the box bottom and top are the 25th and 75th percentiles. The whiskers extend to the min and max AP of each method. Each AP is plotted as a green dot over the whiskers (best viewed digitally with zoom).

![Figure 4](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x4.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 5: Figure 4: Top regions for six pool 5 subscript pool 5 \textrm{pool}_{5} units. Receptive fields and activation values are drawn in white. Some units are aligned to concepts, such as people (row 1) or text (4). Other units capture texture and material properties, such as dot arrays (2) and specular reflections (6).

![Figure 5](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x5.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 6: Figure 4: Top regions for six pool 5 subscript pool 5 \textrm{pool}_{5} units. Receptive fields and activation values are drawn in white. Some units are aligned to concepts, such as people (row 1) or text (4). Other units capture texture and material properties, such as dot arrays (2) and specular reflections (6).

![Figure 6](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x6.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 7: Figure 4: Top regions for six pool 5 subscript pool 5 \textrm{pool}_{5} units. Receptive fields and activation values are drawn in white. Some units are aligned to concepts, such as people (row 1) or text (4). Other units capture texture and material properties, such as dot arrays (2) and specular reflections (6).

![Figure 7](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x7.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 8: Figure 4: Top regions for six pool 5 subscript pool 5 \textrm{pool}_{5} units. Receptive fields and activation values are drawn in white. Some units are aligned to concepts, such as people (row 1) or text (4). Other units capture texture and material properties, such as dot arrays (2) and specular reflections (6).

![Figure 8](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x8.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 9: Figure 4: Top regions for six pool 5 subscript pool 5 \textrm{pool}_{5} units. Receptive fields and activation values are drawn in white. Some units are aligned to concepts, such as people (row 1) or text (4). Other units capture texture and material properties, such as dot arrays (2) and specular reflections (6).

![Figure 9](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x9.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 10: Figure 4: Top regions for six pool 5 subscript pool 5 \textrm{pool}_{5} units. Receptive fields and activation values are drawn in white. Some units are aligned to concepts, such as people (row 1) or text (4). Other units capture texture and material properties, such as dot arrays (2) and specular reflections (6).

![Figure 10](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x10.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 11: Figure 5: Distribution of top-ranked false positive (FP) types. Each plot shows the evolving distribution of FP types as more FPs are considered in order of decreasing score. Each FP is categorized into 1 of 4 types: Loc—poor localization (a detection with an IoU overlap with the correct class between 0.1 and 0.5, or a duplicate); Sim—confusion with a similar category; Oth—confusion with a dissimilar object category; BG—a FP that fired on background. Compared with DPM (see [ 23 ] ), significantly more of our errors result from poor localization, rather than confusion with background or other object classes, indicating that the CNN features are much more discriminative than HOG. Loose localization likely results from our use of bottom-up region proposals and the positional invariance learned from pre-training the CNN for whole-image classification. Column three shows how our simple bounding-box regression method fixes many localization errors.

![Figure 11](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x11.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 12: Figure 5: Distribution of top-ranked false positive (FP) types. Each plot shows the evolving distribution of FP types as more FPs are considered in order of decreasing score. Each FP is categorized into 1 of 4 types: Loc—poor localization (a detection with an IoU overlap with the correct class between 0.1 and 0.5, or a duplicate); Sim—confusion with a similar category; Oth—confusion with a dissimilar object category; BG—a FP that fired on background. Compared with DPM (see [ 23 ] ), significantly more of our errors result from poor localization, rather than confusion with background or other object classes, indicating that the CNN features are much more discriminative than HOG. Loose localization likely results from our use of bottom-up region proposals and the positional invariance learned from pre-training the CNN for whole-image classification. Column three shows how our simple bounding-box regression method fixes many localization errors.

![Figure 12](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x12.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 13: Figure 5: Distribution of top-ranked false positive (FP) types. Each plot shows the evolving distribution of FP types as more FPs are considered in order of decreasing score. Each FP is categorized into 1 of 4 types: Loc—poor localization (a detection with an IoU overlap with the correct class between 0.1 and 0.5, or a duplicate); Sim—confusion with a similar category; Oth—confusion with a dissimilar object category; BG—a FP that fired on background. Compared with DPM (see [ 23 ] ), significantly more of our errors result from poor localization, rather than confusion with background or other object classes, indicating that the CNN features are much more discriminative than HOG. Loose localization likely results from our use of bottom-up region proposals and the positional invariance learned from pre-training the CNN for whole-image classification. Column three shows how our simple bounding-box regression method fixes many localization errors.

![Figure 13](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x13.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 14: Figure 5: Distribution of top-ranked false positive (FP) types. Each plot shows the evolving distribution of FP types as more FPs are considered in order of decreasing score. Each FP is categorized into 1 of 4 types: Loc—poor localization (a detection with an IoU overlap with the correct class between 0.1 and 0.5, or a duplicate); Sim—confusion with a similar category; Oth—confusion with a dissimilar object category; BG—a FP that fired on background. Compared with DPM (see [ 23 ] ), significantly more of our errors result from poor localization, rather than confusion with background or other object classes, indicating that the CNN features are much more discriminative than HOG. Loose localization likely results from our use of bottom-up region proposals and the positional invariance learned from pre-training the CNN for whole-image classification. Column three shows how our simple bounding-box regression method fixes many localization errors.

![Figure 14](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x14.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 15: Figure 5: Distribution of top-ranked false positive (FP) types. Each plot shows the evolving distribution of FP types as more FPs are considered in order of decreasing score. Each FP is categorized into 1 of 4 types: Loc—poor localization (a detection with an IoU overlap with the correct class between 0.1 and 0.5, or a duplicate); Sim—confusion with a similar category; Oth—confusion with a dissimilar object category; BG—a FP that fired on background. Compared with DPM (see [ 23 ] ), significantly more of our errors result from poor localization, rather than confusion with background or other object classes, indicating that the CNN features are much more discriminative than HOG. Loose localization likely results from our use of bottom-up region proposals and the positional invariance learned from pre-training the CNN for whole-image classification. Column three shows how our simple bounding-box regression method fixes many localization errors.

![Figure 15](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x15.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 16: Figure 5: Distribution of top-ranked false positive (FP) types. Each plot shows the evolving distribution of FP types as more FPs are considered in order of decreasing score. Each FP is categorized into 1 of 4 types: Loc—poor localization (a detection with an IoU overlap with the correct class between 0.1 and 0.5, or a duplicate); Sim—confusion with a similar category; Oth—confusion with a dissimilar object category; BG—a FP that fired on background. Compared with DPM (see [ 23 ] ), significantly more of our errors result from poor localization, rather than confusion with background or other object classes, indicating that the CNN features are much more discriminative than HOG. Loose localization likely results from our use of bottom-up region proposals and the positional invariance learned from pre-training the CNN for whole-image classification. Column three shows how our simple bounding-box regression method fixes many localization errors.

![Figure 16](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x16.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 17: Figure 6: Sensitivity to object characteristics. Each plot shows the mean (over classes) normalized AP (see [ 23 ] ) for the highest and lowest performing subsets within six different object characteristics (occlusion, truncation, bounding-box area, aspect ratio, viewpoint, part visibility). We show plots for our method (R-CNN) with and without fine-tuning (FT) and bounding-box regression (BB) as well as for DPM voc-release5. Overall, fine-tuning does not reduce sensitivity (the difference between max and min), but does substantially improve both the highest and lowest performing subsets for nearly all characteristics. This indicates that fine-tuning does more than simply improve the lowest performing subsets for aspect ratio and bounding-box area, as one might conjecture based on how we warp network inputs. Instead, fine-tuning improves robustness for all characteristics including occlusion, truncation, viewpoint, and part visibility.

![Figure 17](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x17.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 18: Figure 6: Sensitivity to object characteristics. Each plot shows the mean (over classes) normalized AP (see [ 23 ] ) for the highest and lowest performing subsets within six different object characteristics (occlusion, truncation, bounding-box area, aspect ratio, viewpoint, part visibility). We show plots for our method (R-CNN) with and without fine-tuning (FT) and bounding-box regression (BB) as well as for DPM voc-release5. Overall, fine-tuning does not reduce sensitivity (the difference between max and min), but does substantially improve both the highest and lowest performing subsets for nearly all characteristics. This indicates that fine-tuning does more than simply improve the lowest performing subsets for aspect ratio and bounding-box area, as one might conjecture based on how we warp network inputs. Instead, fine-tuning improves robustness for all characteristics including occlusion, truncation, viewpoint, and part visibility.

![Figure 18](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x18.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 19: Figure 6: Sensitivity to object characteristics. Each plot shows the mean (over classes) normalized AP (see [ 23 ] ) for the highest and lowest performing subsets within six different object characteristics (occlusion, truncation, bounding-box area, aspect ratio, viewpoint, part visibility). We show plots for our method (R-CNN) with and without fine-tuning (FT) and bounding-box regression (BB) as well as for DPM voc-release5. Overall, fine-tuning does not reduce sensitivity (the difference between max and min), but does substantially improve both the highest and lowest performing subsets for nearly all characteristics. This indicates that fine-tuning does more than simply improve the lowest performing subsets for aspect ratio and bounding-box area, as one might conjecture based on how we warp network inputs. Instead, fine-tuning improves robustness for all characteristics including occlusion, truncation, viewpoint, and part visibility.

![Figure 19](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x19.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 20: Figure 6: Sensitivity to object characteristics. Each plot shows the mean (over classes) normalized AP (see [ 23 ] ) for the highest and lowest performing subsets within six different object characteristics (occlusion, truncation, bounding-box area, aspect ratio, viewpoint, part visibility). We show plots for our method (R-CNN) with and without fine-tuning (FT) and bounding-box regression (BB) as well as for DPM voc-release5. Overall, fine-tuning does not reduce sensitivity (the difference between max and min), but does substantially improve both the highest and lowest performing subsets for nearly all characteristics. This indicates that fine-tuning does more than simply improve the lowest performing subsets for aspect ratio and bounding-box area, as one might conjecture based on how we warp network inputs. Instead, fine-tuning improves robustness for all characteristics including occlusion, truncation, viewpoint, and part visibility.

![Figure 20](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x20.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 21: Figure 7: Different object proposal transformations. (A) the original object proposal at its actual scale relative to the transformed CNN inputs; (B) tightest square with context; (C) tightest square without context; (D) warp. Within each column and example proposal, the top row corresponds to p = 0 𝑝 0 p=0 pixels of context padding while the bottom row has p = 16 𝑝 16 p=16 pixels of context padding.

![Figure 21](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x21.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 22: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 22](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x22.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 23: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 23](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x23.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 24: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 24](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x24.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 25: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 25](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x25.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 26: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 26](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x26.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 27: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 27](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x27.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 28: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 28](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x28.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 29: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 29](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x29.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 30: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 30](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x30.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 31: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 31](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x31.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 32: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 32](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x32.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 33: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 33](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x33.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 34: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 34](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x34.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 35: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 35](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x35.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 36: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 36](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x36.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 37: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 37](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x37.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 38: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 38](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x38.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 39: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 39](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x39.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 40: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 40](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x40.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 41: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 41](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x41.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 42: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 42](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x42.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 43: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 43](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x43.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 44: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 44](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x44.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 45: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 45](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x45.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 46: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 46](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x46.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 47: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 47](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x47.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 48: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 48](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x48.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 49: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 49](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x49.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 50: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 50](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x50.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 51: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 51](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x51.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 52: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 52](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x52.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 53: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 53](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x53.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 54: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 54](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x54.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 55: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 55](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x55.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 56: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 56](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x56.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 57: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 57](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x57.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 58: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 58](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x58.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 59: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 59](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x59.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 60: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 60](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x60.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 61: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 61](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x61.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 62: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 62](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x62.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 63: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 63](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x63.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 64: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 64](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x64.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 65: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 65](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x65.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 66: Figure 8: Example detections on the val 2 set from the configuration that achieved 31.0% mAP on val 2 . Each image was sampled randomly (these are not curated). All detections at precision greater than 0.5 are shown. Each detection is labeled with the predicted class and the precision value of that detection from the detector’s precision-recall curve. Viewing digitally with zoom is recommended.

![Figure 66](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x66.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 67: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 67](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x67.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 68: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 68](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x68.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 69: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 69](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x69.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 70: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 70](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x70.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 71: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 71](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x71.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 72: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 72](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x72.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 73: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 73](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x73.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 74: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 74](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x74.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 75: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 75](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x75.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 76: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 76](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x76.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 77: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 77](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x77.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 78: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 78](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x78.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 79: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 79](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x79.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 80: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 80](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x80.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 81: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 81](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x81.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 82: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 82](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x82.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 83: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 83](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x83.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 84: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 84](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x84.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 85: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 85](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x85.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 86: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 86](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x86.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 87: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 87](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x87.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 88: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 88](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x88.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 89: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 89](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x89.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 90: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 90](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x90.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 91: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 91](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x91.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 92: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 92](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x92.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 93: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 93](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x93.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 94: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 94](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x94.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 95: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 95](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x95.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 96: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 96](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x96.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 97: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 97](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x97.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 98: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 98](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x98.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 99: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 99](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x99.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 100: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 100](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x100.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 101: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 101](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x101.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 102: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 102](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x102.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 103: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 103](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x103.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 104: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 104](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x104.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 105: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 105](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x105.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 106: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 106](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x106.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 107: Figure 9: More randomly selected examples. See Figure 8 caption for details. Viewing digitally with zoom is recommended.

![Figure 107](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x107.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 108: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 108](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x108.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 109: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 109](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x109.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 110: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 110](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x110.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 111: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 111](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x111.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 112: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 112](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x112.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 113: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 113](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x113.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 114: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 114](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x114.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 115: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 115](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x115.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 116: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 116](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x116.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 117: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 117](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x117.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 118: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 118](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x118.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 119: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 119](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x119.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 120: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 120](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x120.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 121: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 121](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x121.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 122: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 122](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x122.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 123: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 123](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x123.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 124: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 124](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x124.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 125: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 125](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x125.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 126: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 126](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x126.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 127: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 127](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x127.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 128: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 128](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x128.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 129: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 129](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x129.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 130: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 130](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x130.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 131: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 131](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x131.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 132: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 132](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x132.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 133: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 133](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x133.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 134: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 134](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x134.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 135: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 135](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x135.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 136: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 136](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x136.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 137: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 137](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x137.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 138: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 138](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x138.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 139: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 139](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x139.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 140: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 140](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x140.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 141: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 141](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x141.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 142: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 142](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x142.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 143: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 143](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x143.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 144: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 144](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x144.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 145: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 145](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x145.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 146: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 146](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x146.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 147: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 147](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x147.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 148: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 148](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x148.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 149: Figure 10: Curated examples. Each image was selected because we found it impressive, surprising, interesting, or amusing. Viewing digitally with zoom is recommended.

![Figure 149](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x149.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 150: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 150](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x150.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 151: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 151](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x151.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 152: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 152](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x152.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 153: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 153](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x153.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 154: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 154](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x154.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 155: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 155](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x155.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 156: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 156](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x156.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 157: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 157](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x157.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 158: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 158](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x158.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 159: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 159](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x159.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 160: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 160](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x160.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 161: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 161](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x161.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 162: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 162](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x162.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 163: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 163](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x163.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 164: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 164](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x164.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 165: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 165](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x165.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 166: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 166](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x166.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 167: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 167](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x167.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 168: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 168](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x168.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 169: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 169](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x169.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 170: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 170](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x170.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 171: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 171](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x171.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 172: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 172](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x172.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 173: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 173](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x173.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 174: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 174](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x174.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 175: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 175](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x175.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 176: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 176](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x176.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 177: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 177](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x177.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 178: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 178](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x178.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 179: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 179](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x179.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 180: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 180](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x180.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 181: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 181](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x181.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 182: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 182](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x182.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 183: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 183](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x183.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 184: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 184](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x184.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 185: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 185](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x185.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 186: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 186](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x186.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 187: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 187](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x187.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 188: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 188](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x188.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 189: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 189](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x189.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 190: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 190](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x190.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 191: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 191](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x191.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 192: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 192](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x192.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 193: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 193](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x193.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 194: Figure 11: More curated examples. See Figure 10 caption for details. Viewing digitally with zoom is recommended.

![Figure 194](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x194.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 195: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 195](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x195.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 196: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 196](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x196.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 197: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 197](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x197.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 198: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 198](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x198.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 199: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 199](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x199.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 200: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 200](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x200.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 201: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 201](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x201.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 202: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 202](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x202.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 203: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 203](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x203.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 204: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 204](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x204.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 205: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 205](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x205.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 206: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 206](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x206.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 207: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 207](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x207.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 208: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 208](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x208.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 209: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 209](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x209.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 210: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 210](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x210.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 211: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 211](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x211.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 212: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 212](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x212.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 213: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 213](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x213.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 214: Figure 12: We show the 24 region proposals, out of the approximately 10 million regions in VOC 2007 test, that most strongly activate each of 20 units. Each montage is labeled by the unit’s (y, x, channel) position in the 6 × 6 × 256 6 6 256 6\times 6\times 256 dimensional pool 5 subscript pool 5 \textrm{pool}_{5} feature map. Each image region is drawn with an overlay of the unit’s receptive field in white. The activation value (which we normalize by dividing by the max activation value over all units in a channel) is shown in the receptive field’s upper-left corner. Best viewed digitally with zoom.

![Figure 214](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x214.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### 关键表格

#### Table 1: R-CNN Table 1

| VOC 2010 test | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | mAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DPM v5 [ 20 ] † | 49.2 | 53.8 | 13.1 | 15.3 | 35.5 | 53.4 | 49.7 | 27.0 | 17.2 | 28.8 | 14.7 | 17.8 | 46.4 | 51.2 | 47.7 | 10.8 | 34.2 | 20.7 | 43.8 | 38.3 | 33.4 |
| UVA [ 39 ] | 56.2 | 42.4 | 15.3 | 12.6 | 21.8 | 49.3 | 36.8 | 46.1 | 12.9 | 32.1 | 30.0 | 36.5 | 43.5 | 52.9 | 32.9 | 15.3 | 41.1 | 31.8 | 47.0 | 44.8 | 35.1 |
| Regionlets [ 41 ] | 65.0 | 48.9 | 25.9 | 24.6 | 24.5 | 56.1 | 54.5 | 51.2 | 17.0 | 28.9 | 30.2 | 35.8 | 40.2 | 55.7 | 43.5 | 14.3 | 43.9 | 32.6 | 54.0 | 45.9 | 39.7 |
| SegDPM [ 18 ] † | 61.4 | 53.4 | 25.6 | 25.2 | 35.5 | 51.7 | 50.6 | 50.8 | 19.3 | 33.8 | 26.8 | 40.4 | 48.3 | 54.4 | 47.1 | 14.8 | 38.7 | 35.0 | 52.8 | 43.1 | 40.4 |
| R-CNN | 67.1 | 64.1 | 46.7 | 32.0 | 30.5 | 56.4 | 57.2 | 65.9 | 27.0 | 47.3 | 40.9 | 66.6 | 57.8 | 65.9 | 53.6 | 26.7 | 56.5 | 38.1 | 52.8 | 50.2 | 50.2 |
| R-CNN BB | 71.8 | 65.8 | 53.0 | 36.8 | 35.9 | 59.7 | 60.0 | 69.9 | 27.9 | 50.6 | 41.4 | 70.0 | 62.0 | 69.0 | 58.1 | 29.5 | 59.4 | 39.3 | 61.2 | 52.4 | 53.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: R-CNN Table 2

| VOC 2007 test | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | mAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-CNN pool 5 subscript pool 5 \textrm{pool}_{5} | 51.8 | 60.2 | 36.4 | 27.8 | 23.2 | 52.8 | 60.6 | 49.2 | 18.3 | 47.8 | 44.3 | 40.8 | 56.6 | 58.7 | 42.4 | 23.4 | 46.1 | 36.7 | 51.3 | 55.7 | 44.2 |
| R-CNN fc 6 subscript fc 6 \textrm{fc}_{6} | 59.3 | 61.8 | 43.1 | 34.0 | 25.1 | 53.1 | 60.6 | 52.8 | 21.7 | 47.8 | 42.7 | 47.8 | 52.5 | 58.5 | 44.6 | 25.6 | 48.3 | 34.0 | 53.1 | 58.0 | 46.2 |
| R-CNN fc 7 subscript fc 7 \textrm{fc}_{7} | 57.6 | 57.9 | 38.5 | 31.8 | 23.7 | 51.2 | 58.9 | 51.4 | 20.0 | 50.5 | 40.9 | 46.0 | 51.6 | 55.9 | 43.3 | 23.3 | 48.1 | 35.3 | 51.0 | 57.4 | 44.7 |
| R-CNN FT pool 5 subscript pool 5 \textrm{pool}_{5} | 58.2 | 63.3 | 37.9 | 27.6 | 26.1 | 54.1 | 66.9 | 51.4 | 26.7 | 55.5 | 43.4 | 43.1 | 57.7 | 59.0 | 45.8 | 28.1 | 50.8 | 40.6 | 53.1 | 56.4 | 47.3 |
| R-CNN FT fc 6 subscript fc 6 \textrm{fc}_{6} | 63.5 | 66.0 | 47.9 | 37.7 | 29.9 | 62.5 | 70.2 | 60.2 | 32.0 | 57.9 | 47.0 | 53.5 | 60.1 | 64.2 | 52.2 | 31.3 | 55.0 | 50.0 | 57.7 | 63.0 | 53.1 |
| R-CNN FT fc 7 subscript fc 7 \textrm{fc}_{7} | 64.2 | 69.7 | 50.0 | 41.9 | 32.0 | 62.6 | 71.0 | 60.7 | 32.7 | 58.5 | 46.5 | 56.1 | 60.6 | 66.8 | 54.2 | 31.5 | 52.8 | 48.9 | 57.9 | 64.7 | 54.2 |
| R-CNN FT fc 7 subscript fc 7 \textrm{fc}_{7} BB | 68.1 | 72.8 | 56.8 | 43.0 | 36.8 | 66.3 | 74.2 | 67.6 | 34.4 | 63.5 | 54.5 | 61.2 | 69.1 | 68.6 | 58.7 | 33.4 | 62.9 | 51.1 | 62.5 | 64.8 | 58.5 |
| DPM v5 [ 20 ] | 33.2 | 60.3 | 10.2 | 16.1 | 27.3 | 54.3 | 58.2 | 23.0 | 20.0 | 24.1 | 26.7 | 12.7 | 58.1 | 48.2 | 43.2 | 12.0 | 21.1 | 36.1 | 46.0 | 43.5 | 33.7 |
| DPM ST [ 28 ] | 23.8 | 58.2 | 10.5 | 0 8.5 | 27.1 | 50.4 | 52.0 | 0 7.3 | 19.2 | 22.8 | 18.1 | 0 8.0 | 55.9 | 44.8 | 32.4 | 13.3 | 15.9 | 22.8 | 46.2 | 44.9 | 29.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: R-CNN Table 3

| VOC 2007 test | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | mAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-CNN T-Net | 64.2 | 69.7 | 50.0 | 41.9 | 32.0 | 62.6 | 71.0 | 60.7 | 32.7 | 58.5 | 46.5 | 56.1 | 60.6 | 66.8 | 54.2 | 31.5 | 52.8 | 48.9 | 57.9 | 64.7 | 54.2 |
| R-CNN T-Net BB | 68.1 | 72.8 | 56.8 | 43.0 | 36.8 | 66.3 | 74.2 | 67.6 | 34.4 | 63.5 | 54.5 | 61.2 | 69.1 | 68.6 | 58.7 | 33.4 | 62.9 | 51.1 | 62.5 | 64.8 | 58.5 |
| R-CNN O-Net | 71.6 | 73.5 | 58.1 | 42.2 | 39.4 | 70.7 | 76.0 | 74.5 | 38.7 | 71.0 | 56.9 | 74.5 | 67.9 | 69.6 | 59.3 | 35.7 | 62.1 | 64.0 | 66.5 | 71.2 | 62.2 |
| R-CNN O-Net BB | 73.4 | 77.0 | 63.4 | 45.4 | 44.6 | 75.1 | 78.1 | 79.8 | 40.5 | 73.7 | 62.2 | 79.4 | 78.1 | 73.1 | 64.2 | 35.6 | 66.8 | 67.2 | 70.4 | 71.1 | 66.0 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: R-CNN Table 4

| VOC 2011 test | bg | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R&P [ 2 ] | 83.4 | 46.8 | 18.9 | 36.6 | 31.2 | 42.7 | 57.3 | 47.4 | 44.1 | 0 8.1 | 39.4 | 36.1 | 36.3 | 49.5 | 48.3 | 50.7 | 26.3 | 47.2 | 22.1 | 42.0 | 43.2 | 40.8 |
| O 2 P [ 4 ] | 85.4 | 69.7 | 22.3 | 45.2 | 44.4 | 46.9 | 66.7 | 57.8 | 56.2 | 13.5 | 46.1 | 32.3 | 41.2 | 59.1 | 55.3 | 51.0 | 36.2 | 50.4 | 27.8 | 46.9 | 44.6 | 47.6 |
| ours ( full+fg R-CNN fc 6 subscript fc 6 \textrm{fc}_{6} ) | 84.2 | 66.9 | 23.7 | 58.3 | 37.4 | 55.4 | 73.3 | 58.7 | 56.5 | 0 9.7 | 45.5 | 29.5 | 49.3 | 40.1 | 57.8 | 53.9 | 33.8 | 60.7 | 22.7 | 47.1 | 41.3 | 47.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: R-CNN Table 5

| VOC 2011 val | bg | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O 2 P [ 4 ] | 84.0 | 69.0 | 21.7 | 47.7 | 42.2 | 42.4 | 64.7 | 65.8 | 57.4 | 12.9 | 37.4 | 20.5 | 43.7 | 35.7 | 52.7 | 51.0 | 35.8 | 51.0 | 28.4 | 59.8 | 49.7 | 46.4 |
| full R-CNN fc 6 subscript fc 6 \textrm{fc}_{6} | 81.3 | 56.2 | 23.9 | 42.9 | 40.7 | 38.8 | 59.2 | 56.5 | 53.2 | 11.4 | 34.6 | 16.7 | 48.1 | 37.0 | 51.4 | 46.0 | 31.5 | 44.0 | 24.3 | 53.7 | 51.1 | 43.0 |
| full R-CNN fc 7 subscript fc 7 \textrm{fc}_{7} | 81.0 | 52.8 | 25.1 | 43.8 | 40.5 | 42.7 | 55.4 | 57.7 | 51.3 | 0 8.7 | 32.5 | 11.5 | 48.1 | 37.0 | 50.5 | 46.4 | 30.2 | 42.1 | 21.2 | 57.7 | 56.0 | 42.5 |
| fg R-CNN fc 6 subscript fc 6 \textrm{fc}_{6} | 81.4 | 54.1 | 21.1 | 40.6 | 38.7 | 53.6 | 59.9 | 57.2 | 52.5 | 0 9.1 | 36.5 | 23.6 | 46.4 | 38.1 | 53.2 | 51.3 | 32.2 | 38.7 | 29.0 | 53.0 | 47.5 | 43.7 |
| fg R-CNN fc 7 subscript fc 7 \textrm{fc}_{7} | 80.9 | 50.1 | 20.0 | 40.2 | 34.1 | 40.9 | 59.7 | 59.8 | 52.7 | 0 7.3 | 32.1 | 14.3 | 48.8 | 42.9 | 54.0 | 48.6 | 28.9 | 42.6 | 24.9 | 52.2 | 48.8 | 42.1 |
| full+fg R-CNN fc 6 subscript fc 6 \textrm{fc}_{6} | 83.1 | 60.4 | 23.2 | 48.4 | 47.3 | 52.6 | 61.6 | 60.6 | 59.1 | 10.8 | 45.8 | 20.9 | 57.7 | 43.3 | 57.4 | 52.9 | 34.7 | 48.7 | 28.1 | 60.0 | 48.6 | 47.9 |
| full+fg R-CNN fc 7 subscript fc 7 \textrm{fc}_{7} | 82.3 | 56.7 | 20.6 | 49.9 | 44.2 | 43.6 | 59.3 | 61.3 | 57.8 | 0 7.7 | 38.4 | 15.1 | 53.4 | 43.7 | 50.8 | 52.0 | 34.1 | 47.8 | 24.7 | 60.1 | 55.2 | 45.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: R-CNN Table 6

| class | AP | class | AP | class | AP | class | AP | class | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accordion | 50.8 | centipede | 30.4 | hair spray | 13.8 | pencil box | 11.4 | snowplow | 69.2 |
| airplane | 50.0 | chain saw | 14.1 | hamburger | 34.2 | pencil sharpener | 9.0 | soap dispenser | 16.8 |
| ant | 31.8 | chair | 19.5 | hammer | 9.9 | perfume | 32.8 | soccer ball | 43.7 |
| antelope | 53.8 | chime | 24.6 | hamster | 46.0 | person | 41.7 | sofa | 16.3 |
| apple | 30.9 | cocktail shaker | 46.2 | harmonica | 12.6 | piano | 20.5 | spatula | 6.8 |
| armadillo | 54.0 | coffee maker | 21.5 | harp | 50.4 | pineapple | 22.6 | squirrel | 31.3 |
| artichoke | 45.0 | computer keyboard | 39.6 | hat with a wide brim | 40.5 | ping-pong ball | 21.0 | starfish | 45.1 |
| axe | 11.8 | computer mouse | 21.2 | head cabbage | 17.4 | pitcher | 19.2 | stethoscope | 18.3 |
| baby bed | 42.0 | corkscrew | 24.2 | helmet | 33.4 | pizza | 43.7 | stove | 8.1 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

Object detection performance, as measured on the canonical PASCAL VOC dataset, has plateaued in the last few years. The best-performing methods are complex ensemble systems that typically combine multiple low-level image features with high-level context. In this paper, we propose a simple and scalable detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012---achieving a mAP of 53.3%. Our approach combines two key insights: 

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

它支持当前 FCOS proposal + ROI verifier 的路线：先高召回生成候选，再在候选区域上做更强分类和框质量验证。

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

- [[DPM]]
- [[Fast R-CNN]]
- [[Faster R-CNN]]
- [[FPN]]
