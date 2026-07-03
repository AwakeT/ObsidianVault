---
title: "DN-DETR: Accelerate DETR Training by Introducing Query DeNoising"
method_name: "DN-DETR"
authors: [Feng Li, Hao Zhang, Shilong Liu, Jian Guo, Lionel Ni, Lei Zhang]
year: 2022
venue: CVPR
tags: [object-detection, detr, denoising, query-training]
arxiv: https://arxiv.org/abs/2203.01305
created: 2026-06-11
image_source: online
---

# 论文笔记：DN-DETR: Accelerate DETR Training by Introducing Query DeNoising

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[DN-DETR]] |
| 作者 | [Feng Li, Hao Zhang, Shilong Liu, Jian Guo, Lionel Ni, Lei Zhang] |
| 年份 | 2022 |
| 会议/来源 | CVPR |
| 链接 | [arXiv](https://arxiv.org/abs/2203.01305) / [PDF](https://arxiv.org/pdf/2203.01305.pdf) |
| 相关工作 | [[DETR]]、[[Deformable DETR]]、[[DINO]] |

---

## 一句话总结

> DN-DETR 通过向 decoder 输入带噪 GT query 并训练模型恢复它们，显著稳定和加速 DETR 训练。

---

## 核心贡献

1. **范式贡献**：DN-DETR 通过向 decoder 输入带噪 GT query 并训练模型恢复它们，显著稳定和加速 DETR 训练。
2. **方法贡献**：向 decoder 输入带噪 GT label/box query，训练模型恢复原始 GT，从而稳定 DETR query 学习。
3. **工程/研究影响**：该工作成为后续 [[DETR]]、[[Deformable DETR]]、[[DINO]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

向 decoder 输入带噪 GT label/box query，训练模型恢复原始 GT，从而稳定 DETR query 学习。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[DETR]]、[[Deformable DETR]]。
- 后续影响：[[DINO]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\mathcal{L}=\mathcal{L}_{match}+\lambda_{dn}\mathcal{L}_{dn}
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2203.01305](https://ar5iv.labs.arxiv.org/html/2203.01305)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 10 张、Table 10 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: Convergence curve between our model DN-Deformable-DETR built upon Deformable DETR with denoising training and previous models under ResNet-50 backbone.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2203.01305/assets/cvpr2022/resources/ap_comp.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 denoising query 的构造方式；它能给 DETR 类 head 提供更密集、更稳定的早期监督。

### Figure 2: Figure 2: The I ​ S 𝐼 𝑆 IS of DAB-DETR and DN-DETR during training. For each method, we train 12 12 12 epoch on the same setting. We test the change of the Hungarian matching between each two epochs on the Validation set as the I ​ S 𝐼 𝑆 IS .

![Figure 2](https://ar5iv.labs.arxiv.org/html/2203.01305/assets/x1.png)

**图表解读**：重点看 denoising query 的构造方式；它能给 DETR 类 head 提供更密集、更稳定的早期监督。

### Figure 3: Figure 3: A comparison of DAB-DETR and DN-DETR on anchor-target distance.

![Figure 3](https://ar5iv.labs.arxiv.org/html/2203.01305/assets/x2.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 denoising query 的构造方式；它能给 DETR 类 head 提供更密集、更稳定的早期监督。

### Figure 4: Figure 4: (a)(b)Some examples of anchors and targets for DAB-DETR and DN-DETR, respectively. Each arrow starts from an anchor and points to a target. The color of each arrow shows its l 1 subscript 𝑙 1 l_{1} length and cooler colors denote shorter arrows.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2203.01305/assets/x3.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 denoising query 的构造方式；它能给 DETR 类 head 提供更密集、更稳定的早期监督。

### Figure 5: Figure 4: (a)(b)Some examples of anchors and targets for DAB-DETR and DN-DETR, respectively. Each arrow starts from an anchor and points to a target. The color of each arrow shows its l 1 subscript 𝑙 1 l_{1} length and cooler colors denote shorter arrows.

![Figure 5](https://ar5iv.labs.arxiv.org/html/2203.01305/assets/x4.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 denoising query 的构造方式；它能给 DETR 类 head 提供更密集、更稳定的早期监督。

### Figure 6: Figure 5: Comparison of the cross-attention part DAB-DETR and our DN-DETR (a)DAB-DETR directly uses dynamically updated anchor boxes to provide both a reference query point ( x , y ) 𝑥 𝑦 (x,y) and a reference anchor size ( w , h ) 𝑤 ℎ (w,h) to improve the cross-attention computation. (b) DN-DETR specifies the decoder embeddings as label embeddings and adds an indicator to differentiate the denoising task and matching task.

![Figure 6](https://ar5iv.labs.arxiv.org/html/2203.01305/assets/x5.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 denoising query 的构造方式；它能给 DETR 类 head 提供更密集、更稳定的早期监督。

### Figure 7: Figure 6: The overview of our training method. There are two parts of queries, namely the denoising part and the matching part. The denoising part contains ≥ 1 absent 1 \geq 1 denoising groups. The attention masks from the matching part to the denoising part and among denoising groups are set to 1 1 1 (block) to block information leakage. In the figure, the yellow, brown and green grids in the attention mask represent 0 0 (unblock) and grey grids represent 1 1 1 (block).

![Figure 7](https://ar5iv.labs.arxiv.org/html/2203.01305/assets/x6.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 denoising query 的构造方式；它能给 DETR 类 head 提供更密集、更稳定的早期监督。

### Figure 8: Figure 7: (a) Convergence curves of DAB-DETR and DN-DETR with ResNet-DC5-50. Before learning rate drop, DN-DETR achieves 40 40 40 AP in 20 20 20 epochs, while DAB-DETR needs 40 40 40 epochs. (b) Convergence curves of multi-scale models with ResNet- 50 50 50 . With learning rate drop, DN-Deformable-DETR achieves 47.8 47.8 47.8 AP in 30 epochs, which is 0.9 0.9 0.9 AP higher than the converged DAB-Deformable-DETR.

![Figure 8](https://ar5iv.labs.arxiv.org/html/2203.01305/assets/cvpr2022/resources/ap_dab_r50dc1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 denoising query 的构造方式；它能给 DETR 类 head 提供更密集、更稳定的早期监督。

### Figure 9: Figure 7: (a) Convergence curves of DAB-DETR and DN-DETR with ResNet-DC5-50. Before learning rate drop, DN-DETR achieves 40 40 40 AP in 20 20 20 epochs, while DAB-DETR needs 40 40 40 epochs. (b) Convergence curves of multi-scale models with ResNet- 50 50 50 . With learning rate drop, DN-Deformable-DETR achieves 47.8 47.8 47.8 AP in 30 epochs, which is 0.9 0.9 0.9 AP higher than the converged DAB-Deformable-DETR.

![Figure 9](https://ar5iv.labs.arxiv.org/html/2203.01305/assets/cvpr2022/resources/convergence_comp.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。重点看 denoising query 的构造方式；它能给 DETR 类 head 提供更密集、更稳定的早期监督。

### Figure 10: Figure 8: DN-DETR in different noise scales. We fix one noise scale to 0.4 0.4 0.4 and change the other. Noise scale is defined in 4.3

![Figure 10](https://ar5iv.labs.arxiv.org/html/2203.01305/assets/cvpr2022/resources/noise.png)

**图表解读**：重点看 denoising query 的构造方式；它能给 DETR 类 head 提供更密集、更稳定的早期监督。

### 关键表格

#### Table 1: DN-DETR Table 1

| Model | #epochs | AP | AP 50 | AP 75 | AP S | AP M | AP L | GFLOPs | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DETR-R 50 50 50 [ 1 ] | 500 500 500 | 42.0 42.0 42.0 | 62.4 62.4 62.4 | 44.2 44.2 44.2 | 20.5 20.5 20.5 | 45.8 45.8 45.8 | 61.1 61.1 61.1 | 86 86 86 | 41 41 41 M |
| Faster RCNN-FPN-R 50 50 50 [ 18 ] | 108 108 108 | 42.0 42.0 42.0 | 62.1 62.1 62.1 | 45.5 45.5 45.5 | 26.6 26.6 26.6 | 45.5 45.5 45.5 | 53.4 53.4 53.4 | 180 180 180 | 42 42 42 M |
| Anchor DETR-R 50 50 50 [ 21 ] | 50 50 50 | 42.1 42.1 42.1 | 63.1 63.1 63.1 | 44.9 44.9 44.9 | 22.3 22.3 22.3 | 46.2 46.2 46.2 | 60.0 60.0 60.0 | − - | 39 39 39 M |
| Conditional DETR-R 50 50 50 [ 15 ] | 50 50 50 | 40.9 40.9 40.9 | 61.8 61.8 61.8 | 43.3 43.3 43.3 | 20.8 20.8 20.8 | 44.6 44.6 44.6 | 59.2 59.2 59.2 | 90 90 90 | 44 44 44 M |
| DAB-DETR-R 50 50 50 [ 14 ] | 50 50 50 | 42.2 42.2 42.2 | 63.1 63.1 63.1 | 44.7 44.7 44.7 | 21.5 21.5 21.5 | 45.7 45.7 45.7 | 60.3 60.3 60.3 | 94 94 94 | 44 44 44 M |
| DN-DETR-R 50 50 50 | 50 50 50 | 44.1(+1.9) | 64.4 64.4 64.4 | 46.7 46.7 46.7 | 22.9 22.9 22.9 | 48.0 48.0 48.0 | 63.4 63.4 63.4 | 94 94 94 | 44 44 44 M |
| DETR-R 101 101 101 [ 1 ] | 500 500 500 | 43.5 43.5 43.5 | 63.8 63.8 63.8 | 46.4 46.4 46.4 | 21.9 21.9 21.9 | 48.0 48.0 48.0 | 61.8 61.8 61.8 | 152 152 152 | 60 60 60 M |
| Faster RCNN-FPN-R 101 101 101 [ 18 ] | 108 108 108 | 44.0 44.0 44.0 | 63.9 63.9 63.9 | 47.8 47.8 47.8 | 27.2 27.2 27.2 | 48.1 48.1 48.1 | 56.0 56.0 56.0 | 246 246 246 | 60 60 60 M |
| Anchor DETR-R 101 101 101 [ 21 ] | 50 50 50 | 43.5 43.5 43.5 | 64.3 64.3 64.3 | 46.6 46.6 46.6 | 23.2 23.2 23.2 | 47.7 47.7 47.7 | 61.4 61.4 61.4 | − - | 58 58 58 M |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: DN-DETR Table 2

| Model | MultiScale | #epochs | AP | AP 50 | AP 75 | AP S | AP M | AP L | GFLOPs | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN-FPN-R 50 50 50 1x [ 18 ] | ✓ ✓ \checkmark | 12 12 12 | 37.9 37.9 37.9 | 58.8 58.8 58.8 | 41.1 41.1 41.1 | 22.4 22.4 22.4 | 41.1 41.1 41.1 | 49.1 49.1 49.1 | 180 180 180 | 40 40 40 M |
| DETR-R 50 50 50 1x [ 1 ] |  | 12 12 12 | 15.5 15.5 15.5 | 29.4 29.4 29.4 | 14.5 14.5 14.5 | 4.3 4.3 4.3 | 15.1 15.1 15.1 | 26.7 26.7 26.7 | 86 86 86 | 41 41 41 M |
| DAB-DETR-DC5-R 50 50 50 [ 14 ] |  | 12 12 12 | 38.0 38.0 38.0 | 60.3 60.3 60.3 | 39.8 39.8 39.8 | 19.2 19.2 19.2 | 40.9 40.9 40.9 | 55.4 55.4 55.4 | 216 216 216 | 44 44 44 M |
| DN-DETR-DC5-R 50 50 50 |  | 12 12 12 | 41.7(+3.7) | 61.4 61.4 61.4 | 44.1 44.1 44.1 | 21.2 21.2 21.2 | 45.0 45.0 45.0 | 60.2 60.2 60.2 | 216 216 216 | 44 44 44 M |
| Deformable DETR-R 50 50 50 1x [ 25 ] | ✓ ✓ \checkmark | 12 12 12 | 37.2 37.2 37.2 | 55.5 55.5 55.5 | 40.5 40.5 40.5 | 21.1 21.1 21.1 | 40.7 40.7 40.7 | 50.5 50.5 50.5 | 173 173 173 | 40 40 40 M |
| Dynamic DETR-R 50 † superscript 50 † 50^{{\dagger}} 1x |  |  |  |  |  |  |  |  |  |  |
| w/o dynamic encoder |  |  |  |  |  |  |  |  |  |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: DN-DETR Table 3

| Model | MultiScale | #epochs | AP | AP 50 | AP 75 | AP S | AP M | AP L | GFLOPs | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Extending DN to other detection models |  |  |  |  |  |  |  |  |  |  |
| Anchor-DETR-DC5-R 50 50 50 [ 21 ] |  | 12 12 12 | 38.2 38.2 {38.2} | 58.6 58.6 58.6 | 40.6 40.6 40.6 | 20.3 20.3 20.3 | 41.9 41.9 41.9 | 53.1 53.1 53.1 | − - | 37 37 37 M |
| DN-Anchor-DETR-DC5-R 50 50 50 |  | 12 12 12 | 39.4(+1.2) | 59.1 59.1 59.1 | 41.8 41.8 41.8 | 19.6 19.6 19.6 | 43.4 43.4 43.4 | 56.0 56.0 56.0 | − - | 37 37 37 M |
| Group-DAB-DETR-DC5-R 50 50 50 [ 3 ] |  | 12 12 12 | 41.9 41.9 {41.9} | − - | − - | 23.3 23.3 23.3 | 45.6 45.6 45.6 | 58.4 58.4 58.4 | − - | − - M |
| DN-Group-DAB-DETR-DC5-R 50 ∗ superscript 50 50^{*} [ 3 ] |  | 12 12 12 | 44.5(+2.6) | − - | − - | 25.9 25.9 25.9 | 48.2 48.2 48.2 | 62.2 62.2 62.2 | − - | − - M |
| Faster R-CNN-FPN-R 50 50 50 [ 21 ] | ✓ ✓ \checkmark | 12 12 12 | 37.9 37.9 {37.9} | 58.8 58.8 58.8 | 41.1 41.1 41.1 | 22.4 22.4 22.4 | 41.1 41.1 41.1 | 49.1 49.1 49.1 | 180 180 180 | 40 40 40 M |
| DN-Faster R-CNN-FPN-R 50 50 50 | ✓ ✓ \checkmark | 12 12 12 | 38.4(+0.5) | 59.1 59.1 59.1 | 41.5 41.5 41.5 | 22.7 22.7 22.7 | 41.6 41.6 41.6 | 50.4 50.4 50.4 | 180 180 180 | 40 40 40 M |
| SAM-DETR++-R 50 50 50 [ 23 ] | ✓ ✓ \checkmark | 12 12 12 | 43.2 43.2 {43.2} | 61.5 61.5 61.5 | 46.5 46.5 46.5 | 25.5 25.5 25.5 | 46.5 46.5 46.5 | 58.6 58.6 58.6 | 203 203 203 | 55 55 55 M |
| DN-SAM-DETR++-R 50 ∗ superscript 50 50^{*} [ 23 ] | ✓ ✓ \checkmark | 12 12 12 | 44.8(+1.6) | 62.6 62.6 62.6 | 47.9 47.9 47.9 | 26.7 26.7 26.7 | 48.2 48.2 48.2 | 60.9 60.9 60.9 | 203 203 203 | 55 55 55 M |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: DN-DETR Table 4

| Model | MultiScale | #epochs | AP | AP 50 | AP 75 | AP S | AP M | AP L | GFLOPs | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deformable DETR-R 50 50 50 [ 25 ] | ✓ | 50 50 50 | 43.8 43.8 43.8 | 62.6 62.6 62.6 | 47.7 47.7 47.7 | 26.4 26.4 26.4 | 47.1 47.1 47.1 | 58.0 58.0 58.0 | 173 173 173 | 40 40 40 M |
| SMCA-R 50 50 50 [ 8 ] | ✓ | 50 50 50 | 43.7 43.7 43.7 | 63.6 63.6 63.6 | 47.2 47.2 47.2 | 24.2 24.2 24.2 | 47.0 47.0 47.0 | 60.4 60.4 60.4 | 152 152 152 | 40 40 40 M |
| TSP-RCNN-R 50 50 50 [ 19 ] | ✓ | 96 96 96 | 45.0 45.0 45.0 | 64.5 64.5 64.5 | 49.6 49.6 49.6 | 29.7 29.7 29.7 | 47.7 47.7 47.7 | 58.0 58.0 58.0 | 188 188 188 | − - |
| Dynamic DETR-R 50 ∗ superscript 50 50^{*} [ 6 ] | ✓ | 50 50 50 | 47.2 47.2 {47.2} | 65.9 65.9 65.9 | 51.1 51.1 51.1 | 28.6 28.6 28.6 | 49.3 49.3 49.3 | 59.1 59.1 59.1 | − - | − - |
| DAB-Deformable-DETR-R 50 50 50 | ✓ | 50 50 50 | 46.9 46.9 46.9 | 66.0 66.0 66.0 | 50.8 50.8 50.8 | 30.1 30.1 30.1 | 50.4 50.4 50.4 | 62.5 62.5 62.5 | 195 195 195 | 48 48 48 M |
| DN-Deformable-DETR-R 50 50 50 | ✓ | 50 | 48.6 | 67.4 67.4 67.4 | 52.7 52.7 52.7 | 31.0 31.0 31.0 | 52.0 52.0 52.0 | 63.7 63.7 63.7 | 195 195 195 | 48 48 48 M |
| DN-Deformable-DETR-R 50 50 50 ++ | ✓ | 50 | 49.5 | 67.6 67.6 67.6 | 53.8 53.8 53.8 | 31.3 31.3 31.3 | 52.6 52.6 52.6 | 65.4 65.4 65.4 | − - | 47 47 47 M |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: DN-DETR Table 5

| Box Denoising | Label Denoising | Attention Mask | AP |
| --- | --- | --- | --- |
| ✓ | ✓ | ✓ | 43.4 |
| ✓ |  | ✓ | 43.0 |
|  |  | ✓ | 42.2 |
| ✓ | ✓ |  | 24.0 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: DN-DETR Table 6

|  | No Group | 1 Group | 5 Groups |
| --- | --- | --- | --- |
| R50 | 42.2 | 43.4 | 44.1 |
| R50-DC5 | 44.5 | 45.6 | 46.3 |
| R101 | 43.5 | 45.0 | 45.2 |
| R101-DC5 | 45.8 | 46.5 | 47.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: DN-DETR Table 7

| Model | MultiScale | #epochs | AP | AP 50 | AP 75 | AP S | AP M | AP L | GFLOPs | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DAB-DETR-DC 5 5 5 -R 50 50 50 |  | 50 50 50 | 44.5 44.5 {44.5} | 65.1 65.1 65.1 | 47.7 47.7 47.7 | 25.3 25.3 25.3 | 48.2 48.2 48.2 | 62.3 62.3 62.3 | 202 202 202 | 44 44 44 M |
| DN-DETR-DC 5 5 5 -R 50 50 50 |  | 25 | 44.4 44.4 44.4 | 64.5 64.5 64.5 | 47.3 47.3 47.3 | 24.4 24.4 24.4 | 48.0 48.0 48.0 | 63.0 63.0 63.0 | 202 202 202 | 44 44 44 M |
| DAB-Deformable-DETR-R 50 50 50 | ✓ | 50 50 50 | 46.9 46.9 46.9 | 66.0 66.0 66.0 | 50.8 50.8 50.8 | 30.1 30.1 30.1 | 50.4 50.4 50.4 | 62.5 62.5 62.5 | 195 195 195 | 48 48 48 M |
| DN-Deformable-DETR-R 50 50 50 | ✓ | 25 | 46.8 46.8 46.8 | 65.5 65.5 65.5 | 50.8 50.8 50.8 | 28.9 28.9 28.9 | 50.2 50.2 50.2 | 62.5 62.5 62.5 | 195 195 195 | 48 48 48 M |
| DAB-Deformable-DETR-R 50 50 50 ++ | ✓ | 50 50 50 | 48.7 48.7 48.7 | 67.2 67.2 67.2 | 53.0 53.0 53.0 | 31.4 31.4 31.4 | 51.6 51.6 51.6 | 63.9 63.9 63.9 | − - | 47 47 47 M |
| DN-Deformable-DETR-R 50 50 50 ++ | ✓ | 25 | 48.4 48.4 48.4 | 66.6 66.6 66.6 | 52.7 52.7 52.7 | 30.0 30.0 30.0 | 51.7 51.7 51.7 | 64.4 64.4 64.4 | − - | 47 47 47 M |
| Vanilla-DETR-R 50 50 50 [ 1 ] |  | 500 500 500 | 42.0 42.0 42.0 | 62.4 62.4 62.4 | 44.2 44.2 44.2 | 20.5 20.5 20.5 | 45.8 45.8 45.8 | 61.1 61.1 61.1 | 86 86 86 | 41 41 41 M |
| DN-Vanilla-DETR-R 50 50 50 |  | 250 250 250 | 42.2 42.2 {42.2} | 61.8 61.8 61.8 | 44.6 44.6 44.6 | 20.5 20.5 20.5 | 46.0 46.0 46.0 | 61.3 61.3 61.3 | 86 86 86 | 37 37 37 M |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: DN-DETR Table 8

| Model | Total Training time (min) | Training GFLOPs |
| --- | --- | --- |
| DAB-DETR-R 50 50 50 | 2555 2555 2555 ( 50 50 50 epochs) | 94.4 94.4 {94.4} |
| DN-DAB-DETR-R 50 50 50 | 1443 1443 1443 ( 25 25 25 epochs) | 94.5 94.5 94.5 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 9: DN-DETR Table 9

| Method | Setting | AP | AP(Cond) |
| --- | --- | --- | --- |
| DAB-DETR | 0.7 / 0.3 0.7 0.3 0.7/0.3 | 38.4 38.4 38.4 | - |
| DN-DETR | 0.7 / 0.3 0.7 0.3 0.7/0.3 | 42.1 42.1 42.1 | 42.9 42.9 42.9 |
| DAB-DETR | 0.5 / 0.5 0.5 0.5 0.5/0.5 | 37.8 37.8 37.8 | - |
| DN-DETR | 0.5 / 0.5 0.5 0.5 0.5/0.5 | 39.1 39.1 39.1 | 40.3 40.3 40.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 10: DN-DETR Table 10

| Method | Setting | AP |
| --- | --- | --- |
| DAB-DETR | no knwon labels | 42.2 |
| DN-DETR | no knwon labels | 43.4 |
| DN-DETR | known label ( 1 1 1 ep) | 43.8 |
| DN-DETR | known label ( 10 10 10 ep) | 46.6 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

We present in this paper a novel denoising training method to speedup DETR (DEtection TRansformer) training and offer a deepened understanding of the slow convergence issue of DETR-like methods. We show that the slow convergence results from the instability of bipartite graph matching which causes inconsistent optimization goals in early training stages. To address this issue, except for the Hungarian loss, our method additionally feeds ground-truth bounding boxes with noises into Transformer de

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

- [[DETR]]
- [[Deformable DETR]]
- [[DINO]]
