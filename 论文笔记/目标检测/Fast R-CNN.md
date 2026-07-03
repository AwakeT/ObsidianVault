---
title: "Fast R-CNN"
method_name: "Fast R-CNN"
authors: [Ross Girshick]
year: 2015
venue: ICCV
tags: [object-detection, cnn, two-stage-detection, roi-pooling]
created: 2026-06-11
image_source: online
---

# 论文笔记：Fast R-CNN

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[Fast R-CNN]] |
| 作者 | [Ross Girshick] |
| 年份 | 2015 |
| 会议/来源 | ICCV |
| 链接 | 无 arXiv 链接 / 经典论文 |
| 相关工作 | [[R-CNN]]、[[Faster R-CNN]]、[[Mask R-CNN]] |

---

## 一句话总结

> 把 R-CNN 的逐 proposal CNN 前向改成整图共享特征和 ROI Pooling，大幅提升两阶段检测效率。

---

## 核心贡献

1. **范式贡献**：把 R-CNN 的逐 proposal CNN 前向改成整图共享特征和 ROI Pooling，大幅提升两阶段检测效率。
2. **方法贡献**：Fast R-CNN 先对整图提取 feature map，然后用 ROI Pooling 为每个 proposal 抽取固定大小特征，最后用共享 head 同时做分类和 bbox regression。
3. **工程/研究影响**：该工作成为后续 [[R-CNN]]、[[Faster R-CNN]]、[[Mask R-CNN]] 的重要参照，影响了检测器的结构、训练或评测方式。

---

## 问题背景

### 要解决的问题

R-CNN 对每个 proposal 都跑一次 CNN，重复计算极重，训练也被拆成多阶段。

### 现有方法的局限

- 早期或前一代方法通常在候选生成、特征表达、正负样本分配、框质量估计或开放类别泛化上存在瓶颈。
- 单看 AP 或 AP50 容易掩盖部署时的固定阈值可靠性；检测系统还必须处理 FP_bg、FP_loc、重复框和类别混淆。
- 对机器人场景而言，检测器还要考虑端侧延迟、长尾室内类别、置信度标定和与分割/深度头的协同。

### 本文的动机

本文试图用更清晰的检测范式或训练机制解决上述瓶颈，使检测器在精度、速度、可训练性或类别扩展能力上获得更好的平衡。

---

## 方法详解

### 模型架构 / 流程

Fast R-CNN 先对整图提取 feature map，然后用 ROI Pooling 为每个 proposal 抽取固定大小特征，最后用共享 head 同时做分类和 bbox regression。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[R-CNN]]、[[Faster R-CNN]]。
- 后续影响：[[Mask R-CNN]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
L(p,u,t^u,v)=L_{cls}(p,u)+\lambda [u\ge1]L_{loc}(t^u,v)
$$

**含义**：多任务损失同时优化分类和正样本框回归，是后续两阶段检测器的标准形式。

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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/1504.08083](https://ar5iv.labs.arxiv.org/html/1504.08083)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 4 张、Table 6 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: Fast R-CNN architecture. An input image and multiple regions of interest (RoIs) are input into a fully convolutional network. Each RoI is pooled into a fixed-size feature map and then mapped to a feature vector by fully connected layers (FCs). The network has two output vectors per RoI: softmax probabilities and per-class bounding-box regression offsets. The architecture is trained end-to-end with a multi-task loss.

![Figure 1](https://ar5iv.labs.arxiv.org/html/1504.08083/assets/x1.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 2: Figure 2: Timing for VGG16 before and after truncated SVD. Before SVD, fully connected layers fc6 and fc7 take 45% of the time.

![Figure 2](https://ar5iv.labs.arxiv.org/html/1504.08083/assets/x2.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 3: Figure 2: Timing for VGG16 before and after truncated SVD. Before SVD, fully connected layers fc6 and fc7 take 45% of the time.

![Figure 3](https://ar5iv.labs.arxiv.org/html/1504.08083/assets/x3.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 4: Figure 3: VOC07 test mAP and AR for various proposal schemes.

![Figure 4](https://ar5iv.labs.arxiv.org/html/1504.08083/assets/x4.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### 关键表格

#### Table 1: Fast R-CNN Table 1

|  | W ≈ U ​ Σ t ​ V T 𝑊 𝑈 subscript Σ 𝑡 superscript 𝑉 𝑇 W\approx U\Sigma_{t}V^{T} |  | (5) |
| --- | --- | --- | --- |
|  |  |  |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: Fast R-CNN Table 2

| method | train set | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | persn | plant | sheep | sofa | train | tv | mAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPPnet BB [ 11 ] † | 07 ∖ \setminus diff | 73.9 | 72.3 | 62.5 | 51.5 | 44.4 | 74.4 | 73.0 | 74.4 | 42.3 | 73.6 | 57.7 | 70.3 | 74.6 | 74.3 | 54.2 | 34.0 | 56.4 | 56.4 | 67.9 | 73.5 | 63.1 |
| R-CNN BB [ 10 ] | 07 | 73.4 | 77.0 | 63.4 | 45.4 | 44.6 | 75.1 | 78.1 | 79.8 | 40.5 | 73.7 | 62.2 | 79.4 | 78.1 | 73.1 | 64.2 | 35.6 | 66.8 | 67.2 | 70.4 | 71.1 | 66.0 |
| FRCN [ours] | 07 | 74.5 | 78.3 | 69.2 | 53.2 | 36.6 | 77.3 | 78.2 | 82.0 | 40.7 | 72.7 | 67.9 | 79.6 | 79.2 | 73.0 | 69.0 | 30.1 | 65.4 | 70.2 | 75.8 | 65.8 | 66.9 |
| FRCN [ours] | 07 ∖ \setminus diff | 74.6 | 79.0 | 68.6 | 57.0 | 39.3 | 79.5 | 78.6 | 81.9 | 48.0 | 74.0 | 67.4 | 80.5 | 80.7 | 74.1 | 69.6 | 31.8 | 67.1 | 68.4 | 75.3 | 65.5 | 68.1 |
| FRCN [ours] | 07+12 | 77.0 | 78.1 | 69.3 | 59.4 | 38.3 | 81.6 | 78.6 | 86.7 | 42.8 | 78.8 | 68.9 | 84.7 | 82.0 | 76.6 | 69.9 | 31.8 | 70.1 | 74.8 | 80.4 | 70.4 | 70.0 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: Fast R-CNN Table 3

| method | train set | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | persn | plant | sheep | sofa | train | tv | mAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BabyLearning | Prop. | 77.7 | 73.8 | 62.3 | 48.8 | 45.4 | 67.3 | 67.0 | 80.3 | 41.3 | 70.8 | 49.7 | 79.5 | 74.7 | 78.6 | 64.5 | 36.0 | 69.9 | 55.7 | 70.4 | 61.7 | 63.8 |
| R-CNN BB [ 10 ] | 12 | 79.3 | 72.4 | 63.1 | 44.0 | 44.4 | 64.6 | 66.3 | 84.9 | 38.8 | 67.3 | 48.4 | 82.3 | 75.0 | 76.7 | 65.7 | 35.8 | 66.2 | 54.8 | 69.1 | 58.8 | 62.9 |
| SegDeepM | 12+seg | 82.3 | 75.2 | 67.1 | 50.7 | 49.8 | 71.1 | 69.6 | 88.2 | 42.5 | 71.2 | 50.0 | 85.7 | 76.6 | 81.8 | 69.3 | 41.5 | 71.9 | 62.2 | 73.2 | 64.6 | 67.2 |
| FRCN [ours] | 12 | 80.1 | 74.4 | 67.7 | 49.4 | 41.4 | 74.2 | 68.8 | 87.8 | 41.9 | 70.1 | 50.2 | 86.1 | 77.3 | 81.1 | 70.4 | 33.3 | 67.0 | 63.3 | 77.2 | 60.0 | 66.1 |
| FRCN [ours] | 07++12 | 82.0 | 77.8 | 71.6 | 55.3 | 42.4 | 77.3 | 71.7 | 89.3 | 44.5 | 72.1 | 53.7 | 87.7 | 80.0 | 82.5 | 72.7 | 36.6 | 68.7 | 65.4 | 81.1 | 62.7 | 68.8 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: Fast R-CNN Table 4

| method | train set | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | persn | plant | sheep | sofa | train | tv | mAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BabyLearning | Prop. | 78.0 | 74.2 | 61.3 | 45.7 | 42.7 | 68.2 | 66.8 | 80.2 | 40.6 | 70.0 | 49.8 | 79.0 | 74.5 | 77.9 | 64.0 | 35.3 | 67.9 | 55.7 | 68.7 | 62.6 | 63.2 |
| NUS_NIN_c2000 | Unk. | 80.2 | 73.8 | 61.9 | 43.7 | 43.0 | 70.3 | 67.6 | 80.7 | 41.9 | 69.7 | 51.7 | 78.2 | 75.2 | 76.9 | 65.1 | 38.6 | 68.3 | 58.0 | 68.7 | 63.3 | 63.8 |
| R-CNN BB [ 10 ] | 12 | 79.6 | 72.7 | 61.9 | 41.2 | 41.9 | 65.9 | 66.4 | 84.6 | 38.5 | 67.2 | 46.7 | 82.0 | 74.8 | 76.0 | 65.2 | 35.6 | 65.4 | 54.2 | 67.4 | 60.3 | 62.4 |
| FRCN [ours] | 12 | 80.3 | 74.7 | 66.9 | 46.9 | 37.7 | 73.9 | 68.6 | 87.7 | 41.7 | 71.1 | 51.1 | 86.0 | 77.8 | 79.8 | 69.8 | 32.1 | 65.5 | 63.8 | 76.4 | 61.7 | 65.7 |
| FRCN [ours] | 07++12 | 82.3 | 78.4 | 70.8 | 52.3 | 38.7 | 77.8 | 71.6 | 89.3 | 44.2 | 73.0 | 55.0 | 87.5 | 80.5 | 80.8 | 72.0 | 35.1 | 68.3 | 65.7 | 80.4 | 64.2 | 68.4 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: Fast R-CNN Table 5

|  | layers that are fine-tuned in model L | SPPnet L |  |  |
| --- | --- | --- | --- | --- |
|  | ≥ \geq fc6 | ≥ \geq conv3_1 | ≥ \geq conv2_1 | ≥ \geq fc6 |
| VOC07 mAP | 61.4 | 66.9 | 67.2 | 63.1 |
| test rate (s/im) | 0.32 | 0.32 | 0.32 | 2.3 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: Fast R-CNN Table 6

| method | classifier | S | M | L |
| --- | --- | --- | --- | --- |
| R-CNN [ 9 , 10 ] | SVM | 58.5 | 60.2 | 66.0 |
| FRCN [ours] | SVM | 56.3 | 58.7 | 66.8 |
| FRCN [ours] | softmax | 57.1 | 59.2 | 66.9 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

相比 R-CNN 显著加快训练和推理，同时保持强检测精度。

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

proposal 仍依赖 Selective Search，外部候选生成仍是瓶颈。

---

## 对 DINO-GeoSemMap 的启发

DINO-GeoSemMap 的 ROI verifier 应复用 DINOv2/adapter feature，而不是为每个 proposal 重跑 backbone。

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
- [[Mask R-CNN]]
