---
title: "YOLOv12: Attention-Centric Real-Time Object Detectors"
method_name: "YOLOv12"
authors: [Yunjie Tian, Qixiang Ye, David Doermann]
year: 2025
venue: arXiv
tags: [object-detection, yolo, attention, real-time-detection]
arxiv: https://arxiv.org/abs/2502.12524
created: 2026-06-11
image_source: online
---

# 论文笔记：YOLOv12: Attention-Centric Real-Time Object Detectors

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[YOLOv12]] |
| 作者 | [Yunjie Tian, Qixiang Ye, David Doermann] |
| 年份 | 2025 |
| 会议/来源 | arXiv |
| 链接 | 无 arXiv 链接 / 经典论文 |
| 相关工作 | [[YOLOX]]、[[YOLO-World]]、[[RT-DETR]]、[[D-FINE]] |

---

## 一句话总结

> YOLOv12 将注意力机制系统引入实时 YOLO 框架，试图在保持 YOLO 速度的同时获得 attention 的全局建模能力。

---

## 核心贡献

1. **范式贡献**：YOLOv12 将注意力机制系统引入实时 YOLO 框架，试图在保持 YOLO 速度的同时获得 attention 的全局建模能力。
2. **方法贡献**：把 attention-centric 设计引入实时 YOLO，试图在 YOLO 部署效率和 attention 全局建模之间取得平衡。
3. **工程/研究影响**：该工作成为后续 [[YOLOX]]、[[YOLO-World]]、[[RT-DETR]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

把 attention-centric 设计引入实时 YOLO，试图在 YOLO 部署效率和 attention 全局建模之间取得平衡。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[YOLOX]]、[[YOLO-World]]。
- 后续影响：[[RT-DETR]]、[[D-FINE]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\text{Attn}(Q,K,V)=softmax(QK^\top/\sqrt{d})V
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2502.12524](https://ar5iv.labs.arxiv.org/html/2502.12524)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 5 张、Table 12 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 2 : Comparison of the representative local attention mechanisms with our area attention. Area Attention adopts the most straightforward equal partitioning way to divide the feature map into l l areas vertically or horizontally. (default is 4). This avoids complex operations while ensuring a large receptive field, resulting in high efficiency.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2502.12524/assets/x2.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 2: Figure 3 : The architecture comparison with popular modules including (a): CSPNet [ 55 ] , (b) ELAN [ 56 ] , (c) C3K2 (a case of GELAN) [ 58 , 28 ] , and (d) the proposed R-ELAN (residual efficient layer aggregation networks).

![Figure 2](https://ar5iv.labs.arxiv.org/html/2502.12524/assets/x3.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 3: Figure 4 : Comparison with popular methods in terms of accuracy-parameters (left) and accuracy-latency trade-off on CPU (right).

![Figure 3](https://ar5iv.labs.arxiv.org/html/2502.12524/assets/x4.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 4: Figure 5 : Comparison of heat maps between YOLOv10 [ 53 ] , YOLOv11 [ 28 ] , and the proposed YOLOv12. Compared to the advanced YOLOv10 and YOLOv11, YOLOv12 demonstrates a clearer perception of objects in the image. All the results are obtained using the X scale models. Zoom in to compare the details.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2502.12524/assets/x5.png)

**图表解读**：这是方法主图，优先拆解输入特征、neck/head、匹配策略和损失分支。阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### Figure 5: 1 Figure 1

![Figure 5](https://ar5iv.labs.arxiv.org/html/2502.12524/assets/x1.png)

**图表解读**：阅读时要把图中的模块、监督信号和实验指标对应到检测 pipeline：特征、候选、匹配、分类、定位和置信度校准。

### 关键表格

#### Table 1: YOLOv12 Table 1

| Method | FLOPs | #Param. | AP 50 : 95 v ​ a ​ l \text{AP}^{val}_{50:95} | AP 50 v ​ a ​ l \text{AP}^{val}_{50} | AP 75 v ​ a ​ l \text{AP}^{val}_{75} | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| (G) | (M) | (%) | (%) | (%) | (ms) |  |
| YOLOv6-3.0-N [ 32 ] | 11.4 | 4.7 | 37.0 | 52.7 | – | 2.69 |
| Gold-YOLO-N [ 54 ] | 12.1 | 5.6 | 39.6 | 55.7 | – | 2.92 |
| YOLOv8-N [ 24 ] | 8.7 | 3.2 | 37.4 | 52.6 | 40.5 | 1.77 |
| YOLOv10-N [ 53 ] | 6.7 | 2.3 | 38.5 | 53.8 | 41.7 | 1.84 |
| YOLO11-N [ 28 ] | 6.5 | 2.6 | 39.4 | 55.3 | 42.8 | 1.5 |
| YOLOv12-N (Ours) | 6.5 | 2.6 | 40.6 | 56.7 | 43.8 | 1.64 |
| YOLOv6-3.0-S [ 32 ] | 45.3 | 18.5 | 44.3 | 61.2 | – | 3.42 |
| Gold-YOLO-S [ 54 ] | 46.0 | 21.5 | 45.4 | 62.5 | – | 3.82 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: YOLOv12 Table 2

| Model | Vanilla | Re-Aggre. | Resi. | Scaling | Convergence | FLOPs (G) | #Param. (M) | AP 50 : 95 v ​ a ​ l \text{AP}^{val}_{50:95} (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv12-N | ✔ | ✗ | ✗ | – | ✔ | 6.9 | 2.7 | 40.8 |
| ✗ | ✔ | ✗ | – | ✔ | 6.5 | 2.6 | 40.6 |  |
| ✗ | ✗ | ✔ | 0.1 | ✔ | 6.5 | 2.6 | 40.3 |  |
| YOLOv12-L | ✔ | ✗ | ✗ | – | ✗ | – | – | – |
| ✗ | ✔ | ✗ | – | ✗ | – | – | – |  |
| ✗ | ✔ | ✔ | 0.1 | ✔ | 88.9 | 26.4 | 53.3 |  |
| ✗ | ✔ | ✔ | 0.01 | ✔ | 88.9 | 26.4 | 53.7 |  |
| ✗ | ✗ | ✔ | 0.01 | ✔ | 94.3 | 27.8 | 53.8 |  |
| YOLOv12-X | ✔ | ✗ | ✗ | – | ✗ | – | – | – |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: YOLOv12 Table 3

| Model | Scale | FLOPs | RTX 3080 | A5000 | A6000 |
| --- | --- | --- | --- | --- | --- |
| (G) |  |  |  |  |  |
| YOLOv9 [ 58 ] | T | 8.2 | 2.4/1.5 | 2.4/1.6 | 2.3/1.7 |
| S | 26.4 | 3.7/1.9 | 3.4/2.0 | 3.5/1.9 |  |
| M | 76.3 | 6.5/2.8 | 5.5/2.6 | 5.2/2.6 |  |
| C | 102.1 | 8.0/2.9 | 6.4/2.7 | 6.0/2.7 |  |
| E | 189.0 | 17.2/6.7 | 14.2/6.3 | 13.1/5.9 |  |
| YOLOv10 [ 53 ] | N | 6.7 | 1.6/1.0 | 1.6/1.0 | 1.6/1.0 |
| S | 21.6 | 2.8/1.4 | 2.4/1.4 | 2.4/1.3 |  |
| M | 59.1 | 5.7/2.5 | 4.5/2.4 | 4.2/2.2 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: YOLOv12 Table 4

| Method | AP 50 : 95 v ​ a ​ l \text{AP}^{val}_{50:95} | Latency |
| --- | --- | --- |
| Linear + LN {}_{+\text{LN}} | 40.5 | 1.68 |
| Linear + BN {}_{+\text{BN}} | 39.5 | 1.70 |
| Conv + BN {}_{+\text{BN}} | 40.6 | 1.64 |
| Conv + LN {}_{+\text{LN}} | 40.3 | 1.66 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 5: YOLOv12 Table 5

| Method | AP 50 : 95 v ​ a ​ l \text{AP}^{val}_{50:95} | Latency |
| --- | --- | --- |
| N/A | 38.3 | 1.60 |
| S 1 | 40.1 | 1.63 |
| S 4 | 39.8 | 1.71 |
| Ours | 40.6 | 1.64 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 6: YOLOv12 Table 6

| Epochs | AP v ​ a ​ l \text{AP}^{val} (N) | AP v ​ a ​ l \text{AP}^{val} (S) |
| --- | --- | --- |
| 300 | 39.5 | 47.0 |
| 500 | 40.3 | 47.8 |
| 600 | 40.6 | 48.0 |
| 800 | 40.4 | 47.7 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 7: YOLOv12 Table 7

| kernel | AP 50 : 95 v ​ a ​ l \text{AP}^{val}_{50:95} | Latency |
| --- | --- | --- |
| 3 × 3 3\times 3 | 40.4 | 1.60 |
| 5 × 5 5\times 5 | 40.4 | 1.61 |
| 7 × 7 7\times 7 | 40.6 | 1.64 |
| 9 × 9 9\times 9 | 40.7 | 1.79 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 8: YOLOv12 Table 8

| Pos. | AP 50 : 95 v ​ a ​ l \text{AP}^{val}_{50:95} | Latency |
| --- | --- | --- |
| RPE | 40.3 | 1.76 |
| APE | 40.5 | 1.69 |
| N/A | 40.6 | 1.64 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 9: YOLOv12 Table 9

| Area | AP 50 : 95 v ​ a ​ l \text{AP}^{val}_{50:95} | Latency |
| --- | --- | --- |
| ✗ | 40.8 | 1.70 |
| ✔ | 40.6 | 1.64 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 10: YOLOv12 Table 10

| Ratio (L) | AP 50 : 95 v ​ a ​ l \text{AP}^{val}_{50:95} | Latency |
| --- | --- | --- |
| 1.2 | 53.8 | 6.77 |
| 2.0 | 53.6 | 6.75 |
| 4.0 | 53.1 | 6.68 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 11: YOLOv12 Table 11

| FA | Latency (N) | Latency (S) |
| --- | --- | --- |
| ✗ | 1.92 | 3.02 |
| ✔ | 1.64 | 2.61 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 12: YOLOv12 Table 12

|  | AP 50 : 95 v ​ a ​ l \text{AP}^{val}_{50:95} (%) | AP 50 v ​ a ​ l \text{AP}^{val}_{50} (%) | AP 75 v ​ a ​ l \text{AP}^{val}_{75} (%) | AP s ​ m ​ a ​ l ​ l v ​ a ​ l \text{AP}^{val}_{small} (%) | AP m ​ e ​ d ​ i ​ u ​ m v ​ a ​ l \text{AP}^{val}_{medium} (%) | AP l ​ a ​ r ​ g ​ e v ​ a ​ l \text{AP}^{val}_{large} (%) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv12-N | 40.6 | 56.7 | 43.8 | 20.2 | 45.2 | 58.4 |
| YOLOv12-S | 48.0 | 65.0 | 51.8 | 29.8 | 53.2 | 65.6 |
| YOLOv12-M | 52.5 | 69.6 | 57.1 | 35.7 | 58.2 | 68.8 |
| YOLOv12-L | 53.7 | 70.7 | 58.5 | 36.9 | 59.5 | 69.9 |
| YOLOv12-X | 55.2 | 72.0 | 60.2 | 39.6 | 60.7 | 70.9 |

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

- [[YOLOX]]
- [[YOLO-World]]
- [[RT-DETR]]
- [[D-FINE]]
