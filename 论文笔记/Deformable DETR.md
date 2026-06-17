---
title: "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
method_name: "Deformable DETR"
authors: [Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai]
year: 2020
venue: ICLR
tags: [object-detection, detr, deformable-attention, transformer]
arxiv: https://arxiv.org/abs/2010.04159
created: 2026-06-11
image_source: online
---

# 论文笔记：Deformable DETR: Deformable Transformers for End-to-End Object Detection

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | [[Deformable DETR]] |
| 作者 | [Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai] |
| 年份 | 2020 |
| 会议/来源 | ICLR |
| 链接 | [arXiv](https://arxiv.org/abs/2010.04159) / [PDF](https://arxiv.org/pdf/2010.04159.pdf) |
| 相关工作 | [[DETR]]、[[DN-DETR]]、[[DINO]]、[[RT-DETR]] |

---

## 一句话总结

> Deformable DETR 将 DETR 的全局 attention 改为围绕参考点的稀疏采样 attention，大幅提升收敛速度和小物体检测能力。

---

## 核心贡献

1. **范式贡献**：Deformable DETR 将 DETR 的全局 attention 改为围绕参考点的稀疏采样 attention，大幅提升收敛速度和小物体检测能力。
2. **方法贡献**：把 cross-attention 改成围绕 reference points 的少量可学习采样点，显著加快 DETR 收敛。
3. **工程/研究影响**：该工作成为后续 [[DETR]]、[[DN-DETR]]、[[DINO]] 的重要参照，影响了检测器的结构、训练或评测方式。

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

把 cross-attention 改成围绕 reference points 的少量可学习采样点，显著加快 DETR 收敛。

### 核心模块

1. **特征表示**：决定检测器能否把局部纹理、形状、语义和上下文组织成可用于分类与定位的表示。
2. **候选/位置机制**：可能是滑窗、proposal、anchor、anchor-free point、object query 或文本 prompt。
3. **监督与匹配**：决定哪些位置或 query 被当作正样本，以及分类分数是否能反映定位质量。
4. **后处理/输出**：可能依赖 NMS、cascade refinement、set prediction 或开放词汇文本匹配。

### 与前后方法的关系

- 前置脉络：[[DETR]]、[[DN-DETR]]。
- 后续影响：[[DINO]]、[[RT-DETR]]。

---

## 关键公式

### 公式1：核心目标 / 评分形式

$$
\sum_{m,k}A_{mqk}W_m x(p_q+\Delta p_{mqk})
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

> 图表来源：[https://ar5iv.labs.arxiv.org/html/2010.04159](https://ar5iv.labs.arxiv.org/html/2010.04159)；图片使用网络链接，未下载到本地。
> 自动提取到 Figure 39 张、Table 4 张；下方按论文 HTML 顺序保留。

### Figure 1: Figure 1: Illustration of the proposed Deformable DETR object detector.

![Figure 1](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/x1.png)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 2: Figure 2: Illustration of the proposed deformable attention module.

![Figure 2](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/x2.png)

**图表解读**：这是可视化结果，重点看失败样例、遮挡、小目标、密集目标和背景误检。重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 3: Figure 3: Convergence curves of Deformable DETR and DETR-DC5 on COCO 2017 val set. For Deformable DETR, we explore different training schedules by varying the epochs at which the learning rate is reduced (where the AP score leaps).

![Figure 3](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/x3.png)

**图表解读**：这是实验对比图/表，重点看 AP、AP50、AP75、FPS/latency 与模型规模之间的取舍。重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 4: Figure 4: Constructing mult-scale feature maps for Deformable DETR.

![Figure 4](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/x4.png)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 5: Refer to caption

![Figure 5](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_bx_000000077396_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 6: Refer to caption

![Figure 6](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_bx_000000011760_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 7: Refer to caption

![Figure 7](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_bx_000000084170_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 8: Refer to caption

![Figure 8](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_by_000000077396_00_3_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 9: Refer to caption

![Figure 9](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_by_000000011760_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 10: Refer to caption

![Figure 10](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_by_000000084170_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 11: Refer to caption

![Figure 11](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_bw_000000077396_00_3_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 12: Refer to caption

![Figure 12](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_bw_000000011760_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 13: Refer to caption

![Figure 13](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_bw_000000084170_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 14: Refer to caption

![Figure 14](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_bh_000000077396_00_3_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 15: Refer to caption

![Figure 15](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_bh_000000011760_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 16: Refer to caption

![Figure 16](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_bh_000000084170_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 17: Refer to caption

![Figure 17](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_cs_000000077396_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 18: Refer to caption

![Figure 18](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_cs_000000011760_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 19: Refer to caption

![Figure 19](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_look_at/grad_cs_000000084170_00_8_infnorm.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 20: Refer to caption

![Figure 20](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_encoder_attention/tv_laptop_cat_1.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 21: Refer to caption

![Figure 21](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_encoder_attention/tv_laptop_cat_2.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 22: Refer to caption

![Figure 22](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_encoder_attention/tv_laptop_cat_3.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 23: Refer to caption

![Figure 23](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/x5.png)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 24: Refer to caption

![Figure 24](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_encoder_attention/three_zebra_1.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 25: Refer to caption

![Figure 25](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_encoder_attention/three_zebra_2.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 26: Refer to caption

![Figure 26](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_encoder_attention/three_zebra_3.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 27: Refer to caption

![Figure 27](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_encoder_attention/two_sofa_and_tv_1.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 28: Refer to caption

![Figure 28](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_encoder_attention/two_sofa_and_tv_2.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 29: Refer to caption

![Figure 29](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_encoder_attention/two_sofa_and_tv_3.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 30: Refer to caption

![Figure 30](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_decoder_attention/bus_and_two_car_1.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 31: Refer to caption

![Figure 31](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_decoder_attention/bus_and_two_car_2.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 32: Refer to caption

![Figure 32](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_decoder_attention/bus_and_two_car_3.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 33: Refer to caption

![Figure 33](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/x6.png)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 34: Refer to caption

![Figure 34](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_decoder_attention/moto_and_two_person_1.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 35: Refer to caption

![Figure 35](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_decoder_attention/moto_and_two_person_2.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 36: Refer to caption

![Figure 36](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_decoder_attention/moto_and_two_person_3.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 37: Refer to caption

![Figure 37](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_decoder_attention/tv_and_two_cat_1.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 38: Refer to caption

![Figure 38](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_decoder_attention/tv_and_two_cat_2.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### Figure 39: Refer to caption

![Figure 39](https://ar5iv.labs.arxiv.org/html/2010.04159/assets/fig/viz_decoder_attention/tv_and_two_cat_3.jpg)

**图表解读**：重点看 reference point 周围稀疏采样 attention；它说明冻结 DINOv2 特征上做轻量 decoder 时必须避免全局 dense attention 过重。

### 关键表格

#### Table 1: Deformable DETR Table 1

| Method | Epochs | AP | AP 50 50 {}_{\text{50}} | AP 75 75 {}_{\text{75}} | AP S S {}_{\text{S}} | AP M M {}_{\text{M}} | AP L L {}_{\text{L}} | params | FLOPs | Training |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU hours |  |  |  |  |  |  |  |  |  |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 2: Deformable DETR Table 2

| Inference |
| --- |
| FPS |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 3: Deformable DETR Table 3

| MS inputs | MS attention | K | FPNs | AP | AP 50 50 {}_{\text{50}} | AP 75 75 {}_{\text{75}} | AP S S {}_{\text{S}} | AP M M {}_{\text{M}} | AP L L {}_{\text{L}} |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ✓ ✓ \checkmark | ✓ ✓ \checkmark | 4 | FPN (Lin et al., 2017a ) | 43.8 | 62.6 | 47.8 | 26.5 | 47.3 | 58.1 |
| ✓ ✓ \checkmark | ✓ ✓ \checkmark | 4 | BiFPN (Tan et al., 2020 ) | 43.9 | 62.5 | 47.7 | 25.6 | 47.4 | 57.7 |
|  |  | 1 | w/o | 39.7 | 60.1 | 42.4 | 21.2 | 44.3 | 56.0 |
| ✓ ✓ \checkmark |  | 1 | 41.4 | 60.9 | 44.9 | 24.1 | 44.6 | 56.1 |  |
| ✓ ✓ \checkmark |  | 4 | 42.3 | 61.4 | 46.0 | 24.8 | 45.1 | 56.3 |  |
| ✓ ✓ \checkmark | ✓ ✓ \checkmark | 4 | 43.8 | 62.6 | 47.7 | 26.4 | 47.1 | 58.0 |  |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

#### Table 4: Deformable DETR Table 4

| Method | Backbone | TTA | AP | AP 50 50 {}_{\text{50}} | AP 75 75 {}_{\text{75}} | AP S S {}_{\text{S}} | AP M M {}_{\text{M}} | AP L L {}_{\text{L}} |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCOS (Tian et al., 2019 ) | ResNeXt-101 |  | 44.7 | 64.1 | 48.4 | 27.6 | 47.5 | 55.6 |
| ATSS (Zhang et al., 2020 ) | ResNeXt-101 + DCN | ✓ ✓ \checkmark | 50.7 | 68.9 | 56.3 | 33.2 | 52.9 | 62.4 |
| TSD (Song et al., 2020 ) | SENet154 + DCN | ✓ ✓ \checkmark | 51.2 | 71.9 | 56.0 | 33.8 | 54.8 | 64.2 |
| EfficientDet-D7 (Tan et al., 2020 ) | EfficientNet-B6 |  | 52.2 | 71.4 | 56.3 | - | - | - |
| Deformable DETR | ResNet-50 |  | 46.9 | 66.4 | 50.8 | 27.7 | 49.7 | 59.9 |
| Deformable DETR | ResNet-101 |  | 48.7 | 68.1 | 52.9 | 29.1 | 51.5 | 62.0 |
| Deformable DETR | ResNeXt-101 |  | 49.0 | 68.5 | 53.2 | 29.7 | 51.7 | 62.8 |
| Deformable DETR | ResNeXt-101 + DCN |  | 50.1 | 69.7 | 54.6 | 30.6 | 52.8 | 64.7 |
| Deformable DETR | ResNeXt-101 + DCN | ✓ ✓ \checkmark | 52.3 | 71.9 | 58.1 | 34.4 | 54.4 | 65.6 |

**表格解读**：优先读取强基线、本文方法、消融项和速度/精度指标；把 AP50/AP75 与 recall/precision 分开看，避免只凭单一 AP 判断是否适合当前项目。

---

## 实验结果

DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better

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
- [[DN-DETR]]
- [[DINO]]
- [[RT-DETR]]
