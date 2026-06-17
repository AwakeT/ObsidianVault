# 目标检测 Roadmap

> 日期：2026-06-11  
> 主题：Object Detection 从起源到当前研究路线  
> 目标：建立一条完整学习路径，并服务 DINO-GeoSemMap 检测头优化

---

## 0. 一句话理解目标检测

目标检测的任务是：

```text
输入图像
  -> 输出一组目标实例
  -> 每个实例包含 class / bbox / score
```

它一直围绕五件事演化：

1. **在哪里**：bbox 定位是否准。
2. **是什么**：类别预测是否准。
3. **有没有**：前景 / 背景判别是否稳。
4. **敢不敢报**：置信度是否校准。
5. **能不能跑**：速度、显存、部署是否可用。

---

## 1. 任务定义与评价指标

### 1.1 输出格式

目标检测输出：

```text
{class_id, bbox, score}
```

- `class_id`：类别。
- `bbox`：边界框，常见格式为 `[x1, y1, x2, y2]` 或 `[cx, cy, w, h]`。
- `score`：置信度，可能表示类别概率、前景概率、框质量，或它们的组合。

### 1.2 IoU

```text
IoU = area(pred ∩ gt) / area(pred ∪ gt)
```

| 阈值 | 含义 |
|---|---|
| IoU >= 0.5 | 宽松检测成功 |
| IoU >= 0.75 | 高质量定位 |
| [[COCO]] AP | IoU 从 0.5 到 0.95 多阈值平均 |

### 1.3 Precision / Recall

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

目标检测里的 TP 需要同时满足：

1. 类别正确。
2. IoU 达到阈值。
3. 没有被更高分预测框占用匹配。

DINO-GeoSemMap 当前关注：

```text
标定后 recall@conf0.5
```

这比普通 recall 更严格，因为它要求模型在高置信门槛下仍能找回目标。

### 1.4 AP / mAP

AP 是 precision-recall 曲线下面积。

| 指标 | 含义 |
|---|---|
| AP50 | IoU=0.5 的 AP |
| AP75 | IoU=0.75 的 AP |
| [[COCO]] AP | IoU=0.5:0.95 的平均 AP |
| mAP | 多类别 AP 平均 |

AP 适合论文比较，但工程部署还需要看：

- 固定 conf 阈值下 recall。
- 固定延迟下 recall。
- 按类别召回。
- FP 类型构成。
- 置信度校准。

### 1.5 NMS 与校准

NMS：

```text
1. 按 score 排序
2. 取最高分框
3. 删除与它 IoU 高的重复框
4. 重复直到结束
```

[[DETR]] 系尝试用 set prediction 去掉 NMS，但 YOLO / [[FCOS]] / [[RetinaNet]] 等 dense detector 通常仍依赖 NMS。

检测分数不一定等于真实正确概率。常见校准方式：

```text
temperature scaling
Platt scaling
isotonic regression
```

DINO-GeoSemMap 当前使用 Platt：

```text
score_calibrated = sigmoid(a * logit(score) + b)
```

原则：每改一次模型或 score 组合方式，都必须重新标定。

---

## 2. 总体演化脉络

```text
古典检测
  手工特征 + 滑窗 + SVM / Boosting
  Viola-Jones / HOG / DPM

CNN 两阶段检测
  proposal -> ROI feature -> classification + bbox regression
  R-CNN / Fast R-CNN / Faster R-CNN / FPN / Mask R-CNN

一阶段实时检测
  dense prediction on feature map
  YOLO / SSD / RetinaNet

Anchor-free 与质量估计
  point / center based detection
  FCOS / CenterNet / ATSS / GFL / TOOD / YOLOX

DETR set prediction
  object query + Hungarian matching + no NMS
  DETR / Deformable DETR / DN-DETR / DINO / RT-DETR / D-FINE / RF-DETR

开放词汇与基础模型
  vision-language grounding
  GLIP / Detic / Grounding DINO / OWLv2 / YOLO-World / Florence-2
```

---

## 3. 古典检测时代：手工特征到 [[DPM]]

时间大致为 2001-2013。

主流流程：

```text
图像金字塔
  -> 滑动窗口 / 候选区域
  -> 手工特征
  -> 分类器
  -> NMS
```

关键论文：

| 年份   | 论文                                                                                   | 贡献                                                                                      |                              |
| ---- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- | ---------------------------- |
| 2001 | [[Viola-Jones]], *Rapid Object Detection using a Boosted Cascade of Simple Features* | Haar 特征、Integral Image、AdaBoost、cascade，人脸检测实时化                                         |                              |
| 2005 | [[HOG                                                                                | Dalal & Triggs, Histograms of Oriented Gradients for Human Detection]]                  | [[HOG]] + SVM，行人检测经典范式       |
| 2010 | [[DPM                                                                                | Felzenszwalb et al., Object Detection with Discriminatively Trained Part-Based Models]] | [[DPM]]，可变形部件模型，深度学习前最强检测器之一 |

古典检测留下的思想遗产：

| 古典概念 | 现代对应 |
|---|---|
| 图像金字塔 | [[FPN]] / 多尺度特征 |
| 滑动窗口 | dense prediction |
| hard negative mining | focal loss / OHEM / background replay |
| cascade | [[Cascade R-CNN]] / ROI verifier |
| 部件模型 | deformable convolution / attention |
| 前景快速过滤 | objectness branch |

对 DINO-GeoSemMap 当前最有价值的是：

```text
hard negative mining
cascade / verifier
foreground filtering
```

---

## 4. CNN 两阶段检测：[[R-CNN]] 到 [[Mask R-CNN]]

两阶段检测器把问题拆成：

```text
Stage 1: 找可能有物体的位置
Stage 2: 判断类别并精修框
```

典型结构：

```text
image
  -> CNN backbone
  -> proposal / RPN
  -> ROI pooling / ROIAlign
  -> classification + bbox regression
```

关键论文：

| 论文 | 链接 | 贡献 |
|---|---|---|
| [[R-CNN]] | https://arxiv.org/abs/1311.2524 | CNN 检测起点，用 Selective Search proposals + CNN features |
| [[Fast R-CNN]] | 2015 | 整图只跑一次 CNN，在 feature map 上做 ROI pooling |
| [[Faster R-CNN]] | https://arxiv.org/abs/1506.01497 | RPN 替代 Selective Search，使 proposal 生成可训练 |
| [[FPN]] | https://arxiv.org/abs/1612.03144 | 高层语义 + 低层空间细节，多尺度检测基础 |
| [[Mask R-CNN]] | https://arxiv.org/abs/1703.06870 | 加入 mask branch，ROIAlign，检测与实例分割合流 |
| [[Cascade R-CNN]] | https://arxiv.org/abs/1712.00726 | 低质量框 -> 中质量框 -> 高质量框，递进 refinement |

[[Cascade R-CNN]] 对当前项目特别重要：它说明高质量检测常常需要递进验证和 refinement。

可以迁移为：

```text
FCOS proposals
  -> ROI verifier
  -> foreground / quality / bbox refinement
```

---

## 5. 一阶段检测：YOLO、[[SSD]]、[[RetinaNet]]

一阶段检测器直接在密集特征图上预测：

```text
feature map location
  -> class
  -> box
  -> objectness / quality
```

相比两阶段检测：

- 更快。
- 工程更简单。
- 但前景/背景不均衡更严重。

关键论文：

| 论文 | 链接 | 贡献 |
|---|---|---|
| [[YOLOv1]] | https://arxiv.org/abs/1506.02640 | 把检测看成一次前向回归，实时检测范式 |
| [[SSD]] | https://arxiv.org/abs/1512.02325 | 多尺度 feature map 上预测 default boxes |
| [[RetinaNet]] | https://arxiv.org/abs/1708.02002 | Focal Loss 解决 dense detector 前景/背景不均衡 |

YOLO 系列演化：

| 版本 | 主要特点 |
|---|---|
| [[YOLOv1]] | grid-based direct regression |
| YOLOv2 / YOLO9000 | anchor、多尺度、更多类别 |
| YOLOv3 | FPN-like 多尺度预测 |
| YOLOv4/v5/v6/v7/v8 | 工程优化、增强、neck/head/loss 迭代 |
| YOLOv10/v11 | 继续优化实时部署和 NMS/assignment |
| [[YOLOv12]] | attention-centric 实时检测尝试 |

[[RetinaNet]] 的 Focal Loss 核心：

```text
降低简单样本权重
提高 hard examples 权重
```

当前项目的 FP_bg 问题，本质上就是 dense detector 的经典难题：

```text
大量背景位置中有一部分会被模型误认为物体
```

---

## 6. Anchor-Free 与质量估计：[[FCOS]]、[[ATSS]]、[[GFL]]、[[TOOD]]、[[YOLOX]]

Anchor-based 检测依赖预设框：

```text
scale / aspect ratio / stride / matching threshold
```

Anchor-free 思路：

```text
直接用点、中心或像素位置预测目标
```

关键论文：

| 论文 | 链接 | 贡献 |
|---|---|---|
| [[FCOS]] | https://arxiv.org/abs/1904.01355 | anchor-free dense head，预测 class + ltrb + centerness |
| [[ATSS]] | https://arxiv.org/abs/1912.02424 | 自适应正负样本分配，指出 assignment 是关键 |
| [[GFL]] | https://arxiv.org/abs/2006.04388 | QFL + DFL，分类分数包含 localization quality |
| [[TOOD]] | https://arxiv.org/abs/2108.07755 | Task Alignment Learning，对齐分类和定位 |
| [[YOLOX]] | https://arxiv.org/abs/2107.08430 | decoupled head、objectness、SimOTA 动态匹配 |

### 6.1 [[FCOS]]

[[FCOS]] 预测：

```text
class
l, t, r, b
centerness
```

DINO-GeoSemMap 当前选择 [[FCOS]] 的关键原因：

```text
冻结 DINOv2 特征下，FCOS 能把 train recall 拉到 97%，证明密集头比旧 DETR 更稳。
```

### 6.2 [[ATSS]]

[[ATSS]] 的核心观点：

```text
anchor-based 与 anchor-free 的差异不是根本
正负样本分配才是关键
```

对当前项目价值：

- 减少错误正样本。
- 降低 FP_loc。
- 改善分类与定位分数对齐。

### 6.3 [[GFL]]：QFL + DFL

[[GFL]] 的核心：

```text
分类分数应该包含 localization quality
框回归不应只是单点估计
```

| 组件 | 作用 |
|---|---|
| QFL | 用 IoU 质量作为分类软标签 |
| DFL | 把边界距离预测成离散分布 |

DINO-GeoSemMap 已采用 QFL + DFL。但当前问题是：

```text
QFL 去掉 centerness 后，前景门控被削弱，FP_bg 上升。
```

因此下一步要补回独立 objectness。

### 6.4 [[TOOD]] / TAL 与 [[YOLOX]]

[[TOOD]] 关注：

```text
分类最优位置和定位最优位置可能不一致
```

[[YOLOX]] 值得借鉴：

- decoupled head。
- objectness branch。
- SimOTA 动态匹配。
- anchor-free 工程实践。

可迁移为：

```text
FCOS++ = FCOS dense proposal + YOLOX-style objectness + ATSS/TAL assignment + GFL quality
```

---

## 7. [[DETR]] 路线：Set Prediction 到实时 [[DETR]]

[[DETR]] 把检测改写为集合预测：

```text
object queries
  -> transformer decoder
  -> set of predictions
  -> Hungarian matching
```

目标：

- 去掉 anchor。
- 去掉 NMS。
- 端到端训练。

关键论文：

| 论文 | 链接 | 贡献 |
|---|---|---|
| [[DETR]] | https://arxiv.org/abs/2005.12872 | object query、Hungarian matching、no NMS、set loss |
| [[Deformable DETR]] | https://arxiv.org/abs/2010.04159 | 稀疏采样 cross-attention，多尺度，收敛更快 |
| [[DN-DETR]] | https://arxiv.org/abs/2203.01305 | denoising query，稳定匹配 |
| [[DINO]] | https://arxiv.org/abs/2203.03605 | denoising、mixed query selection、anchor refinement |
| [[RT-DETR]] | https://arxiv.org/abs/2304.08069 | hybrid encoder + query selection，实时 [[DETR]] |
| [[RT-DETRv2]] | https://arxiv.org/abs/2407.17140 | bag-of-freebies 与训练改进 |
| [[D-FINE]] | https://arxiv.org/abs/2410.13842 | fine-grained distribution refinement |
| [[DEIM]] | https://arxiv.org/abs/2412.04234 | Dense O2O matching |
| [[RF-DETR]] | https://arxiv.org/abs/2511.09554 | DINOv2 + Objects365 + weight-sharing NAS |

[[DETR]] 的问题：

- 收敛慢。
- 小物体差。
- 对数据和训练策略敏感。
- query 匹配不稳定。

DINO-GeoSemMap 当前不建议直接回到旧 [[DETR]] 头，因为已有实验证明：

- train recall 低。
- 匈牙利匹配不稳。
- 过训坍塌。

如果未来重试 [[DETR]]，应满足：

1. 使用 Deformable / [[RT-DETR]] / [[DINO]] / [[D-FINE]] 类现代结构。
2. 加 denoising query。
3. 用 Objects365 / [[COCO]] 预训练。
4. 保留 [[FCOS]] proposal 或 teacher，避免 query 从零学定位。

当前最现实借鉴：

```text
不是换成 DETR
而是借 DETR 系的 query selection / quality refinement / Objects365 训练流程
```

---

## 8. 开放词汇检测：Grounding 与 VLM

闭集检测：

```text
训练集有什么类，模型只能检测什么类
```

开放词汇检测：

```text
输入文本 prompt
模型检测对应视觉区域
```

关键论文：

| 论文 | 链接 | 贡献 |
|---|---|---|
| [[GLIP]] | https://arxiv.org/abs/2112.03857 | grounded language-image pretraining，统一 detection 和 phrase grounding |
| [[Detic]] | https://arxiv.org/abs/2201.02605 | 用 image-level labels 扩展到大词表检测 |
| [[Grounding DINO]] | https://arxiv.org/abs/2303.05499 | [[DINO]] 检测器与 grounded pretraining 结合 |
| [[OWLv2|OWLv2 / OWL-ST]] | https://arxiv.org/abs/2306.09683 | web-scale self-training，扩展开放词汇检测 |
| [[YOLO-World]] | https://arxiv.org/abs/2401.17270 | 实时开放词汇检测 |
| [[Florence-2]] | https://arxiv.org/abs/2311.06242 | prompt-based 多任务统一 |

开放词汇检测器适合作为：

- teacher。
- pseudo-label generator。
- 长尾类别补充。
- 真实环境数据标注工具。

但端侧生产检测器仍建议保持 specialist closed-set：

```text
Top-30 / Top-50 类
轻量
高置信可校准
速度稳定
```

---

## 9. 数据集与训练范式

目标检测能力很大程度由数据决定：

```text
类别覆盖
标注密度
场景多样性
长尾分布
框质量
是否穷尽标注
```

关键数据集：

| 数据集 | 价值 |
|---|---|
| PASCAL VOC | 早期标准基准 |
| [[COCO]] | 现代检测核心基准，复杂场景、多实例、AP 指标体系 |
| [[LVIS]] | 大词表、长尾、实例分割、非穷尽标注问题明显 |
| Objects365 | 大规模检测预训练，覆盖大量 [[COCO]] 缺失类别 |
| OpenImages | 大规模、多标签、开放类别 |
| Roboflow100 / ODinW | 真实世界跨域泛化评估 |

[[COCO]]：

- 链接：https://arxiv.org/abs/1405.0312
- 家居类干净。
- 但室内导航关键类别不足。

[[LVIS]]：

- 链接：https://arxiv.org/abs/1908.03195
- 大词表、长尾。
- 需要 federated loss / ignore 未标注类。

Objects365 对 DINO-GeoSemMap 特别重要的类别：

```text
coffee_table
nightstand
wardrobe
cabinet
bookshelf
door
window
washing_machine
lamp
curtain
```

推荐接入顺序：

1. annotations。
2. validation images。
3. 部分 train shards。
4. 全量 train。
5. 暂时跳过 test。

---

## 10. 2026 研究趋势

### 10.1 实时 [[DETR]] 继续增强

代表：

- [[RT-DETR]]。
- [[D-FINE]]。
- [[DEIM]]。
- [[RF-DETR]]。
- [[Le-DETR]]。

趋势：

```text
更快 encoder
更稳定 matching
更强 query selection
更细粒度 box refinement
更低训练成本
```

### 10.2 YOLO 仍是部署主力

YOLO 强在：

- 生态成熟。
- 部署路径完整。
- 速度快。
- 工程工具链丰富。

两条路线正在互相靠近：

```text
YOLO 吸收 attention / dynamic assignment / open-vocabulary
DETR 吸收实时部署和轻量化设计
```

### 10.3 质量估计成为核心

现代检测分数不应只是类别概率，还要回答：

```text
这个框准不准？
这个位置是不是前景？
这个类别和定位是否一致？
这个预测是否值得高置信输出？
```

相关方法：

- QFL。
- DFL。
- IoU prediction。
- objectness。
- TAL。
- cascade refinement。

### 10.4 Assignment 是检测器灵魂

很多检测器的关键差异不在 head，而在：

```text
哪些位置 / anchor / query 负责哪个 GT
```

代表：

- [[ATSS]]。
- SimOTA。
- TAL。
- Dense O2O。
- Hungarian matching。

对当前项目：

```text
若 FP_bg / FP_loc 高，必须检查 assignment，而不只是换 head。
```

### 10.5 开放词汇检测逐渐实用

[[Grounding DINO]]、[[YOLO-World]]、[[OWLv2]]、[[Florence-2]] 都在推进：

```text
文本查询 -> 视觉区域
```

但现实用法通常是：

```text
open-vocabulary model 作为 teacher
closed-set model 作为 student / deployment head
```

---

## 11. 关键论文阅读路线

### 11.1 入门必读

| 顺序 | 论文 | 链接 | 目的 |
|---|---|---|---|
| 1 | [[R-CNN]] | https://arxiv.org/abs/1311.2524 | CNN 检测起点 |
| 2 | [[Faster R-CNN]] | https://arxiv.org/abs/1506.01497 | RPN 与两阶段检测 |
| 3 | [[FPN]] | https://arxiv.org/abs/1612.03144 | 多尺度特征 |
| 4 | [[Mask R-CNN]] | https://arxiv.org/abs/1703.06870 | 检测与实例分割统一 |

### 11.2 一阶段检测

| 顺序 | 论文 | 链接 | 目的 |
|---|---|---|---|
| 1 | [[YOLOv1]] | https://arxiv.org/abs/1506.02640 | 实时检测范式 |
| 2 | [[SSD]] | https://arxiv.org/abs/1512.02325 | 多尺度 dense prediction |
| 3 | [[RetinaNet]] | https://arxiv.org/abs/1708.02002 | Focal Loss 与前景/背景不均衡 |

### 11.3 Anchor-Free 与质量估计

| 顺序 | 论文 | 链接 | 目的 |
|---|---|---|---|
| 1 | [[FCOS]] | https://arxiv.org/abs/1904.01355 | anchor-free dense head |
| 2 | [[ATSS]] | https://arxiv.org/abs/1912.02424 | 自适应正负样本分配 |
| 3 | [[GFL]] | https://arxiv.org/abs/2006.04388 | QFL / DFL / quality-aware score |
| 4 | [[TOOD]] | https://arxiv.org/abs/2108.07755 | 分类定位任务对齐 |
| 5 | [[YOLOX]] | https://arxiv.org/abs/2107.08430 | objectness / SimOTA / decoupled head |

### 11.4 [[DETR]] 系

| 顺序 | 论文 | 链接 | 目的 |
|---|---|---|---|
| 1 | [[DETR]] | https://arxiv.org/abs/2005.12872 | set prediction |
| 2 | [[Deformable DETR]] | https://arxiv.org/abs/2010.04159 | 多尺度稀疏 attention |
| 3 | [[DN-DETR]] | https://arxiv.org/abs/2203.01305 | denoising query |
| 4 | [[DINO]] | https://arxiv.org/abs/2203.03605 | 强 [[DETR]] 训练范式 |
| 5 | [[RT-DETR]] | https://arxiv.org/abs/2304.08069 | 实时 [[DETR]] |
| 6 | [[D-FINE]] | https://arxiv.org/abs/2410.13842 | 分布式框精修 |
| 7 | [[DEIM]] | https://arxiv.org/abs/2412.04234 | Dense O2O matching |
| 8 | [[RF-DETR]] | https://arxiv.org/abs/2511.09554 | DINOv2 + Objects365 + NAS |

### 11.5 开放词汇检测

| 顺序 | 论文 | 链接 | 目的 |
|---|---|---|---|
| 1 | [[GLIP]] | https://arxiv.org/abs/2112.03857 | grounded language-image pretraining |
| 2 | [[Detic]] | https://arxiv.org/abs/2201.02605 | 大词表检测与 image-level supervision |
| 3 | [[Grounding DINO]] | https://arxiv.org/abs/2303.05499 | 开放集 grounding detector |
| 4 | [[OWLv2]] | https://arxiv.org/abs/2306.09683 | web-scale open-vocabulary detection |
| 5 | [[YOLO-World]] | https://arxiv.org/abs/2401.17270 | 实时开放词汇检测 |
| 6 | [[Florence-2]] | https://arxiv.org/abs/2311.06242 | prompt-based 多任务统一 |

### 11.6 面向 DINO-GeoSemMap 的优先阅读

如果只为当前项目服务，优先读：

1. [[FCOS]]。
2. [[RetinaNet]] / Focal Loss。
3. [[GFL]]。
4. [[ATSS]]。
5. [[TOOD]]。
6. [[YOLOX]]。
7. [[Cascade R-CNN]]。
8. [[DINO]]。
9. [[D-FINE]]。
10. [[RF-DETR]]。

原因：

```text
当前项目最缺的是：
  高置信精度
  背景抑制
  框质量估计
  正负样本分配
  O365 大数据补类
```

---

## 12. 对 DINO-GeoSemMap 的启发

### 12.1 当前项目问题重述

当前检测头已经完成关键翻案：

```text
旧 DETR 头失败
  -> 不是冻结 DINOv2 特征完全不行
  -> 而是 DETR 头在当前设置下训练不稳、召回低

FCOS 头成功
  -> train recall 可达 97%
  -> D2a 后 conf0.05 召回容量约 89%
```

当前真正瓶颈：

```text
标定后 recall@conf0.5 约 31.8%
FP_bg 约 50.3%
FP_loc 约 39.8%
```

也就是说：

```text
模型能提出候选
但不能稳定区分真物体和背景
也不能把正确框推到足够高置信度
```

### 12.2 最适合的检测路线

推荐路线：

```text
FCOS(D2a+DFL)
  -> add objectness
  -> hard negative mining
  -> ATSS / TAL assignment
  -> FCOS proposal + ROI verifier
  -> Objects365 Top-30/Top-50 training
  -> optional D2b LoRA
```

原因：

1. 不推翻已经验证有效的 [[FCOS]]。
2. 直接攻击 FP_bg。
3. 保留冻结 DINOv2 backbone。
4. 与多头共享 forward 的架构兼容。
5. 可以自然接入分割 head。

### 12.3 为什么不是直接换 [[RF-DETR]]

[[RF-DETR]] 值得借鉴：

- DINOv2 先验。
- Objects365 训练。
- 实时检测设计。
- 质量-延迟 Pareto 搜索。
- 检测与分割共享特征。

但不建议当前直接替换为 [[RF-DETR]] 式 [[DETR]]：

```text
旧 DETR 在当前项目已有失败记录
当前瓶颈不是召回容量，而是 FP_bg 与高置信质量
重建 DETR 训练链路成本较高
```

更好的方式：

```text
借 RF-DETR 的训练流程和数据
保留 FCOS 作为 proposal 主体
再加 verifier / quality refinement
```

### 12.4 objectness 是第一优先级

当前 QFL 去掉 centerness 后，分数更像：

```text
class quality
```

但缺少独立回答：

```text
这里到底是不是前景物体？
```

因此下一步应加：

```text
objectness / foreground branch
```

最终分数：

```text
score = calibrated(cls_quality) * objectness * quality
```

预期：

- FP_bg 下降。
- 高置信框更可信。
- 标定后 recall@conf0.5 上升。

### 12.5 ROI verifier 是最可能冲高的结构

[[FCOS]] 已经能给出高召回 proposals。

下一步可让 verifier 专门回答：

```text
这个 proposal 是否真的是目标？
类别是否正确？
框是否足够准？
```

结构：

```text
FCOS proposal
  -> ROIAlign / token crop pooling
  -> verifier MLP / lightweight transformer
  -> objectness + class + IoU quality
```

这是一条轻量二阶段路线，借鉴 [[Faster R-CNN]] / [[Cascade R-CNN]]，但不重建完整大框架。

### 12.6 Objects365 的角色

Objects365 不是单纯扩大数据，而是补类别：

```text
coffee_table
nightstand
wardrobe
cabinet
bookshelf
door
window
washing_machine
```

接入原则：

1. 先 annotation + val。
2. 做类别映射。
3. 生成 Top-30/Top-50 manifest。
4. 小规模 train shards 验证。
5. 再决定是否全量下载和训练。

### 12.7 每轮实验纪律

每次改动都必须输出：

| 指标 | 目的 |
|---|---|
| raw recall@conf0.5 | 看原始分数 |
| Platt 参数 | 记录标定变化 |
| calibrated recall@conf0.5 | 唯一验收指标 |
| recall@conf0.05 | 召回容量 |
| Top-K proposal recall | proposal 能力 |
| FP 分解 | 看 FP_bg / FP_loc 是否下降 |
| TP 分数分布 | 看高置信是否改善 |
| per-class recall | 看长尾和 O365-only 类 |

核心原则：

```text
conf0.05 / top300 只诊断
calibrated conf0.5 才验收
```

