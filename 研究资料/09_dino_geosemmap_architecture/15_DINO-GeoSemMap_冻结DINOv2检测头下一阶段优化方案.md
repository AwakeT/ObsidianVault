# DINO-GeoSemMap 冻结 DINOv2 检测头下一阶段优化方案

> 版本：v0.1  
> 日期：2026-06-10  
> 关联：[13 检测头优化方案](13_DINO-GeoSemMap_检测头优化方案.md) · [14 检测头优化实施记录](14_DINO-GeoSemMap_检测头优化实施记录.md) · [10 数据集构成](10_DINO-GeoSemMap_数据集构成.md) · [RF-DETR 笔记](../../论文笔记/RF-DETR.md)

---

## 0. 文档定位

本文记录检测头下一阶段的优化路线：在 **DINOv2 backbone 冻结** 的前提下，继续推进目标检测头，使可部署口径 **标定后 `recall@conf0.5` 从当前约 31.8% 向 70% 逼近**。

核心判断：

- 当前检测头已经不是"看不到物体"的问题。D2a 后 `conf0.05` 召回容量约 89%，说明冻结 DINOv2 + 私有 adapter + FCOS 能提出足够多候选。
- 当前真正瓶颈是"高置信精度"：背景幻觉 FP_bg 已成为第一大假阳来源，定位误差 FP_loc 仍是第二大来源。
- 因此下一阶段不应盲目换大头，而应围绕 **前景/背景判别、框质量估计、二阶段验证、Objects365 数据覆盖** 组合推进。

一句话方案：

> 用 **FCOS++ 高召回 proposal** 保住召回容量，再用 **objectness / hard negative / task-aligned assignment / ROI verifier** 抑制背景幻觉，最后用 Objects365 补齐室内长尾类别覆盖。

---

## 0.5 实验进展与结论修正（2026-06-11，实测后回写）

> 本节为执行 §4 候选方法后的**实测结果**，部分推翻了 §0/§1.1/§4 的原始假设。冲突处以本节为准。
> 口径：val = COCO val2017（det_coco_L1，1504 图，11 类 det11）；硬验收 = Platt 标定后 recall@conf0.5。

### 实验结果汇总

| 实验 | 关键变量（均含 D2a adapter、冻结 backbone） | @.05 容量 | raw @0.5 | 标定后 @0.5 | 判定 |
|---|---|---|---|---|---|
| **E3** ✅生产 | QFL+DFL + **TAL 分配**，**无 objectness** | **0.913** | 0.402 | **0.454** | 当前最佳，可部署 |
| E1 | + objectness 分支（score=cls×obj）+ center 分配 | 0.586 | 0.006 | — | ❌ objectness 伤容量 |
| E2 | + objectness + **OHEM**（ratio 3 与 10） | ~0.75 | ~0.08 | — | ❌ 护栏触发 |

E3 标定：`score_cal = sigmoid(2.425·logit(s)+0.376)`；标定后 recall 曲线 conf 0.3/0.4/**0.5**/0.6/0.7 = 0.564/0.506/**0.454**/0.398/0.338。

### 三条关键修正

1. **E3 实际无 objectness 分支**（ckpt `use_objectness=False`、det_head 无 obj 权重）。E3 = 纯 QFL(cls 目标=IoU) + DFL + TAL。§0/§1.1 中"E3 已含 objectness"的描述有误。

2. **objectness / OHEM 路线（E1、E2）整体作废**。加 objectness 乘法门控每次都压低容量（E1=0.59、E2=0.75，均 < E3 的 0.92）。根因：dense head 上位置数 L≈49000 ≫ 正样本 n_pos≈数十，OHEM"丢弃易负例"会丢掉 ~99% 负例 → objectness 全图校准崩坏 → 容量塌。§4.1(E1)/§4.2(E2) 判定为死路。

3. **"特征混叠 → 必须 D2b"的前提不成立**。诊断 `diag_det_obj_sep.py`：E3 的 **cls_quality 对 TP-vs-FP_bg 的 AUC = 0.883**（TP 均值 0.526 vs FP_bg 0.283）→ 冻结 DINOv2 + D2a 特征**已足够可分**。故 §4.6 的 D2b **暂缓**，先不动 backbone。

### FP 成分（E3, thr0.2，diag_det_fp）

FP_bg **49.0%** + FP_loc **42.4%** + FP_cls 3.9% + FP_dup 4.6%（precision 0.298 / recall 0.738 @thr0.2）。

### 修正后的下一步方向 = 攻 FP_loc（定位质量），而非 FP_bg

FP_loc 占 42%，且 IoU∈[0.1,0.5) 的框**本质是"定位差一点的真阳"**，直接压着 recall@IoU0.5。而 QFL 的 cls 目标就是 IoU → **框定位变准 → 这些框 cls_quality 自动升过阈值 → 同时砍 FP_loc + 提 recall@conf0.5（两头通吃）**。候选：强化 DFL（lambda_dfl↑ / reg_max↑ / GFLv2 分布统计辅助 cls）、调 TAL 的 β、标定层榨取已有 0.88 可分性。E4(ROI verifier)/E5(Objects365)/E6(D2b) 仍为后备，按需启用。

> 产物：`scripts/diag_det_obj_sep.py`（可分性探针）、`checkpoints/fcos_e3/best.pt`（含 calib）、失败记录 `checkpoints/fcos_e2_r3` / `fcos_e2_r10` + `logs/fcos_e2_r3.log`。

---

## 1. 当前基线与目标

### 1.1 当前基线

来自 14 号实施记录：

> ⚠️ 本表为 v0.1 初稿数据；**实测后更新见 §0.5**（生产头为 **E3**：QFL+DFL+**TAL**，**无 objectness**；标定后 recall@conf0.5 = **45.4%**）。

| 项 | 当前状态（v0.1） | 实测更新（2026-06-11，§0.5） |
|---|---|---|
| 生产头 | FCOS 密集 anchor-free head | E3 = QFL+DFL+**TAL**，**无 objectness** |
| Backbone | DINOv2 冻结 | 不变；诊断 cls AUC 0.88 → 暂不上 D2b |
| Adapter | D2a 检测私有 adapter 已训练 | 不变 |
| 损失 | QFL + DFL + Platt 标定 | 不变 |
| 召回容量 | `conf0.05` 约 89% | **91.3%** |
| 可部署指标 | 标定后 `recall@conf0.5` 约 31.8% | **45.4%** |
| 主要 FP | FP_bg 约 50.3%，FP_loc 约 39.8% | FP_bg **49.0%** / FP_loc **42.4%**（thr0.2） |

### 1.2 目标口径

唯一硬验收口径：

```text
标定后 recall@conf0.5
```

目标：

```text
recall@conf0.5 >= 70%
```

诊断口径仍保留，但不作为验收：

- `conf0.05`：能力上限参考。
- `top300`：proposal 召回参考。
- raw `conf0.5`：观察训练刻度，但最终必须重做标定。

---

## 2. RF-DETR 可借鉴点

RF-DETR 本身是 DETR 系路线，但当前项目已经证明普通 DETR 头在冻结特征 + 小数据下存在数据饥饿、匈牙利匹配不稳和过训坍塌。因此本阶段不建议直接回到原 DETR 头。

真正值得借鉴的是 **训练流程和系统设计**，而不是简单照搬检测头。

### 2.1 可直接借鉴

| RF-DETR 机制 | 对本项目的借鉴方式 |
|---|---|
| DINOv2 视觉先验 | 继续保持 DINOv2 冻结，不破坏共享 forward |
| Objects365 预训练 | 引入 Objects365，补 Top-30/Top-50 中 COCO 缺失的室内类别 |
| 简单增强 | 避免过度增强，优先水平翻转、随机裁剪、尺度/焦距一致性增强 |
| EMA / 稳定训练 | 每轮 head 改动后使用 EMA 或 best checkpoint 固化稳定点 |
| 全层辅助损失思想 | 若引入 verifier / decoder，应在中间输出也施加质量监督 |
| query / proposal 数量调节 | 在 FCOS proposal 阶段做 Top-K 工作点搜索，形成精度-延迟曲线 |
| 检测 + 分割共享低分辨率特征 | 后续 mask head 可复用检测 proposal 与 ROI 特征 |

### 2.2 暂不照搬

| RF-DETR 机制 | 暂缓原因 |
|---|---|
| 纯 DETR decoder | 已有 DETR 头失败记录，需避免重复踩坑 |
| 匈牙利匹配主训练范式 | 当前密集监督更稳定，召回容量已足够 |
| 大规模 NAS | 当前瓶颈是 FP_bg，不是速度-精度 Pareto 搜索 |
| patch size / 多分辨率搜索 | 项目训练计划固定 896×504，不做多分辨率 |

### 2.3 可作为后备的 DETR 路线

如果未来重新尝试 DETR，必须满足以下前提：

1. 使用 RT-DETR / Deformable DETR / RF-DETR 式 two-stage query selection，而不是旧 DETR 头。
2. 加 denoising query 稳定匹配。
3. 先在 Objects365 / COCO 上预训练，再做项目 Top-30/Top-50 微调。
4. 保留 FCOS 作为 proposal 或教师，避免稀疏 query 从零学定位。

---

## 3. 主路线：FCOS++ 高召回 proposal + 强前景验证

### 3.1 为什么不立刻换掉 FCOS

FCOS 已经完成了三个关键证明：

1. 同冻结特征下，train recall 可达 97%，推翻"冻结特征完全不支持检测"的假设。
2. 扩 COCO 后 train≈val，说明数据扩展有效。
3. D2a 后 `conf0.05` 容量达到约 89%，说明候选框召回能力已经够高。

因此下一步应把 FCOS 从"最终检测器"升级为：

```text
高召回 proposal generator + 可校准质量预测器
```

再叠加前景验证，解决 FP_bg。

### 3.2 架构草图

```text
RGB
  -> DINOv2 backbone (frozen)
  -> shared / private adapter features
  -> FCOS++ dense proposal head
       - cls quality branch
       - box DFL branch
       - objectness / foreground branch
       - optional IoU quality branch
  -> Top-K proposals
  -> ROI verifier
       - ROIAlign / token crop pooling
       - category verification
       - objectness verification
       - IoU quality regression
  -> calibration
  -> final boxes
```

最终分数建议：

```text
score = calibrated(cls_quality) * objectness * quality
```

其中：

- `cls_quality`：类别 + 框质量的 QFL 分支。
- `objectness`：是否为真实前景物体，主攻 FP_bg。
- `quality`：预测框与 GT 的 IoU 质量，主攻 FP_loc 与高置信排序。

---

## 4. 候选方法与优先级

### 4.1 E1：加回 objectness / foreground 分支

> ❌ **已实测作废（2026-06-11，见 §0.5）**：加 objectness 后容量从 0.92 塌到 0.59、@0.5≈0。objectness 乘法门控在冻结 dense head 上伤容量。

**目标**：解决 QFL 去 centerness 后前景门控被撤，导致 FP_bg 升高的问题。

做法：

- 在 FCOS head 上新增独立 objectness branch。
- 正样本位置 objectness=1，明确背景位置 objectness=0。
- 推理分数从 `cls_quality` 改为 `cls_quality * objectness`。
- 每次训练后重新 Platt 标定。

预期：

- FP_bg 占比下降。
- raw 分数可能下降，但标定后 `recall@conf0.5` 应上升。

判据：

| 指标 | 期望 |
|---|---|
| FP_bg 占比 | 明显下降 |
| precision@thr0.2 | 上升 |
| 标定后 recall@conf0.5 | 高于 31.8% |

优先级：最高。

### 4.2 E2：hard negative mining / 背景难例加权

> ❌ **已实测作废（2026-06-11，见 §0.5）**：在 objectness 上做在线 OHEM（ratio 3/10），容量塌到 ~0.75。根因：L≈49000≫n_pos，"丢弃易负例"丢掉 ~99% 负例 → objectness 全图校准崩坏。若日后重试，须改"保留全量 focal + 仅对最硬负例叠加加权"，且**不建立在 objectness 分支上**（该分支本身已判定有害）。

**目标**：专门惩罚最像物体的背景框。

做法：

- 用当前 best ckpt 在训练集或 held-out 上收集高分 FP_bg。
- 将这些位置或 proposal 标成 hard negatives。
- 对 hard negative 的 objectness / cls loss 加权。
- 可采用在线 OHEM 或离线 hard negative replay。

预期：

- 高分背景框减少。
- `conf0.5` 门槛下保留更多真阳，滤掉更多背景。

注意：

- 权重不宜过大，否则会把低频真实物体一起压低。
- 必须按类别统计，避免长尾类别被 hard negative 误伤。

优先级：最高，与 E1 可先后消融。

### 4.3 E3：ATSS / TAL 动态正负样本分配

**目标**：减少固定 center sampling 带来的正负样本噪声。

候选：

- ATSS：按每个 GT 的候选位置统计 IoU，自适应选择正样本。
- TAL / TOOD：分类分数与定位质量共同决定正样本。

预期：

- 分类与定位更对齐。
- 减少"类别高分但框不准"和"框准但类别低分"。
- 对 FP_loc 和高置信排序均有帮助。

优先级：高。

### 4.4 E4：FCOS proposal + ROI verifier

**目标**：二阶段验证真实前景，直接打 FP_bg。

做法：

1. FCOS 用低阈值或 Top-K 输出 proposals。
2. 对每个 proposal 从 adapter feature 中做 ROIAlign 或 token crop pooling。
3. ROI verifier 输出：
   - category logits
   - objectness
   - IoU quality
   - optional bbox refinement
4. 用 verifier 分数重排并校准。

训练标签：

- IoU >= 0.5 且类别匹配：positive。
- IoU < 0.1：background negative。
- 0.1 <= IoU < 0.5：localization hard case，可单独监督 IoU quality 或 bbox refinement。

优点：

- 与当前 FCOS 兼容，不推翻已有工作。
- 直接利用 FCOS 的高召回 proposal。
- ROI verifier 任务比全图密集分类更简单，专注判断"这个框是不是真的物体"。

风险：

- 训练和推理链路更复杂。
- 需要控制 proposal 数量，否则延迟增加。

建议：

- 首轮只做轻量 verifier，不做复杂 cascade。
- Top-K 从 50 / 100 / 200 / 300 搜索。
- 若 verifier 明显降低 FP_bg，可将其定为生产路径。

优先级：很高。若 E1-E3 后仍无法接近 50% 以上，必须上 E4。

### 4.5 E5：Objects365 数据接入

**目标**：补齐 COCO 缺失的室内长尾类别，降低类别覆盖不足导致的背景/混淆误报。

优先下载顺序：

1. annotations。
2. validation images。
3. 部分 train shards。
4. 全量 train shards。
5. 暂不下载 test。

接入步骤：

1. 解析 Objects365 categories。
2. 建立 `Objects365 category -> DINO-GeoSemMap Top-30/Top-50` 映射。
3. 生成只包含目标类别的 JSONL manifest。
4. 与 COCO 组成混合训练。
5. 对 O365-only 类别单独统计召回。

重点类别：

- `coffee_table`
- `nightstand`
- `wardrobe`
- `cabinet`
- `bookshelf`
- `door`
- `window`
- `washing_machine`
- `lamp`
- `curtain`

优先级：高，但应在 E1/E2 至少有一轮结果后接入，避免结构问题和数据问题混在一起。

### 4.6 E6：D2b LoRA 松动 backbone 末层

> ⏸️ **暂缓（2026-06-11，见 §0.5）**：触发 D2b 的前提是"FP_bg 源于前景/背景特征混叠"。诊断显示 E3 cls_quality 对 TP-vs-FP_bg 的 AUC=0.88（非混叠）→ 前提不成立。先攻 FP_loc/定位质量，backbone 暂不动；若定位与标定都榨干仍卡 conf0.5，再回到 D2b。

**目标**：验证更深层的特征松动是否能继续提升高置信精度。

做法：

- DINOv2 backbone 主体仍冻结。
- 只在末 N 层 attention / MLP 加 LoRA。
- 与 FCOS++ / verifier 联训。
- 保持检测专属分支，不影响深度、房间、分割头。

判据：

- 若 E1-E4 主要解决 FP_bg 后仍卡住，再上 D2b。
- 若 FP_bg 诊断显示背景与前景在特征空间本就混在一起，提前上 D2b。

优先级：中。D2a 已证明松动有用，但增益温和，不能作为第一解法。

---

## 5. 其他检测头方案评估

### 5.1 可借鉴但不优先替换

| 方法 | 借鉴点 | 结论 |
|---|---|---|
| GFL / GFLv2 | QFL + DFL + 质量估计 | 已部分采用，继续补质量预测 |
| YOLOX | decoupled head + objectness + SimOTA | 可借 objectness 和动态匹配，不必整体换 YOLO |
| TOOD | task-aligned head / assigner | 借 TAL 处理分类定位错位 |
| ATSS | 自适应正样本选择 | 可替换固定 center sampling |
| D-FINE | 分布式框精修、实时检测经验 | 可借框质量 refinement，不作为首选框架 |
| RT-DETR | two-stage query selection、实时 DETR | 作为后备 DETR 路线，不替换当前 FCOS 主线 |
| Deformable DETR | 稀疏多尺度 cross-attention | 只在重新尝试 DETR 时考虑 |
| Detic | 大词表 / LVIS federated loss | 接 LVIS/O365 时必须借鉴 |

### 5.2 当前不建议

| 方案 | 原因 |
|---|---|
| 重新启用旧 DETR 头 | 已验证坍塌且召回低 |
| 直接整体换 YOLO | 工程代价大，且破坏当前冻结特征/多头合并简洁性 |
| 全量解冻 DINOv2 | 违背共享 backbone 设计，可能牵动其他 head |
| 直接堆更大 adapter | D2a 增益温和，当前主因更像 FP_bg 与分数质量 |

---

## 6. 实验顺序

### 6.1 推荐顺序

> ⚠️ 实测后顺序已调整（2026-06-11，见 §0.5）：E1/E2 作废，E3 为生产头，下一步插入 **E-loc（攻定位质量）**。

| 顺序 | 实验 | 目的 | 实测状态 |
|---|---|---|---|
| E1 | D2a + DFL + objectness | 恢复前景门控，打 FP_bg | ❌ 作废（objectness 伤容量） |
| E2 | hard negative mining | 压高分背景幻觉 | ❌ 作废（OHEM 护栏触发） |
| E3 | ATSS / TAL | 改善正负样本分配与分类定位对齐 | ✅ 生产头，标定后 45.4%@conf0.5 |
| **E-loc**（新，下一步）| 强化 DFL / 框质量 | 攻 FP_loc（42%）+ 抬 cls_quality 过阈值 | 📋 待 brainstorm |
| E4 | FCOS proposal + ROI verifier | 二阶段验证，强打 FP_bg | 后备 |
| E5 | Objects365 接入 | 补室内长尾类别和真实世界多样性 | 后备 |
| E6 | D2b LoRA | 特征级松动兜底 | ⏸️ 暂缓（cls AUC 0.88，非混叠） |

### 6.2 每轮必须输出

每次实验都需要产出同一组表，避免指标口径漂移：

| 项 | 必填 |
|---|---|
| raw `recall@conf0.5` | 是 |
| Platt 标定参数 | 是 |
| 标定后 `recall@conf0.5` | 是 |
| `recall@conf0.05` | 是，能力上限 |
| Top-K proposal recall | 是 |
| FP 成分诊断 | dup / cls / loc / bg |
| TP 分数分布 | p50 / p90 / frac>0.5 |
| 按类别 recall | Top-30 / O365-only 类别单独列出 |

---

## 7. 预期里程碑

> ⚠️ 实测后修正（2026-06-11，见 §0.5）：里程碑达成路径与原假设不同——**40% 由 TAL（非 objectness）达成**。

| 阶段 | 原目标 | 实测 |
|---|---|---|
| M1 | objectness 后 FP_bg 明显下降，标定后 `recall@conf0.5` 超过 40% | ✅ **45.4%**，但由 **E3=TAL（无 objectness）** 达成；objectness 反而有害 |
| M2 | hard negative + ATSS/TAL 后，接近或超过 50% | ⏳ 改由 **E-loc（攻定位质量）** 冲 50%，非 hard-neg |
| M3 | ROI verifier 后，达到 60% 左右 | 后备 |
| M4 | Objects365 + in-domain 数据后，向 70% 冲刺 | 后备 |
| M5 | 若仍不足，用 D2b LoRA 补特征精度 | ⏸️ 暂缓（cls AUC 0.88 表明特征非瓶颈） |

这些里程碑不是硬承诺，而是判断路线是否有效的观察点。若某阶段没有带来预期增益，应回到 FP 归因表重新定位，而不是继续叠复杂度。

---

## 8. 最终推荐

当前最推荐的实现路径是：

```text
FCOS(D2a+DFL)
  -> add objectness
  -> hard negative mining
  -> ATSS / TAL assignment
  -> FCOS proposal + ROI verifier
  -> Objects365 Top-30/Top-50 pretrain / finetune
  -> optional D2b LoRA
```

原因：

1. 它不推翻当前已验证有效的 FCOS 生产头。
2. 它直接攻击当前最大瓶颈 FP_bg，而不是重复解决召回容量。
3. 它保留冻结 DINOv2 backbone 与多头无损合并的总体设计。
4. 它能自然承接 RF-DETR 的 Objects365 训练经验，但避免回到已失败的旧 DETR 路线。

结论：

> 在冻结 DINOv2 backbone 的约束下，`recall@conf0.5 >= 70%` 更可能来自 **高召回密集头 + 背景抑制 + 二阶段验证 + Objects365 数据覆盖** 的组合，而不是单一换头。下一阶段应先把 FCOS 升级为 FCOS++，再按需加入 ROI verifier。

