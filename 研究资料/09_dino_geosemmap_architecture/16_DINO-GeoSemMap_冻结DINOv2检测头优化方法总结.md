# DINO-GeoSemMap 冻结 DINOv2 检测头优化方法总结

> 版本：v1.0
> 日期：2026-06-11 ~ 2026-06-12
> 关联：[13 检测头优化方案](13_DINO-GeoSemMap_检测头优化方案.md) · [14 检测头优化实施记录](14_DINO-GeoSemMap_检测头优化实施记录.md) · [15 下一阶段优化方案](15_DINO-GeoSemMap_冻结DINOv2检测头下一阶段优化方案.md)（本文是其 E 系列的执行总结）
> 分支：`feat/det-d0-d1-diagnosis`

---

## 目录

- [0. 摘要](#0-摘要)
- [1. 目标与验收口径](#1-目标与验收口径)
- [2. 方法总览](#2-方法总览)
- [3. 各方法详解](#3-各方法详解)
- [4. 诊断工具与关键发现](#4-诊断工具与关键发现)
- [5. 统一结论：三堵墙](#5-统一结论三堵墙)
- [6. 当前状态与产物](#6-当前状态与产物)
- [7. 边界与下一阶段](#7-边界与下一阶段)
- [8. 复现命令](#8-复现命令)
- [9. 推理性能（单帧耗时）](#9-推理性能单帧耗时)

---

## 0. 摘要

本轮在 **DINOv2 backbone 全程冻结** 的前提下，穷尽了**损失级 / 分配级 / 标定级**的检测头优化手段（E 系列）。

- **生产头 = E3**（QFL + DFL + **TAL 动态分配**，**无 objectness**，配 D2a 私有 adapter）：标定后 **recall@conf0.5 = 0.454**，召回容量 **0.913@conf0.05**。
- **其余三个干预均无实质增益**：E1（objectness）伤容量作废；E2（objectness + OHEM）伤容量作废；E-loc（CIoU + DFL 加权）平局（recall +0.6pp 噪声级，FP_loc 未降）。
- **诊断把瓶颈钉死在冻结 patch-14 特征本身**：FP_bg、FP_loc、小物体召回三个问题都对损失级杠杆免疫。
- **结论：冻结 backbone 下的损失级优化已到顶。** 进一步提质必须动**特征层**（D2b LoRA 松动末层 / 提高输入分辨率），留作下一阶段。

---

## 1. 目标与验收口径

- **北极星**：常见类 `recall@(IoU≥0.5, conf≥0.5) ≥ 70%`（研究级目标，不保证可达）。
- **唯一硬验收口径**：**Platt 标定后 `recall@conf0.5`**。
- **诊断口径**（不作验收）：`conf0.05`（召回容量上限）、`top-K proposal recall`、raw `conf0.5`（训练刻度）。
- **数据**：11 类子集 `det11`（sofa/chair/dining_table/bed/tv/refrigerator/microwave/oven_stove/toilet/sink/potted_plant）；train = COCO train2017 映射 35326 图（`det_cocotrain_L1`），val = COCO val2017 1504 图（`det_coco_L1`）。
- **统一配置**：冻结 DINOv2 ViT-B/14 + D2a 检测私有 adapter（9.2M，warm-start）+ FCOS 密集头；40k steps，batch 8，lr 1e-4，best.pt 抓峰。

---

## 2. 方法总览

| 实验       | 关键变量（均：冻结 backbone + D2a adapter + det11）              | @.05 容量   | raw @0.5 峰 | 标定后 @0.5  | 判定      |
| -------- | ------------------------------------------------------ | --------- | ---------- | --------- | ------- |
| **E3** ✅ | QFL + DFL + **TAL 分配**，无 objectness                    | **0.913** | 0.402      | **0.454** | **生产头** |
| E1       | + objectness 分支（score = cls×obj）+ center 分配            | 0.586     | 0.006      | —         | ❌ 伤容量   |
| E2       | + objectness + **OHEM**（ratio 3 / 10）                  | ~0.75     | ~0.08      | —         | ❌ 伤容量   |
| E-loc    | TAL + **GIoU→CIoU + lambda_dfl 0.25→1.0**，无 objectness | 0.913     | 0.390      | 0.460     | ⚖️ 平局   |

> 关键认知更正（见 15 号 §0.5）：**E3 实际不含 objectness 分支**（ckpt `use_objectness=False`）。早期"E3 含 objectness"的记载有误，导致 E1/E2 误把工夫下在一个有害的分支上。

---

## 3. 各方法详解

### 3.0 为什么检测头选 FCOS（而非 DETR）

检测头首轮用的是 **DETR 风格头**（稀疏 query + 匈牙利二分匹配），结果卡死：val recall ~12.8%、`conf0.5` 处几乎无输出，且 **过训坍塌**（step5000 见顶后退化，step20000 趋近 0）。当时一度归因为"冻结特征不行"。

**关键对照实验 D1'（见 14 号 §2.3）** 排除了这个误判：**同一冻结 base + 同一 1504 穷尽集，唯一变量 = 检测头范式**（密集 anchor-free vs DETR 稀疏 query）：

| 检测头        | train recall 峰值                      | 坍塌点（step6k）后 |
| ---------- | ------------------------------------ | ------------ |
| DETR 头     | ~20%（9.8→…→19.9% 后**坍塌到 3.5%→1.1%**） | 崩            |
| **FCOS 头** | **97.3%**（54→76→91→96→97）            | **全程稳定**     |

→ **推翻"冻结特征是召回容量瓶颈"**：冻结 DINOv2 + adapter 的特征足以支撑 97% train recall，"找得到"物体；那个 ~20% 是 DETR 头造成的，不是特征上限。

**DETR 头在"冻结特征 + 小数据"下的三宗罪**：
1. **数据饥饿**：稀疏 query 要从零学"物体在哪"，需要大数据；本项目数据量小，学不动。
2. **匈牙利匹配不稳**：二分匹配的目标分配在训练中跳变，优化不稳定。
3. **过训坍塌**：set loss 仍降但检测退化（recall 26%→6%），DETR 专属病；FCOS 在同一坍塌点免疫。

**FCOS（密集 anchor-free）为什么合适**：
1. **密集监督**：金字塔上**每个格点都是训练信号**（约 49000 个）→ 监督极密、**不数据饥饿**，小数据也学得动。
2. **无二分匹配**：正负样本由几何/对齐规则直接确定，**训练稳定、不坍塌**。
3. **置信度可校准 + 定位好**：QFL 让分数 = 框的 IoU 质量（配 Platt 标定解决欠自信），DFL 做分布式框回归。
4. **与冻结特征配合好、最简、合并友好**：D2a 后召回容量达 89-91%@conf0.05。

**决策：FCOS 定为生产检测头**，后续 E 系列全部在其上做。

### 3.1 E3（生产头）：QFL + DFL + TAL

- **做法**：QFL（cls 分数目标 = pred 框与 GT 的 IoU，去 centerness）+ DFL（框边分布回归）+ **TAL/TOOD 动态正样本分配**（align = score^α·IoU^β，每 GT 取 top-k）。**不使用 objectness**，分数 = cls_quality。
- **结果**：raw @0.5 峰 0.402（step27k）→ Platt 标定 `sigmoid(2.425·logit+0.376)` → **0.454**；容量 0.913@conf0.05。标定曲线 conf0.3/0.5/0.7 = 0.564/0.454/0.338。
- **为什么赢**：TAL 给了"分类-定位对齐"的干净正样本，QFL 让 TP 分数随 IoU 抬升，标定（a=2.43>1）把欠自信 TP 拉过 0.5。**E1→E3 容量从 0.59 跳到 0.92，唯一变量就是 center→TAL 分配** —— 增益来自"把正样本做对"。

### 3.2 E1：objectness 前景门控（作废）

- **做法**：加 class-agnostic objectness 分支，分数改 `cls_quality × objectness`，正样本 obj=1、背景 obj=0（sigmoid focal over all）。
- **结果**：容量塌到 **0.586**、raw @0.5 ≈ 0.006。
- **为什么败**：乘法门控的容量完全取决于 objectness 的全图校准；冻结特征下 objectness 一旦不准，就把 TP 分数一起乘小，容量崩。

### 3.3 E2：hard-negative mining / OHEM（作废）

- **做法**：在 objectness 分支上做在线 OHEM —— 每图保留全部正样本 + top-K 最硬背景（K = ratio·n_pos），丢弃易负例。试了 ratio 3 与 10。
- **结果**：容量塌到 ~0.75（ratio 3/10 几乎一致），raw @0.5 ≈ 0.08。护栏触发。
- **为什么败**：dense head 位置数 **L ≈ 49000 ≫ n_pos ≈ 数十**，即便 ratio 10 也只保留 <1% 负例、**丢弃 99% 易负例** → objectness 全图校准崩坏 → 容量塌。"丢弃式 OHEM" 在 dense head 上结构性不可行；调 ratio 救不了。

### 3.4 E-loc：CIoU + DFL 加权（平局）

- **做法**：框回归损失 GIoU→**CIoU**（加中心距 + 长宽比惩罚）+ `lambda_dfl 0.25→1.0`。目标：提升定位精度 → 抬高低 IoU 正样本（FP_loc / 小物体 / 拥挤）的 cls_quality（=IoU 目标）→ 过 conf0.5。
- **结果**：raw @0.5 峰 0.390（step37k）→ 标定 `sigmoid(2.411·logit+0.581)` → **0.460**（vs E3 0.454，**+0.6pp，噪声级**）；容量 0.913（护栏守住）。

| 细分对比（thr0.2 / 标定后） | E3 | E-loc |
|---|---|---|
| FP_loc 占比 | 42.4% | **42.9%（未降）** |
| FP_bg 占比 | 49.0% | 48.4% |
| precision@thr0.2 | 0.298 | **0.344** |
| 总预测数 | 10527 | 8915 |
| 小物体 dep@.5 | 0.115 | **0.102（略降）** |
| 大物体 dep@.5 | 0.687 | **0.713（+2.6pp）** |
| 拥挤 dep@.5 | 0.306 | 0.300 |

- **为什么平**：CIoU **没有**把"定位差一点"的 FP_loc 框转成 TP（FP_loc 占比未降）。它实际做的是：**预测更干净**（少 1600 框、precision +4.6pp）、**大物体定位更好**（长宽比惩罚对大件家具有用），但**小物体略退**，两者抵消。可视化印证：清晰大物体框更紧，杂乱小物体该漏还漏、杂物误检仍在。
- **含义**：更好的回归损失都改善不了 FP_loc → **定位不准本身也是冻结 patch-14 特征的粒度限制，不是损失能解的**。

---

## 4. 诊断工具与关键发现

本轮新建了一套"先诊断再下药"的探针（避免盲调），均为纯 eval、几分钟，复用 calibrate 的 load/match 逻辑：

| 脚本 | 作用 | E3 关键发现 |
|---|---|---|
| `scripts/calibrate_det.py` | Platt 标定（score_cal = sigmoid(a·logit+b)）| a=2.43>1，欠自信 TP 被抬升 |
| `scripts/diag_det_fp.py` | FP 成分 dup/cls/loc/bg | FP_bg 49% + FP_loc 42% 为两大来源 |
| `scripts/diag_det_obj_sep.py` ⭐新 | cls/obj 对 TP-vs-FP_bg 的可分性 AUC | **cls_quality AUC = 0.883**（TP 0.526 vs FP_bg 0.283）→ **特征已可分，非混叠** |
| `scripts/diag_det_loc.py` ⭐新 | FP_loc 成因（H1 reg_max 撞顶 / H2 TAL 跨层 / H3 纯回归不准）| reg_max 撞顶 1.7%、跨层 0.2% → **H3 纯回归不准** |
| `scripts/diag_det_size.py` ⭐新 | 按尺寸 / 拥挤分层的 cap@.05 vs dep@.5 | 小物体/拥挤**主要是置信问题**（容量 0.74-0.86 在，部署塌到 0.11-0.31）；小物体有 0.737 容量墙 |
| `scripts/infer_fcos_viz.py` ⭐新 | FCOS 框可视化（标定后 conf 门控）| 清晰大物体准，小/杂乱漏检 + 杂物误检 |

**两个最关键的诊断结论**：
1. **`diag_det_obj_sep`：cls AUC 0.88 推翻了"特征混叠 → 必须 D2b"的早期假设** —— 冻结特征对 TP-vs-FP_bg 已足够可分，所以 D2b 一度被判"无依据、暂缓"。
2. **`diag_det_loc`：FP_loc = H3 纯回归不准** —— 但 E-loc 证明"更好的回归损失也救不了它"，所以**定位精度的瓶颈仍回到特征粒度**。

---

## 5. 统一结论：三堵墙

把整个 E 系列串起来，所有失败/平局都指向**同一个根因 —— 冻结 DINOv2 patch-14 特征的粒度/容量限制**：

1. **objectness 乘法门控有害**（E1/E2）：任何"更狠地压背景"的损失干预，都会破坏 objectness 在全图的校准 → 把 TP 分数一起乘小 → 容量塌。E3 赢恰恰是**不用** objectness。
2. **FP_bg 不可损失级解决**：cls AUC 0.88 说明特征"能排序"，但你想用损失把背景压得更低，就连带压低前景（特征相近）。
3. **FP_loc 不可损失级解决**（E-loc）：H3 纯回归不准，但 CIoU/DFL 加权都没改善 → 框的精度被 patch-14 的粗空间特征封顶。
4. **小物体容量墙**：cap@.05 = 0.737 —— 约 26% 的小物体在 patch-14 下根本提不出候选，损失够不到。

→ **每一分真增益都来自"把正样本/分配做对"（TAL）+ 标定；每一次"和特征较劲"的损失干预都撞墙。损失级杠杆到顶。**

---

## 6. 当前状态与产物

- **生产头**：**E3**，`checkpoints/fcos_e3/best.pt`（含 calib），标定后 **recall@conf0.5 = 0.454**，距 70% 目标尚远，瓶颈 = 高置信精度（受冻结特征限制）。
- **E-loc** 无实质增益、不取代 E3；`checkpoints/fcos_eloc/best.pt` 留作记录。失败记录：`checkpoints/fcos_e2_r3` / `fcos_e2_r10`、`logs/fcos_e2_r3.log`。
- **代码**（分支 `feat/det-d0-d1-diagnosis`）：
  - `dino_gsm/heads/box_ops.py`：`complete_box_iou`（CIoU）
  - `dino_gsm/losses/fcos_loss.py`：`_ohem_mask`（OHEM，判定无效）+ box_loss giou/ciou 切换
  - `dino_gsm/config.py`：`ohem_enabled/ohem_neg_pos_ratio/ohem_min_neg`、`box_loss`
  - `scripts/train_fcos.py`：`--ohem/--ohem_ratio/--box_loss/--lambda_dfl`
  - 诊断脚本：`diag_det_obj_sep.py` / `diag_det_loc.py` / `diag_det_size.py` / `infer_fcos_viz.py`
  - 测试：`tests/test_fcos_ohem.py`、`tests/test_fcos_ciou.py`
- **设计/计划文档**：`docs/superpowers/specs/2026-06-11-det-e2-ohem-design.md`、`...-eloc-localization-design.md` 及对应 plans。

---

## 7. 边界与下一阶段

**本轮边界明确：不微调 backbone 的优化到此为止。** 损失级三连（E1/E2/E-loc）已证明冻结 patch-14 特征是共同上限。

**下一阶段候选（需动特征层，留待后续立项）**：
- **D2b：LoRA 松动 DINOv2 末 N 层**（attention/MLP）。注意：早期因 cls AUC 0.88 判"暂缓"，但那只针对 FP_bg 的**排序可分性**；**定位精度（FP_loc）与小物体候选（容量墙）是不同的轴，可能仍受特征粒度限制**，故 D2b 重新具备依据。保持检测专属 delta，不影响深度/房间/分割头的共享冻结基座。
- **提高输入分辨率 / 加更细特征层（P1）**：直攻小物体 0.737 容量墙。项目原定 896×504 固定，需评估代价。
- **E4（ROI verifier）/ E5（Objects365 数据）**：仍为后备，但在"特征是共同瓶颈"的认知下优先级低于 D2b。

> 决策点：是否突破"四头共享冻结基座"。已倾向：语义头（检测/分割）走"共享冻结权重 + 各自轻量 LoRA delta"，几何头（深度/房间）继续干净冻结。

---

## 8. 复现命令

```bash
# E3 生产头（复刻）：QFL+DFL+TAL，无 objectness
python scripts/train_fcos.py --base_ckpt checkpoints/depth_v1/step_0098000.pt \
  --train_manifests /mnt2/dgsm_data/manifests/det_cocotrain_L1.jsonl \
  --val_manifests /mnt2/dgsm_data/manifests/det_coco_L1.jsonl \
  --class_subset det11 --assigner tal --train_adapter \
  --steps 40000 --batch_size 8 --eval_every 1000 --out checkpoints/fcos_e3

# E-loc（CIoU + DFL 加权）：在 E3 基础上加两旗标
#   ... 同上 ... --box_loss ciou --lambda_dfl 1.0 --out checkpoints/fcos_eloc

# 标定（写回 ckpt 的 calib 字段）
python scripts/calibrate_det.py --ckpt checkpoints/fcos_e3/best.pt \
  --base_ckpt checkpoints/depth_v1/step_0098000.pt \
  --val_manifests /mnt2/dgsm_data/manifests/det_coco_L1.jsonl

# 四诊断（FP 成分 / 可分性 / 定位成因 / 尺寸·拥挤分层）+ 可视化
python scripts/diag_det_fp.py      --ckpt <ckpt> --base_ckpt <base> --val_manifests <val> --thr 0.2
python scripts/diag_det_obj_sep.py --ckpt <ckpt> --base_ckpt <base> --val_manifests <val> --thr 0.2
python scripts/diag_det_loc.py     --ckpt <ckpt> --base_ckpt <base> --val_manifests <val> --thr 0.2
python scripts/diag_det_size.py    --ckpt <ckpt> --base_ckpt <base> --val_manifests <val>
python scripts/infer_fcos_viz.py   --ckpt <ckpt> --num 12 --conf 0.5
```

---

## 9. 推理性能（单帧耗时）

实测条件：GPU = **NVIDIA RTX PRO 6000 Blackwell**，输入 **1×3×504×896**，batch=1，预热 10 次取 40-50 次中位数。fp16 = `torch.autocast`（权重 fp32、计算自动转半精度，数值安全、精度近无损）。后处理（解码 + NMS）约 1ms，已含在检测完整链路内。

### 9.1 单头（检测 E3）

| 精度 | 完整链路（backbone+头+解码+NMS） | FPS |
|---|---|---|
| fp32 | 26.8 ms | 37 |
| **fp16** | **~15 ms** | **~63-69** |

### 9.2 四头共享 backbone（深度 + 房间 + 检测 + 分割）

四头共享同一个冻结 DINOv2 ViT，backbone 只跑 **1 次**，各头只加各自的 adapter+head 增量。

| | fp32 | fp16 |
|---|---|---|
| backbone（ViT，共享 1 次） | 18.1 ms | 7.5 ms |
| + 深度头 | +3.0 | +2.6 |
| + 房间头 | +0.9 | +0.8 |
| + 检测头（FCOS 密集头，最重） | +7.6 | +7.1 |
| + 分割头（RoIAlign，按 15 框计）| +1.0 | +0.9 |
| **四头共享总计** | **30.6 ms（33 FPS）** | **18.8 ms（53 FPS）** |
| 对比：四头各自独立（backbone 跑 4 次） | 84.9 ms | 41.2 ms |

### 9.3 结论

- **"共享冻结 backbone"架构的价值在推理端兑现**：backbone 是最重的部分，四头共享只跑 1 次 → 30.6ms vs 各自独立 84.9ms，**省近 2/3**。
- **半精度把耗时砍掉近一半**（四头 30.6→18.8ms，33→53 FPS），实时无压力；建议默认 autocast fp16。
- **加一个语义头的边际成本很低**（房间/分割 ~1ms、深度 ~3ms），检测头最重（~7.6ms，FCOS 密集头 + 4 层金字塔 adapter）。瓶颈始终是共享的 ViT-B/14 backbone。
- 以上为顶配卡数值；嵌入式部署（Orin 等）需 **TensorRT 导出 + fp16/int8**，并可考虑更小的 DINOv2 变体。

### 9.4 显存占用（峰值 VRAM）

同条件（1×3×504×896，batch=1）。fp16 = autocast。

**权重（参数）**：DINOv2 backbone（四头共享）**330 MB**；检测 adapter 15 MB + 检测 head 20 MB。

| 场景 | fp32 峰值 | fp16(autocast) 峰值 |
|---|---|---|
| 单头（检测 E3） | **599 MB** | 553 MB |
| 四头（当前代码，4 份 backbone 副本） | **1835 MB** | 1769 MB |
| 四头（共享 backbone 部署，估算） | **~0.85 GB** | — |

- 单头 ~0.6 GB、四头 ~1.8 GB，显存压力极小（普通 8GB 卡轻松）。
- 四头当前 1835MB 含 **4 份冗余 backbone 副本**（浪费 ~990MB）。共享部署只需 1 份 → 四头总权重 1422→**431 MB**，峰值降到 **~0.85GB** —— "共享冻结基座"的显存红利。
- ⚠️ **autocast fp16 几乎不省显存**（599→553）：权重仍 fp32、只有激活转半精度。要省显存须用**真 fp16/int8 权重**（`.half()` / 量化），可再砍 backbone 权重一半。**速度与显存是两件事**：autocast 提速近 2× 但基本不降显存。
