# DINO-GeoSemMap 检测数据扩展与长尾平衡（det24）

> 版本：v1.0
> 日期：2026-06-15
> 关联：[16 冻结DINOv2检测头优化方法总结](16_DINO-GeoSemMap_冻结DINOv2检测头优化方法总结.md)（E 系列；本文是其 §7 "E5 数据扩展" 一轴的执行）· [10 数据集构成](10_DINO-GeoSemMap_数据集构成.md)
> 状态：**配方确定 + 管线就绪 + 单元/smoke 验证通过；40k 正式训练尚未跑（无 recall 结果）**

---

## 0. 摘要

doc 16 的 E 系列把检测**锁在 det11（11 类）**上做损失级优化，结论是"损失级到顶、瓶颈=冻结 patch-14 特征"，并把 **Objects365 数据扩展列为后备 E5**。本轮执行这条**数据轴**：把检测从 det11 扩到 **det24**，训练集从 COCO 单源扩到 **LVIS train + Objects365 train（59,696 图 / ~324k 实例）**，并配套解决随之暴露的**类别长尾**（实例数跨 4 个数量级）。

- **类空间 det24**：OBJ30 中 LVIS 可映射的 25 类，再剔除 wardrobe（仅 12 实例，学不动）→ **24 类**。
- **长尾两层处理**：L1 = LVIS 式 repeat-factor 重采样（thr=0.01，按子集 remap 计频）；L2 = inverse-√freq 类别加权 QFL（**仅作用于正样本 cell**，频率加权归一化保总量级）。
- **干净 val**：从 LVIS **val** 均衡采样 `det_det24val_L2`（941 图，24/24 覆盖，与 train 不交叉）。
- **数据量已远超 L2 目标（det L2=40k 图）**，瓶颈不再是数据总量，而是长尾平衡 + （承 doc 16）冻结特征上限。

> ⚠️ **本文不含 recall 结果**：正式 40k 训练 + 标定尚未执行。期望管理见 §6。

---

## 1. 类空间：det11 → det24

| 子集 | 类数 | 来源 | 说明 |
|---|---|---|---|
| det11（E 系列用） | 11 | COCO | COCO 可映射的 11 个家居类 |
| **det24（本轮）** | **24** | LVIS + O365 | LVIS 覆盖 25 类，剔除 wardrobe |
| det25 | 25 | LVIS | 完整 LVIS 覆盖（保留作参考，不用于训练） |

- **OBJ30 各源覆盖**：COCO 11 / LVIS 25 / O365 23。LVIS∪O365 = 26（仅 tv_stand·shoe_cabinet·door·window 四类全无数据源）。
- **det24 = LVIS 的 25 类去 wardrobe(idx 7)**：wardrobe 在 lvis+o365train 仅 **12 实例 / 9 图**（O365 完全没有，LVIS 独有，数据源枯竭）→ 任何重采样/加权都只是反复看这 12 个框，过拟合，故从 head 剔除（不让其假装可检测）。
- O365 独有的 nightstand(idx 5) 落在 det24 之外，由 collate 的 remap（==-1）自动丢弃，保持单一干净标签空间。

定义：`dino_gsm/data/class_subset.py` 的 `DET24`（24 个 OBJ30 索引），经 `build_remap` 收缩到连续 `[0,24)`；越界标签在 `make_detection_collate(remap)` 处丢弃。

---

## 2. 数据组成

| 集 | manifest | 图 | 实例(det24) |
|---|---|---|---|
| **train** | `det_lvis_L2` + `det_o365train_L2` | **59,696** | **~324k**（丢弃 nightstand+wardrobe ~2.3k） |
| **val** | `det_det24val_L2`（新建） | **941** | ~3.5k，24/24 类覆盖 |

- **val 为何用 LVIS val 单源**：train 用的是 LVIS *train* + O365 *train*；LVIS *val* 复用 COCO val2017 图、与 train 天然不交叉，且**唯一能覆盖全 24 类**（O365 val 缺 bookshelf/curtain）。生成器 `dgsm-data/scripts/prep_det25_val.py`（稀有类优先贪心填充，cap 2000）。
- ⚠️ **footgun**：`det_det24val_L2` 名匹配 `det_*_L2` glob，训练务必用 `--train_manifests` 显式指定，**勿用 `--level L2`**（否则 val 混入 train）。

### 2.1 长尾全貌（train 集每类实例数）

| 头部 | | 尾部 | |
|---|---|---|---|
| chair | 110,060 | dining_table | 511 |
| lamp | 49,115 | washing_machine | 288 |
| cabinet | 41,738 | bookshelf | 113 |
| desk | 29,462 | (wardrobe 12 → 已剔除) | — |

→ 跨 **4 个数量级**。随机采样 + 等权 QFL 下，尾部被头部淹没。

---

## 3. 长尾两层处理

### L1 — repeat-factor 重采样（数据级）
- 复用 `dino_gsm/data/split.py:repeat_factor_weights`（LVIS 式：类 rf = max(1, √(thr/f_c))，图权重 = 所含类 rf 的 max）。
- 本轮**泛化**：支持裸 `UnifiedDataset`（显式 manifest 路径下 train_ds 非 Subset）+ 接 `remap`（records 存 OBJ30 标签，head 是 det24 连续空间，不 remap 则计频全错）。
- thr=0.01 精准只放大尾部：**bookshelf ×2.9 / washing_machine ×1.9 / dining_table ×1.4**，其余 ≈1.0。`train_fcos.py --balance --rfs_thr 0.01` 接 `WeightedRandomSampler`。

### L2 — inverse-√freq 类别加权 QFL（损失级）
- 新增 `inverse_sqrt_class_weights`：raw_c = 1/√(max(n_c, floor=50))，再按**实例频率加权归一化**（加权均值=1）→ 保持 cls 损失总量级、只重分配。验证：min(chair) 0.46 → max(bookshelf) 14.29，加权均值 **1.000**。
- **关键设计（正负号陷阱）**：QFL 张量 `[B,L,C]`，`cls_t>0` 即正样本 cell。**权重只乘到正样本 cell**——若整列 c 加权，会把稀有类通道在所有背景位置的负梯度也放大，反而压制该类（这正是 EQL 系列要解决的）。本方案只上调正样本梯度、负样本保持 1，正确符号、低风险。
- 实现：`FCOSCriterion` 加 `cls_weight` buffer + `set_class_weights()`；`forward` 在 `.sum()/n` 前对正样本 cell 乘权。`cls_balance="none"` 时路径与改动前**完全一致（回归安全）**。`FCOSConfig` 加 `cls_balance / cls_weight_floor`。

---

## 4. 代码改动

| 文件 | 改动 |
|---|---|
| `dino_gsm/data/class_subset.py` | `DET24`（去 wardrobe）入 SUBSETS |
| `dino_gsm/data/split.py` | `repeat_factor_weights` 泛化(裸 ds + remap)；新增 `inverse_sqrt_class_weights` |
| `dino_gsm/config.py` | FCOSConfig 加 `cls_balance` / `cls_weight_floor` |
| `dino_gsm/losses/fcos_loss.py` | `cls_weight` buffer + `set_class_weights` + QFL 正样本 cell 加权 |
| `scripts/train_fcos.py` | `--balance` / `--rfs_thr` / `--cls_balance`；采样器分叉；权重从 train 集计算并注入 |
| `dgsm-data/scripts/prep_det25_val.py` | DET24 val 切片生成器 → `det_det24val_L2.jsonl` |

**验证**：① 单元——权重/采样直方图如上数值，无 NaN；② 回归——`cls_balance=none` buffer 全 1、loss 不变；③ smoke——`det24: 24 classes`、采样器+权重+QFL+标签范围端到端 6 步跑通、eval 在 941 图上跑通。

---

## 5. 复现命令

```bash
# det24 长尾平衡训练（与 E3 一致：D2a 私有 adapter + depth_v1 warm-start）
MAN=/mnt2/dgsm_data/manifests
python scripts/train_fcos.py --base_ckpt checkpoints/depth_v1/step_0098000.pt \
  --train_manifests $MAN/det_lvis_L2.jsonl $MAN/det_o365train_L2.jsonl \
  --val_manifests   $MAN/det_det24val_L2.jsonl \
  --class_subset det24 --assigner tal --train_adapter \
  --balance --cls_balance invsqrt --rfs_thr 0.01 \
  --steps 40000 --batch_size 8 --eval_every 1000 --out checkpoints/fcos_det24

# 重建 val 切片（如需）
python /home/zktian3/dgsm-data/scripts/prep_det25_val.py --level L2 --cap 2000
```

> ⚠️ **与 E3 对齐**：E3 生产头用 D2a 私有 adapter（`--train_adapter --base_ckpt depth_v1`）。开发期 smoke 用的是 `--dinov2_weights` 直接冻结底座（更简），正式 det24 训练应如上用 adapter 路径，才能与 E3 公平对比。

---

## 6. 边界与期望管理

- **数据总量不再是瓶颈**（59.7k 图 ≫ L2 目标 40k）；本轮解决的是**长尾平衡**与**类空间扩展**。
- **承 doc 16 的硬约束**：E 系列已证冻结 patch-14 特征是损失级共同上限（FP_bg/FP_loc/小物体容量墙）。det24 是**数据轴**干预，与 **D2b（LoRA 松动特征层）** 正交——**数据扩展能否突破特征上限是开放问题**，不应预期单凭 det24 就达 70%@conf0.5。合理预期：尾部新类从"完全检不到"变为"可检测但偏弱"，常见类受益于更大更多样的训练集。
- **观察重点**：bookshelf(113)/washing_machine(288)/dining_table(511) 这些靠 L1+L2 救的中段类，看 val recall 是否上来；若仍上不去，下一步升级 **EQLv2**（下调稀有类负梯度，而非仅上调正梯度；需在 criterion 加 pos/neg 梯度累积 buffer）。
- door/window/tv_stand/shoe_cabinet 四类全数据源缺失，仍待 Habitat 渲染补（见 doc 16 §7）。

---

## 7. 一句话总结

把检测从 det11 扩到 **det24**、训练集从 COCO 扩到 **LVIS+O365（59.7k 图）**，并用 **RFS 重采样 + 正样本加权 QFL** 两层处理跨 4 数量级的长尾、剔除数据枯竭的 wardrobe；管线与 val 切片就绪、单元/smoke 通过，但 **40k 正式训练与 recall 验证待跑**，且需注意这是与 D2b 正交的**数据轴**——能否突破 doc 16 钉死的冻结特征上限尚待实测。
