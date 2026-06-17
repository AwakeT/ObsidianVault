# DINO-GeoSemMap 房间头与分割头 数据处理

> 版本：v1.2（2026-06-16 追加 §9.3 边界优化执行结果 + 过拟合探针，结论翻转为"泛化差距"）
> 日期：2026-06-15 ~ 06-16
> 关联：[17 检测数据扩展与长尾平衡 det24](17_DINO-GeoSemMap_检测数据扩展与长尾平衡_det24.md) · [10 数据集构成](10_DINO-GeoSemMap_数据集构成.md)
> 状态：**数据/管线就绪；room10/mask_det24 已训完并诊断（§8）；分割边界优化已做 §9.2 第1+2步(A 0.704→0.729) 并经探针定因=泛化差距(§9.3，非特征上限，D2b 不需要)，下一步 aug+边界损失；房间弱类留待后续**

---

## 0. 摘要

继检测头（det24）后处理另两个头。二者性质截然不同，处理力度也不同：

- **分割(mask)头是 class-agnostic**（单通道 BCE+Dice、per-RoI 监督，head 不读 labels）→ 检测那套类空间/per-class 加权**不适用**。只补两个缺口：干净可比的 val 切片 + `train_mask` 显式 manifest 入口。数据源仅 COCO+LVIS（O365 无 mask）。
- **房间(room)头是单标签 CE 分类** → 真长尾 + 两个 ScanNet 给不了的死类。做了三件事：**剔除 balcony/dining → ROOM10**、**数据扩充**（sceneType .txt 补全 + 全量 957 场景重 prep）、**CE 类权重 + 分层 scene-disjoint val**。

> ⚠️ 本文不含训练结果：mask/room 的 40k/20k 正式训练待跑。

---

## 1. 分割头（轻量）

### 关键认知：class-agnostic
`mask_head.py` 输出通道=1，`mask_loss.py` 单通道 BCE+Dice，`mask_targets.py` 按 GT box 裁出 28×28 二值掩码监督——**全程不读类别**。所以 mask 头**不需要** DET24 类空间统一、不需要 per-class 加权；长尾在 class-agnostic 下不致命（不丢类、只让边界偏向高频形状）。当前 `mask_lvis` 框内 IoU **0.71**，瓶颈是"边缘松"（patch-14 粗特征 + 28×28 RoI 分辨率），非类别问题。

### 改动
- **新建 `dgsm-data/scripts/prep_mask_val.py`**：仿 `prep_det25_val.py` 的 rare-class-first 采样，但写 `masks_rle`（polygon→RLE）+ `heads=["det","mask"]`。源 LVIS **val**（与 mask train 的 LVIS train 不交叉）→ `mask_det24val_L2.jsonl`（**941 图 / 3673 实例 / 24 类 / 100% 带 mask**）。
- **`scripts/train_mask.py`** 加 `--train_manifests/--val_manifests`（对齐 train_fcos；无则保持现状 level glob + 随机 split，回归安全）。collate 仍用 `detection_collate`（无 remap，避开 `collate.py` 的 masks-不跟随-remap-过滤 bug）。
- 不动 head/loss/target。长尾重采样本轮不做（优先级低）。

### 训练命令
```bash
MAN=/mnt2/dgsm_data/manifests
python scripts/train_mask.py --base_ckpt checkpoints/depth_v1/step_0098000.pt \
  --train_manifests $MAN/mask_cocotrain_L1.jsonl $MAN/mask_lvis_L2.jsonl \
  --val_manifests   $MAN/mask_det24val_L2.jsonl \
  --steps 30000 --batch_size 4 --eval_every 1000 --out checkpoints/mask_det24
```
train = cocotrain(35326,穷尽) + lvis(19696,非穷尽) = **55022 图**。class-agnostic 下每 RoI 由自己 GT mask 监督，混源不打架。smoke 通过（val_iou 0.64 未训基线）。

---

## 2. 房间头：ROOM10

`ROOM12` 中 **balcony（ScanNet 0 帧）+ dining_room（全库 ~1 场景）** 学不动也测不准，从训练头剔除 → **ROOM10**：
`[living_room, bedroom, bathroom, kitchen, study, hallway, storage, entryway, laundry, closet]`。

- `taxonomy.py` 加 `ROOM10` / `ROOM10_IDX` / `room10_index_from_scannet_scenetype`（dining/balcony/类外 → -1 丢弃）。**保留 ROOM12 不动**（MP3D region 映射未来用，接 MP3D 时另建空间再调和）。
- `prep_scannet_room.py` 产出 ROOM10 连续索引；`RoomConfig.num_rooms 12→10`。

---

## 3. 房间头：数据扩充

复用机制：`prep_scannet_room` 从 `depth_scannet_{level}` manifest 继承帧 + 附 room 字段。故扩 room = 补 sceneType + 扩 depth prep（**同时惠及 depth 头**）。

| 步骤 | 前 | 后 |
|---|---|---|
| sceneType .txt | 469 | **1513**（train+val 全覆盖，0 失败） |
| depth_scannet_L1 | 28931 帧 / 450 场景 | **36897 帧 / 951 场景** |
| room_scannet_L1 | 21716 帧 / 335 场景 (ROOM12) | **24400 帧 / 630 场景 (ROOM10)** |

**踩的坑**：第一次重跑 depth prep 只覆盖 450 场景——本地 `scannetv2_train.txt` 是旧的 450 场景子集列表（非官方完整 train）。改用**全量盘上 957 场景列表** `scannet_ondisk_all.txt` + `--num 38280`（=957×40，整除避免提前 break 漏掉尾部稀有类场景）后才真正扩开。

**ROOM10 帧分布（630 场景）**：living 5361 / study 4885 / bed 4852 / bath 4757 / kitchen 2383（强类充足）；entryway 889 / hallway 756（中）；storage 199 / laundry 159 / closet 159（弱，4-5 场景）。
> 场景数 335→630（~1.9×）是主要收益；弱类 storage/laundry/closet 各仅 4-5 场景是 **ScanNet 本质上限**，扩 prep 救不了，靠 §4 兜底，根治需 MP3D。

---

## 4. 房间头：类别均衡（CE 权重 + 分层 val）

`train_room` 原为裸 CE + shuffle + 普通 scene-disjoint，零均衡。改：

- **`split.py:room_class_weights`**：读单值 `record["room"]`，inverse-√(帧数) + 频率加权归一化（均值≈1）。实测 living_room 0.78 → laundry/closet 4.97。
- **`split.py:split_train_val_by_scene_stratified`**：按 room 类分桶 scene，每桶留 ~val_frac 进 val（n≥2 时 ≥1）→ 稀有类两边都有。
- `room_loss` 加 `weight` 参数；`train_room` 默认用分层 split + CE 权重（`--no_balance_ce`/`--unstratified`/`--frame_split` 退回旧行为）。

**分层 split 实测修复弱类不可测**（val_frac=0.1）：

| 类 | 普通 scene-disjoint val 场景 | 分层 val 场景 |
|---|---|---|
| storage | **0（不可测）** | 1 |
| laundry | **0** | 1 |
| closet | **0** | 1 |

### 训练命令
```bash
python scripts/train_room.py --base_ckpt checkpoints/depth_v1/step_0098000.pt \
  --level L1 --steps 20000 --batch_size 16 --out checkpoints/room10
```
smoke 通过：分层 split train 21732 / val 2668（68 held-out 场景）、CE 权重 min0.78/max4.97、num_rooms=10、eval 跑通。

---

## 5. 代码与产物

| 文件 | 改动 |
|---|---|
| `dgsm-data/scripts/prep_mask_val.py` | 新建：LVIS-val mask 切片（带 masks_rle） |
| `scripts/train_mask.py` | 加 `--train_manifests/--val_manifests` |
| `dgsm-data/dgsm_data/taxonomy.py` | 加 ROOM10 + room10 映射 |
| `dgsm-data/scripts/prep_scannet_room.py` | 产出 ROOM10 |
| `dino_gsm/config.py` | RoomConfig num_rooms 12→10 |
| `dino_gsm/data/split.py` | room_class_weights + split_train_val_by_scene_stratified |
| `dino_gsm/losses/room_loss.py` | CE 加 weight 参数 |
| `scripts/train_room.py` | 分层 val + CE 权重接线 + 开关 |
| `scripts/download_scannet_meta.py` | 新建：补 sceneType .txt |

**新增/重建 manifest**：`mask_det24val_L2`（新）、`depth_scannet_L1`（重建 36897/951，旧版 `.bak`）、`room_scannet_L1`（重建 ROOM10 24400/630，旧 ROOM12 `.bak`）。

## 6. 风险 / 备注
- **depth_scannet_L1 被重 prep 覆盖**（450→951 场景）→ depth 头若重训用新数据（场景多样性↑，符合全量训练目标）；旧版已备份。
- 弱类 storage/laundry/closet（4-5 场景）、balcony/dining（剔除）是 ScanNet 本质缺 → 根治需 MP3D region（`taxonomy.py` 字母码/全名映射已就绪，但无 `prep_mp3d_room.py`，且 Habitat 无头渲染未解锁，见 doc 16 §7）。
- 全量 957 含同一物理房间的 _01/_02 重扫 → scene_key 视为不同场景，分层 split 下同房间可能跨 train/val（轻微泄漏）；本轮未去重，后续可按 base scene 号合并。
- mask 长尾重采样、按类 mIoU 评估：本轮不做（class-agnostic 优先级低）。

## 7. 一句话总结

分割头因 class-agnostic 只补了干净 val 切片（LVIS-val 941 图带 mask）+ 显式 manifest 入口；房间头剔除 balcony/dining 成 **ROOM10**、靠"补 sceneType .txt + 全量 957 场景重 prep"把场景从 335 扩到 **630**（depth 帧同步 28931→36897）、并用 **CE 类权重 + 分层 scene-disjoint val** 让 storage/laundry/closet 等弱类从"val 不可测"变为两边都有——数据与管线就绪，正式训练待跑；弱类与 balcony/dining 的根治仍待 MP3D。

---

## 8. 训练结果与诊断（2026-06-15 追加）

两头均已训完：`checkpoints/room10/best.pt`（step12000）、`checkpoints/mask_det24/best.pt`（step25000）。

### 8.1 分割头 mask_det24
- **框内 IoU = 0.709**（在新 LVIS-val mask 切片 `mask_det24val_L2` 上，与旧 mask_lvis 的 0.71 持平，但口径更干净）。
- 可视化（`checkpoints/mask_det24/viz/`，PRED|GT 对照）：掩码覆盖与 GT 高度吻合，大件物体（沙发/床/TV）边界准；**边缘偏松**仍可见（patch-14 粗特征 + 28×28 RoI 分辨率）。→ 见 §9 边界优化方向。

### 8.2 房间头 room10
- **帧级 Top-1 = 0.722**（分层 scene-disjoint val，2668 帧）；per-frame 泄漏 split 为 0.970（再次印证泄漏假象）。
- **场景级多帧投票 Top-1 = 0.824**（68 场景，softmax 概率求和）——比帧级 +10pp，是部署相关指标。

| 类别 | 帧级 acc | 场景级 acc | 场景 n |
|---|---|---|---|
| bedroom / bathroom / kitchen | 0.81 / 0.95 / 0.85 | **1.00 / 1.00 / 1.00** | 13/13/7 |
| study | 0.678 | 0.846 | 13 |
| living_room | 0.712 | 0.714 | 14 |
| storage | 0.575 | (1/1) | 1 |
| hallway | 0.188 | (1/2) | 2 |
| entryway | 0.100 | **0/3** | 3 |
| laundry | 0.125 | **0/1** | 1 |
| closet | 0.225 | **0/1** | 1 |

**弱类混淆分析（帧级，预测去向）**——不是随机错，是被语义相邻类系统性吃掉：
- entryway → **living_room 69%** / study 20%（门厅单帧只见家具，"Lobby"泛指）
- laundry → bathroom 32% / closet 32% / kitchen 20%（瓷砖+管道+电器，视觉撞型）
- hallway → **study 40%** / kitchen 20%（过渡空间，单帧拍到相邻房间）
- closet → **bedroom 57%**（衣帽间像卧室一角）

**结论**：强类场景级 ~100%、房间头实际可用（~82% 场景级）；弱类天花板是 **ScanNet 本质约束**——(a) entryway/hallway 单帧语义歧义，多帧投票也救不了（整场景就像客厅）；(b) laundry/closet/storage val 仅剩 1 场景、train 仅 3-4 场景，已不可测。CE 权重+分层 val+多帧投票均已用上。弱类根治需 MP3D 补场景或合并模糊类（用户已有大致想法，留待后续）。

> 修复：`infer_room_viz.py` / `eval_room_breakdown.py` 原用 ROOM12 标签渲染 ROOM10 模型（错位），已改为按 `cfg.num_rooms` 选 ROOM10/12，并把 eval 的 scene-disjoint 改用训练同款分层 split。

---

## 9. 分割头边界优化（待办，方向梳理）

### 9.1 诊断：边界松不是分辨率问题，是模型/特征问题

先做了"全框分辨率 IoU 对比"诊断（250 图 / 903 RoI，`mask_det24val_L2`），分清边界损失来自上采样还是特征：

| 框大小 | n | **A 模型@全分辨率** | **B 28×28 网格上限** | C 56×56 上限 |
|---|---|---|---|---|
| 小(<32px) | 118 | 0.721 | 0.993 | 0.999 |
| 中 | 345 | 0.709 | 0.947 | 0.984 |
| 大(>96px) | 440 | 0.695 | 0.942 | 0.970 |
| **全部** | 903 | **0.704** | **0.951** | 0.979 |

- **B=0.951**：完美 28×28 掩码上采到全分辨率即得 IoU 0.95 → **28×28 网格本身不是瓶颈**。
- **A=0.704 ≪ B=0.951**：模型离 28×28 可达上限差 0.25 → **模型连自己的 28×28 目标都没拟合好**。
- **C 仅比 B 高 0.03** → `roi_size 28→56` 近乎无效；A 在大框未显著塌（0.695）→ 再证非分辨率问题。

**结论：边界松的瓶颈是 mask 头容量 / 冻结 patch-14 特征，不是输出分辨率。** → PointRend / roi_size↑ 这类**分辨率手段被诊断排除**。

### 9.2 修正后的优化方向（按性价比）

1. **更强的 mask 头 / FCN 解码器（head-only，先试）**：当前头仅 `3×(3×3 conv) + 1×1 predictor`，太弱。换成更多层/通道的 FCN 掩码解码器（可选 deconv），同一冻结 P2 上重训——若能把 0.70 拉向 0.95 说明是**头太弱**（cheap fix）；拉不动则确认**特征上限**。这一步隔离"头容量 vs 特征"。
2. **边界感知损失 + 训久**：BCE+Dice 加 boundary IoU / Lovász / 边缘加权，30k→更多步。
3. **D2b LoRA 松 backbone 末层**（特征级，最重）：若强化头仍堵不住 0.70→0.95，则边界保真受冻结 patch-14 限制（与 doc 16 检测 FP_loc 同根因、同 D2b 决策点）。

~~推理级阈值/形态学、roi_size↑、PointRend、更细特征层~~ —— 分辨率类手段经 §9.1 诊断排除（28 网格上限已 0.95，模型才 0.70）。

倾向：先 1（强化头，隔离瓶颈），配 2，再据结果决定是否上 3（D2b）。

### 9.3 执行结果 + 过拟合探针：**结论翻转为"泛化差距"**（2026-06-16）

按 §9.2 做了 1+2：mask 头 3conv/128/28×28 → **4conv/256 + deconv → 56×56**（0.44M→2.33M），损失 BCE → **边界加权 BCE**（形态学梯度边界带 ×6）+ Dice，训练 `checkpoints/mask_strong`（base 冻结）。

**强化头 + 边界损失结果**（全分辨率诊断，同 §9.1 口径）：

| 框大小 | 旧 A | 新 A | 28 上限 B |
|---|---|---|---|
| 全部 | 0.704 | **0.729** | 0.951 |

→ +2.5pp（各尺寸一致，大框 +2.8pp 最多），真实但温和；val_iou@56 平台 ~0.733。**只填平 0.704→0.951 缺口的 ~10%**，一度倾向"特征到顶"。

**但过拟合探针（决定性，20 图狠训）证伪了"特征到顶"**：

```
step 0: train_IoU@56 0.614  →  step 500: 0.996  →  step 2000+: 1.0000
```

- **20 图过拟合 IoU = 1.0** → 冻结 patch-14 特征 + 当前头**完全能表达精细边界**，特征**不是**瓶颈。
- 全集 val 仅 0.733 → 这是 **泛化差距（记得住 1.0、泛化只 0.73）**，不是表达能力上限。

**修正后的判据与方向**：
- ❌ **D2b LoRA 不是这里的答案**（特征够用，松 backbone 治不了泛化差距）——§9.2 第 3 条作废。
- ❌ 强化头容量也非主因（能过拟合说明容量足；val 不动是泛化）。
- ✅ **瓶颈 = 泛化差距**，可由 **数据增强 / 正则 / loss** 影响。新优先级：
  1. **数据增强**（翻转/尺度抖动/颜色）——泛化差距 0.27 大，aug 通常杠杆最猛；
  2. **principled 边界损失**（Kervadec Boundary / HD / Lovász，比 band-BCE 更学可泛化边界）；
  3. 正则（dropout/weight decay）、更多/更多样数据。
- **保留**：强化头（2.33M）+ 边界加权 BCE 的 +2.5pp 是白赚的，作为新基线。

> 教训：诊断（全分辨率 A vs 网格上限 B）+ 过拟合探针 两步把"分辨率→特征→泛化"的真因逐层锁定，避免了往 D2b（错方向）砸资源。下一步立项 = 数据增强 + 边界损失，不是 D2b。
