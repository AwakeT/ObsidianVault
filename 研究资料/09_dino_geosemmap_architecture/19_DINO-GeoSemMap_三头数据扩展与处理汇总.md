# DINO-GeoSemMap 三头数据扩展与处理汇总

> 版本：v1.0
> 日期：2026-06-15
> 合并来源：[17 检测数据扩展与长尾平衡 det24](17_DINO-GeoSemMap_检测数据扩展与长尾平衡_det24.md) · [18 房间头与分割头数据处理](18_DINO-GeoSemMap_房间头与分割头数据处理.md)
> 关联：[16 冻结DINOv2检测头优化方法总结](16_DINO-GeoSemMap_冻结DINOv2检测头优化方法总结.md) · [10 数据集构成](10_DINO-GeoSemMap_数据集构成.md)
> 状态：**配方确定 + 数据/管线就绪 + 单元/smoke 验证通过；正式训练尚未跑，无 recall / IoU / Top-1 结果**

---

## 0. 总览

本轮把除深度头外的三个 head 统一收敛到一份数据扩展与处理记录：

- **检测头**：从 COCO 单源 `det11` 扩到 **LVIS + Objects365 的 det24**，并用 RFS 重采样 + 正样本加权 QFL 处理长尾。
- **分割头**：因 mask head 是 class-agnostic，不做 per-class 平衡，只补干净可比的 LVIS-val mask 验证集与显式 manifest 入口。
- **房间头**：从 `ROOM12` 收缩为 **ROOM10**，补全 ScanNet `sceneType` 元数据，基于全量 957 场景重建数据，并接入 CE 类权重与分层 scene-disjoint val。

当前工作重点是**数据轴与管线打通**。效果验证仍待正式训练：检测 40k、mask 30k、room 20k。

---

## 1. 检测头：det11 到 det24

doc 16 的 E 系列已在 `det11` 上证明冻结 DINOv2 patch-14 特征下，损失级优化基本到顶；Objects365 数据扩展被列为后备 E5。本轮执行这条数据轴：把检测类空间和训练数据源同时扩开。

### 1.1 类空间

| 子集 | 类数 | 来源 | 说明 |
|---|---:|---|---|
| `det11` | 11 | COCO | E 系列使用的 COCO 可映射家居类 |
| `det24` | 24 | LVIS + O365 | LVIS 覆盖 25 类，剔除 `wardrobe` |
| `det25` | 25 | LVIS | 完整 LVIS 覆盖，保留作参考，不用于训练 |

- OBJ30 各源覆盖：COCO 11 / LVIS 25 / O365 23；LVIS∪O365 = 26。
- `wardrobe` 在 LVIS+O365 train 只有 **12 实例 / 9 图**，O365 完全没有，继续训练只会过拟合，因此从 head 中剔除。
- O365 独有的 `nightstand` 不在 det24 内，由 collate remap 自动丢弃，保持单一干净标签空间。
- `tv_stand`、`shoe_cabinet`、`door`、`window` 四类仍无公开数据源覆盖，后续需 Habitat 渲染补。

### 1.2 数据组成

| 集 | manifest | 图 | 实例 |
|---|---|---:|---:|
| train | `det_lvis_L2` + `det_o365train_L2` | **59,696** | **~324k** |
| val | `det_det24val_L2` | **941** | **~3.5k** |

- train 从 COCO 单源扩到 LVIS train + Objects365 train，数据量已超过 det L2 的 40k 图目标。
- val 使用 LVIS val 单源：与 LVIS train / O365 train 天然不交叉，且能覆盖全部 24 类；O365 val 缺 `bookshelf` / `curtain`。
- `det_det24val_L2` 会匹配 `det_*_L2` glob，正式训练必须显式传 `--train_manifests`，不要只用 `--level L2`，避免 val 混入 train。

### 1.3 长尾全貌

| 头部类 | 实例数 | 尾部类 | 实例数 |
|---|---:|---|---:|
| chair | 110,060 | dining_table | 511 |
| lamp | 49,115 | washing_machine | 288 |
| cabinet | 41,738 | bookshelf | 113 |
| desk | 29,462 | wardrobe | 12（已剔除） |

实例数跨 **4 个数量级**。如果只做随机采样 + 等权 QFL，尾部类会被头部类淹没。

### 1.4 长尾处理

**L1：repeat-factor 重采样**

- 复用 LVIS 式 repeat factor：类权重 `rf = max(1, sqrt(thr / f_c))`，图权重取所含类别 rf 的最大值。
- 泛化到裸 `UnifiedDataset`，并支持按 det24 remap 后计频。
- `rfs_thr=0.01` 主要放大尾部：`bookshelf x2.9`、`washing_machine x1.9`、`dining_table x1.4`，其余接近 1。

**L2：inverse-sqrt 类别加权 QFL**

- 新增 `inverse_sqrt_class_weights`：按 `1 / sqrt(max(n_c, floor))` 得到 raw weight，再按实例频率加权归一化，让加权均值约为 1。
- 实测权重范围：`chair 0.46` 到 `bookshelf 14.29`。
- 权重**只乘正样本 cell**。如果整列类别通道都加权，会同时放大稀有类在所有背景位置的负梯度，反而压制该类。

### 1.5 代码与产物

| 文件 | 改动 |
|---|---|
| `dino_gsm/data/class_subset.py` | 新增 `DET24`，剔除 `wardrobe` |
| `dino_gsm/data/split.py` | 泛化 `repeat_factor_weights`，新增 `inverse_sqrt_class_weights` |
| `dino_gsm/config.py` | `FCOSConfig` 加 `cls_balance` / `cls_weight_floor` |
| `dino_gsm/losses/fcos_loss.py` | 加 `cls_weight` buffer、`set_class_weights`、QFL 正样本 cell 加权 |
| `scripts/train_fcos.py` | 加 `--balance` / `--rfs_thr` / `--cls_balance`，从 train 集计算并注入权重 |
| `dgsm-data/scripts/prep_det25_val.py` | 生成 `det_det24val_L2.jsonl` |

验证状态：权重/采样单元验证通过，`cls_balance=none` 回归路径保持不变，det24 smoke 6 步端到端跑通，941 图 val eval 跑通。

### 1.6 训练命令

```bash
MAN=/mnt2/dgsm_data/manifests
python scripts/train_fcos.py --base_ckpt checkpoints/depth_v1/step_0098000.pt \
  --train_manifests $MAN/det_lvis_L2.jsonl $MAN/det_o365train_L2.jsonl \
  --val_manifests   $MAN/det_det24val_L2.jsonl \
  --class_subset det24 --assigner tal --train_adapter \
  --balance --cls_balance invsqrt --rfs_thr 0.01 \
  --steps 40000 --batch_size 8 --eval_every 1000 --out checkpoints/fcos_det24
```

正式 det24 训练应使用 D2a 私有 adapter + `depth_v1` warm-start，以便与 E3 公平对比。

### 1.7 边界

- 数据总量不再是检测头瓶颈，本轮主要解决类空间扩展与长尾平衡。
- doc 16 指出的冻结 patch-14 特征上限仍存在，det24 是与 D2b LoRA 正交的数据轴干预。
- 合理预期是尾部新类从“完全检不到”变为“可检测但偏弱”，不应预期单靠 det24 达到 70%@conf0.5。
- 若 `bookshelf` / `washing_machine` / `dining_table` 等尾部仍上不去，下一步考虑 EQLv2，下调稀有类负梯度，而不只是上调正梯度。

---

## 2. 分割头：补 val 与显式入口

### 2.1 关键认知

当前 mask head 是 **class-agnostic**：

- `mask_head.py` 输出通道为 1。
- `mask_loss.py` 使用单通道 BCE + Dice。
- `mask_targets.py` 按 GT box 裁出 28x28 二值 mask 监督。
- 全流程不读类别标签。

因此，检测头的 det24 类空间统一和 per-class 加权不适用于 mask head。mask 的长尾问题不致命：它不丢类别，只可能让边界形状偏向高频类。当前 `mask_lvis` 框内 IoU 约 **0.71**，主要瓶颈是 patch-14 粗特征与 28x28 RoI 分辨率导致的边界偏松。

### 2.2 改动

- 新建 `dgsm-data/scripts/prep_mask_val.py`：仿 `prep_det25_val.py` 的 rare-class-first 采样，但写入 `masks_rle`，并设置 `heads=["det","mask"]`。
- 源数据为 LVIS val，与 mask train 的 LVIS train 不交叉。
- 产物为 `mask_det24val_L2.jsonl`：**941 图 / 3673 实例 / 24 类 / 100% 带 mask**。
- `scripts/train_mask.py` 增加 `--train_manifests` / `--val_manifests`，对齐 `train_fcos.py`。
- collate 仍用 `detection_collate`，不做 remap，避免 masks 不随 remap 过滤的潜在问题。
- 本轮不改 head / loss / target，也不做 mask 长尾重采样。

### 2.3 训练数据与命令

训练集：

- `mask_cocotrain_L1`：35,326 图。
- `mask_lvis_L2`：19,696 图。
- 合计 **55,022 图**。

```bash
MAN=/mnt2/dgsm_data/manifests
python scripts/train_mask.py --base_ckpt checkpoints/depth_v1/step_0098000.pt \
  --train_manifests $MAN/mask_cocotrain_L1.jsonl $MAN/mask_lvis_L2.jsonl \
  --val_manifests   $MAN/mask_det24val_L2.jsonl \
  --steps 30000 --batch_size 4 --eval_every 1000 --out checkpoints/mask_det24
```

验证状态：smoke 通过，未训基线 `val_iou` 约 0.64。

---

## 3. 房间头：ROOM10 与 ScanNet 扩充

### 3.1 类空间：ROOM12 到 ROOM10

原 `ROOM12` 中：

- `balcony`：ScanNet 0 帧。
- `dining_room`：全库约 1 个场景。

二者既学不动，也测不准，因此从训练头剔除，改为 **ROOM10**：

```text
[living_room, bedroom, bathroom, kitchen, study, hallway, storage, entryway, laundry, closet]
```

改动：

- `taxonomy.py` 加 `ROOM10` / `ROOM10_IDX` / `room10_index_from_scannet_scenetype`。
- `dining`、`balcony`、类外映射为 `-1` 并丢弃。
- 保留 `ROOM12` 不动，给未来 MP3D region 映射使用。
- `prep_scannet_room.py` 产出 ROOM10 连续索引。
- `RoomConfig.num_rooms` 从 12 改为 10。

### 3.2 数据扩充

`prep_scannet_room` 从 `depth_scannet_{level}` manifest 继承帧，再附加 `room` 字段。因此扩 room 的实际工作是：补全 `sceneType` 元数据，并重建 depth prep。

| 步骤 | 扩充前 | 扩充后 |
|---|---:|---:|
| `sceneType .txt` | 469 | **1513** |
| `depth_scannet_L1` | 28,931 帧 / 450 场景 | **36,897 帧 / 951 场景** |
| `room_scannet_L1` | 21,716 帧 / 335 场景（ROOM12） | **24,400 帧 / 630 场景（ROOM10）** |

踩坑记录：第一次重跑 depth prep 只覆盖 450 场景，因为本地 `scannetv2_train.txt` 是旧的 450 场景子集。改用全量盘上 957 场景列表 `scannet_ondisk_all.txt`，并设置 `--num 38280`（957x40），才真正扩开。

### 3.3 ROOM10 分布

ROOM10 帧分布：

- 强类：`living_room 5361` / `study 4885` / `bedroom 4852` / `bathroom 4757` / `kitchen 2383`
- 中等类：`entryway 889` / `hallway 756`
- 弱类：`storage 199` / `laundry 159` / `closet 159`

主要收益是场景数从 335 扩到 630，约 1.9x。弱类 `storage` / `laundry` / `closet` 仍只有 4-5 个场景，这是 ScanNet 本身上限，根治需要 MP3D。

### 3.4 类别均衡与分层 val

原 `train_room` 是裸 CE + shuffle + 普通 scene-disjoint split，没有类别均衡。本轮改为：

- `split.py:room_class_weights`：按 `record["room"]` 统计帧数，做 inverse-sqrt 权重，并按频率加权归一化。
- 实测权重范围：`living_room 0.78` 到 `laundry/closet 4.97`。
- `split.py:split_train_val_by_scene_stratified`：按 room 类分桶 scene，每桶留约 `val_frac` 进 val，稀有类在 train/val 两边都有。
- `room_loss` 加 `weight` 参数。
- `train_room` 默认启用分层 split + CE 权重，同时保留 `--no_balance_ce` / `--unstratified` / `--frame_split` 回退旧行为。

分层 split 修复了弱类不可测问题：

| 类 | 普通 scene-disjoint val 场景 | 分层 val 场景 |
|---|---:|---:|
| storage | 0 | 1 |
| laundry | 0 | 1 |
| closet | 0 | 1 |

### 3.5 代码与产物

| 文件 | 改动 |
|---|---|
| `dgsm-data/dgsm_data/taxonomy.py` | 新增 ROOM10 与 room10 映射 |
| `dgsm-data/scripts/prep_scannet_room.py` | 产出 ROOM10 |
| `dino_gsm/config.py` | `RoomConfig.num_rooms` 从 12 改为 10 |
| `dino_gsm/data/split.py` | 新增 `room_class_weights` 与 `split_train_val_by_scene_stratified` |
| `dino_gsm/losses/room_loss.py` | CE 加 `weight` 参数 |
| `scripts/train_room.py` | 接入分层 val、CE 权重和回退开关 |
| `scripts/download_scannet_meta.py` | 新建，用于补 `sceneType .txt` |

新增/重建 manifest：

- `depth_scannet_L1`：重建为 36,897 帧 / 951 场景，旧版 `.bak`。
- `room_scannet_L1`：重建为 ROOM10 24,400 帧 / 630 场景，旧 ROOM12 `.bak`。

### 3.6 训练命令

```bash
python scripts/train_room.py --base_ckpt checkpoints/depth_v1/step_0098000.pt \
  --level L1 --steps 20000 --batch_size 16 --out checkpoints/room10
```

验证状态：smoke 通过，分层 split 为 train 21,732 / val 2,668（68 held-out 场景），CE 权重 min 0.78 / max 4.97，`num_rooms=10`，eval 跑通。

### 3.7 风险与边界

- `depth_scannet_L1` 已被重 prep 覆盖，旧版已备份；depth 头重训时会自然使用更多场景。
- 弱类 `storage` / `laundry` / `closet` 仍受 ScanNet 上限约束，根治需 MP3D region。
- `balcony` / `dining_room` 被剔除，不代表业务上不需要，而是当前 ScanNet 无法支持。
- 全量 957 场景含同一物理房间的 `_01` / `_02` 重扫，当前按不同 scene_key 处理，分层 split 下可能有轻微泄漏；后续可按 base scene 合并。

---

## 4. 当前边界与下一步

- 三个 head 的数据与管线均已就绪，但正式训练结果尚未产生。
- 检测头下一步看 det24 正式 40k 后的 val recall，重点观察 `bookshelf` / `washing_machine` / `dining_table` 等尾部类。
- 分割头下一步跑 30k 正式训练，当前不优先做长尾重采样或按类 mIoU。
- 房间头下一步跑 20k 正式训练，并关注弱类 Top-1 与混淆；中长期需接 MP3D region。
- Habitat 渲染仍是补齐 in-domain 数据、缺源检测类和房间弱类的关键路径。

---

## 5. 除深度头外三头数据量表

| Head | 训练数据 | 训练规模 | 验证数据 | 验证规模 | 备注 |
|---|---|---:|---|---:|---|
| 检测头 Detection | `det_lvis_L2` + `det_o365train_L2` | **59,696 图 / ~324k 实例** | `det_det24val_L2` | **941 图 / ~3.5k 实例** | det24，24/24 类覆盖；`wardrobe` 已剔除 |
| 分割头 Mask | `mask_cocotrain_L1` + `mask_lvis_L2` | **55,022 图** | `mask_det24val_L2` | **941 图 / 3,673 实例** | class-agnostic，val 100% 带 mask |
| 房间头 Room | `room_scannet_L1` | **24,400 帧 / 630 场景** | 分层 scene-disjoint split | **2,668 帧 / 68 held-out 场景** | ROOM10，`balcony` / `dining_room` 已剔除 |

