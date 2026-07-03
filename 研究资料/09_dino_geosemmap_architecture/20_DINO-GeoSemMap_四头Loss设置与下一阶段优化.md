# 20_DINO-GeoSemMap_四头 Loss 设置与下一阶段优化

> 整理日期：2026-06-17
> 适用版本：当前 master（commit `3030971`，E-series detection optimization 已合并）
> 关联文档：[[13_DINO-GeoSemMap_检测头优化方案]]、[[15_DINO-GeoSemMap_冻结DINOv2检测头下一阶段优化方案]]、[[16_DINO-GeoSemMap_冻结DINOv2检测头优化方法总结]]、[[18_DINO-GeoSemMap_房间头与分割头数据处理]]

---

## 0. 总览

共享基座：**冻结 DINOv2-ViT-B/14 + 各头私有 Adapter**，分辨率 896×504，`f_c=448`。
四个推理头（Detection / Mask / Room / Depth）都在这个**冻结 patch-14 特征**上训练。

| 头 | 任务 | Loss 文件 | Config |
|---|---|---|---|
| Detection (FCOS) | 稠密目标检测 | `dino_gsm/losses/fcos_loss.py` | `FCOSConfig` |
| Mask | RoI 实例掩码 | `dino_gsm/losses/mask_loss.py` | `MaskConfig` |
| Room | 全图房间分类 | `dino_gsm/losses/room_loss.py` | `RoomConfig` |
| Depth | DPT 深度回归 | `dino_gsm/losses/depth_loss.py` | `DepthTrainConfig` |

---

## 1. 参数来源标注（按头）

> 这些参数不是一次定死的，来源分四类，便于回看每个值是"借来的默认"还是"实验/诊断打出来的"。

| 标签 | 含义 |
|---|---|
| 📄 **论文默认** | 直接沿用 GFL / TOOD / DPT / MiDaS 等原论文或主流框架默认值，未改 |
| 🔬 **实验调出** | E 系列对照实验跑出来的（有 A/B 数据支撑） |
| 🩺 **诊断驱动** | 由 `diag_*.py` 探针的发现反推定的 |
| 📐 **工程约束** | 由分辨率 / 显存 / 对齐其他模块等硬约束决定 |

### 1.1 Detection（FCOS）

| 参数 | 值 | 来源 | 说明 |
|---|---|---|---|
| `lambda_cls` (QFL) | 1.0 | 📄 GFL 默认 | QFL 主分类项 |
| `focal_gamma` | 2.0 | 📄 Focal Loss 默认 | 经典 γ=2，未动 |
| `lambda_box` (GIoU) | 2.0 | 📄 框架默认 | FCOS/GFL 系框回归惯例权重 |
| `box_loss` | giou | 🔬 E-loc | 试过 ciou 平局（+0.6pp 噪声级），**未切换** |
| `lambda_dfl` | 0.25 | 📄 GFL 默认 + 🔬验证 | E-loc 试过抬到 1.0 平局，回到 0.25 |
| `reg_max` | 16 | 📄 GFL 默认 | `diag_det_loc` 撞顶率仅 1.7%，16 够用 |
| `use_objectness` | False | 🔬 E1/E2 | **实测有害后关**：E1 加之容量 0.91→0.59 |
| `assigner` | tal | 🔬 E1→E3 | center→tal 是容量 0.59→0.92 的唯一变量 |
| `tal_topk/alpha/beta` | 13/1.0/6.0 | 📄 TOOD/YOLOv8 默认 | task-aligned 标准超参 |
| `ohem_enabled` | False | 🔬 E2 | dense head 上结构性失败，实测作废后关 |
| `cls_balance` | 关 | 📐 默认 | det11 子集未启用长尾均衡 |
| `topk_eval` | 300 | 📐 对齐 DETR | 刻意对齐 DETR 300-query 预算以公平对比 |

→ 权重几乎全是 📄 论文默认；项目特有的两个关键决策（`use_objectness=False`、`assigner=tal`）都是 🔬 **实验打出来的**，不是调权重调出来的。

### 1.2 Mask

| 参数 | 值 | 来源 | 说明 |
|---|---|---|---|
| `lambda_bce` | 1.0 | 📄 惯例 | 等权三项之一 |
| `lambda_boundary` | 5（边界×6） | 🔬 项目设定 | 针对"掩码瓶颈在边界"设的加权 |
| `boundary_kernel` | 3 | 📐 工程 | 形态学梯度核大小 |
| `lambda_dice` | 1.0 | 📄 惯例 | Dice 等权 |
| `lambda_lovasz` | 1.0 | 🔬 抗过拟合杠杆 | 攻 val 0.73 平台（§9.3），设 0 可禁用 |
| `n_mask_convs/dim/deconv` | 4/256/True | 🩺 诊断驱动 | 诊断"28×28 非瓶颈、解码器容量才是"→ 加深加宽 |
| `augment` | hflip 0.5 / 0.2 | 🔬 泛化杠杆 | 针对 val 0.73 平台开，val 永不增强 |

→ 三项 loss 权重为 📄 等权惯例（1:1:1）；头结构与增强是 🩺/🔬 项目主动调的。

### 1.3 Room

| 参数                | 值   | 来源      | 说明                                           |
| ----------------- | --- | ------- | -------------------------------------------- |
| `label_smoothing` | 0.1 | 📄 标准默认 | CE 平滑常用值                                     |
| `num_rooms`       | 10  | 🔬 数据驱动 | 从 ROOM12 砍掉 ScanNet 学不动的 dining_room/balcony |

→ 唯一项目特有决策 `num_rooms` 12→10 是**实测学不动**才砍的。

### 1.4 Depth

| 参数 | 值 | 来源 | 说明 |
|---|---|---|---|
| `lambda_l1` | 1.0 | 📄 主项默认 | masked L1 主损失 |
| `lambda_grad` | 0.5 | 📄 DPT/MiDaS 惯例 | 多尺度梯度项典型半权 |
| `lambda_si` (SiLog) | 0.5 | 📄 lam=0.85 | λ=0.85 为 Eigen 原论文经典值 |
| `lambda_conf` (NLL) | 0.2 | 📄 辅助项轻权 | 不确定性估计，刻意压低 |
| `depth_activation` | inv_log | 📐 工程 | 逆深度 + log 空间 |
| `depth_raw_clamp` | 6.0 | 📐 数值稳定 | 防 raw 输出爆炸 |

→ 整套为 📄 单目深度成熟配方（DPT/MiDaS/Eigen）；权重梯度 1.0>0.5=0.5>0.2 即"主损失>正则>辅助"。

### 1.5 总规律

```
检测头：权重全借（📄 GFL/TOOD 默认）；关键决策打出来（🔬 no-obj + TAL）
Mask  ：权重借（1:1:1）；结构/增强打出来（🩺 解码器加深 + 🔬 augment/lovasz）
Room  ：参数借（📄 0.1 平滑）；类别数据驱动（🔬 12→10）
Depth ：整套借（📄 DPT/MiDaS/Eigen 经典配方），尚未专门调过
```

设计哲学：**loss 权重尽量用成熟默认值（少调），精力花在"分配器 / 头结构 / 数据 / 诊断"上**——因为 E 系列已证明冻结 backbone 下调 loss 权重撞墙，真正的杠杆在特征层与正样本分配。
