# DINO-GeoSemMap 架构设计

> 版本：v0.3.1  
> 日期：2026-05-27

---

## 1. 系统定位

DINO-GeoSemMap 是一个面向家居机器人的视觉感知模型，为下游语义地图和 VLN 导航提供结构化感知输出。

模型本身只处理 RGB 图像，不预测位姿。位姿由外部 SLAM 模块提供，感知结果与位姿在下游合成 ASM 时汇合。

```text
SLAM 模块 ─────────── 提供位姿 ──────────┐
                                                  │
RGB_t ──→ DINO-GeoSemMap  ──→ 感知结果 ───→ ASM 合成 ──→ VLN / VLA
            │                         │
            │ depth                   │ bbox + mask + depth + pose
            │ boxes + Top-50 labels   │   → 解算物品 3D 位置
            │ bbox-based masks        │   → 合成 Agent Semantic Map
            │ room type               │
            └─────────────────────────┘
```

感知模型输出四类信息：

| 输出 | 说明 |
|---|---|
| 深度 | 每帧 dense depth + 置信度 |
| 目标检测 | 家居 Top-50 目标物品的 bbox + 类别 |
| 目标分割 | 基于检测 bbox 裁剪的前景/背景 mask |
| 房间分类 | 当前场景的房间类型 |

---

## 2. 模型架构

### 2.1 设计原则：Frozen Backbone + 独立 Heads

DINOv2 backbone **全程冻结**，不参与任何 head 的梯度更新。四个推理头各自独立训练，互不影响：

```text
DINOv2 ViT (frozen)
  │
  ▼
Multi-scale Feature Adapter (训练一次后冻结，作为共享特征基座)
  │
  ├── Depth Head          ← 独立训练
  ├── Detection Head      ← 独立训练
  ├── Mask Head           ← 独立训练（依赖 Detection Head 的 bbox）
  └── Room Head           ← 独立训练
```

这样设计的好处：

- **完全解耦**：任何一个 head 的训练、调参、替换不会影响其他 head 的效果；
- **可独立迭代**：某个 head 效果不好，只需重训该 head，不用担心回归其他任务；
- **训练简单**：每个 head 的训练就是一个标准的"固定特征 + 训练分类/回归头"流程；
- **推理共享**：所有 head 共享一次 DINOv2 forward pass，不增加 backbone 计算开销。

### 2.2 整体结构

```text
RGB frame / keyframe
  │
  ▼
DINOv2 ViT-B/L backbone (frozen)
  │
  ├── intermediate features: layer [4, 11, 17, 23]
  │
  ▼
Multi-scale Feature Adapter (frozen after initial training)
  │                                   ┌──────────────────────┐
  ├── DPT Depth Head                  │                      │
  │     └── depth_t, confidence_t     │  各 head 独立训练      │
  │                                   │  互不影响              │
  ├── RF-DETR-style Object Decoder    │  DINOv2 + Adapter    │
  │     └── boxes_t, top50_labels_t   │  全程冻结              │
  │                                   │                      │
  ├── Bbox-based Mask Head            └──────────────────────┘
  │     └── instance_masks_t
  │
  └── Room Semantics Head
        └── room_type_t, room_confidence_t
```

### 2.3 Backbone：DINOv2 ViT（frozen）

使用 DINOv2 ViT 作为主干网络，权重全程冻结，从 4 个中间层抽取多尺度特征：

```text
F_4, F_11, F_17, F_23  ∈ R^{H_p × W_p × C}

H_p = H / 14,  W_p = W / 14
C = 768 (ViT-B) / 1024 (ViT-L)
```

### 2.4 Multi-scale Feature Adapter（训练一次后冻结）

将 DINOv2 的 patch-grid 特征转换为多尺度 feature pyramid。Adapter 在第一阶段（Depth Head 训练时）一起训练，训练完成后冻结，作为所有 head 共享的固定特征基座：

```text
[F_4, F_11, F_17, F_23]
  → layer norm + linear projection
  → reshape to 2D feature maps
  → upsample / downsample
  → P2 (1/4), P3 (1/8), P4 (1/16), P5 (1/32)
```

| 特征层 | 分辨率 | 用途 |
|---|---|---|
| P2 | 1/4 | bbox 内目标分割 |
| P3 | 1/8 | 检测 decoder cross-attention |
| P4 | 1/16 | 全局语义、房间分类 |
| P5 | 1/32 | 大目标、场景上下文 |

---

## 3. Depth Head

### 3.1 结构

DPT-style decoder，输出全图 dense depth：

```text
P2, P3, P4, P5
  → DPT fusion blocks
  → upsample
  → depth_map (H × W)
  → confidence (H × W)
  → valid_mask (H × W)
```

### 3.2 深度尺度策略

```text
训练：metric depth（米制深度监督）
```

### 3.3 损失函数

```text
L_depth =
  λ_l1   · valid_L1(depth_pred, depth_gt)
+ λ_grad · gradient_L1(depth_pred, depth_gt)
+ λ_si   · scale_invariant_log_loss
+ λ_conf · uncertainty_weighted_loss

可选：L_sparse = L1(sample(depth_pred, sparse_uv), sparse_depth)
```

---

## 4. 家居 Top-50 Detection Head

### 4.1 结构

借鉴 RF-DETR 的 DINOv2 + transformer decoder 范式：

```text
Object Queries Q_obj ∈ R^{Nq × C}
  │
  ├── cross-attend to P2-P5
  │
  ▼
Transformer Decoder layers
  │
  ├── box head      → [cx, cy, w, h]
  ├── objectness    → object / no-object
  └── Top-50 class  → 家居场景 Top-50 分类
```

### 4.2 检测类别

以 **家居场景 Top-50 目标物品** 作为检测对象，覆盖家具、家电、厨卫用品、日常用品、门窗通道等关键类别。50 类使用标准 CE 分类训练，保证高频物品检测的稳定性和高召回。

类别列表可随部署场景迭代调整，总量控制在 50 类左右。

### 4.3 损失函数

```text
L_det =
  λ_box  · L1(box_pred, box_gt)
+ λ_giou · GIoU(box_pred, box_gt)
+ λ_obj  · focal_loss(objectness)
+ λ_cls  · CE(top50_class)

匹配：Hungarian matching (box + giou + class + objectness cost)
```

---

## 5. Bbox-based Mask Head

### 5.1 结构

基于检测 bbox，在框内做前景/背景分割：

```text
检测到的 bbox_i
  → RoIAlign 从 P2 裁剪 bbox 区域特征
  → 几层卷积
  → mask_i: bbox_h × bbox_w 的前景/背景 mask
```

不使用 query-mask 实例分割架构，仅做 bbox 内的轻量二值分割。下游结合 depth + SLAM pose 即可解算物品 3D 位置。

### 5.2 损失函数

```text
L_mask =
  λ_bce  · BCE(mask_pred, mask_gt)
+ λ_dice · Dice(mask_pred, mask_gt)

无 mask 标注时可用 SAM / SAM2 伪标签做弱监督。
```

---

## 6. Room Semantics Head

### 6.1 结构

基于全局特征的轻量场景分类头：

```text
DINOv2 CLS token + Global Average Pooled features (P4 / P5)
  → MLP classifier
  → room_type + room_confidence
```

### 6.2 房间类别

覆盖家居场景 15-25 类常见房间：

```text
客厅、卧室、厨房、卫生间、书房、餐厅、
阳台、走廊、玄关、储物间...
```

过渡区域允许输出低置信度或多标签 soft prediction。

### 6.3 损失函数

```text
L_room = λ_room · CE(room_pred, room_gt)

可选 label smoothing。
```

---

## 7. 输入输出定义

### 7.1 输入

```text
必需：
  I_t: RGB 图像

可选（不进入模型训练，ASM 合成时使用）：
  K_t: 相机内参
  T_wc_t: 相机位姿（SLAM 提供）
  P_sparse_t: SLAM 稀疏点（深度尺度校准用）
```

### 7.2 输出

```text
Y_t = {
  depth: {
    depth_map: H × W,
    confidence: H × W,
    valid_mask: H × W
  },

  objects: [
    {
      box_2d: [x1, y1, x2, y2],
      objectness: float,
      top50_label: string,
      top50_score: float,
      instance_mask: bbox_h × bbox_w,
      mask_score: float
    }
  ],

  room: {
    room_type: string,
    room_confidence: float,
    room_logits: [N_room_classes]
  }
}
```

---

## 8. 下游 ASM 合成

ASM 合成不属于感知模型，是下游流程。感知输出 + SLAM 位姿一起解算物品 3D 位置：

```text
for each detected object i:
  1. bbox_i + mask_i → 确定目标像素区域
  2. depth_t → 取对应区域深度
  3. K_t → 反投影到相机坐标系 3D 点
  4. T_wc_t → 变换到世界坐标系
  5. 计算 3D 中心点 / 3D bbox
```

ASM 输出：

```text
ASM_t = {
  objects: [
    {
      top50_label,
      world_position: [x, y, z],
      world_bbox_3d: optional,
      confidence,
      source_keyframe_id
    }
  ],
  room_type_t
}
```

---

## 9. 推理流程

### 9.1 高频深度

```text
every frame:
  RGB → DINOv2 → DPT Head → depth + confidence

目标频率：server 10-20Hz / edge 3-10Hz
```

### 9.2 关键帧检测 / 分割 / 房间分类

```text
on SLAM keyframe:
  RGB → DINOv2 → Object Decoder → boxes + Top-50 labels
                → Bbox Mask Head → instance masks
                → Room Head → room_type

目标频率：1-5Hz
```

### 9.3 ASM 合成

```text
on keyframe with detection results:
  bbox + mask + depth + SLAM pose
    → 解算物品 3D 位置 → 写入 ASM
```

---

## 10. 训练计划

### 10.1 核心原则

**DINOv2 全程冻结，各 head 完全独立训练。** 训练顺序上只有一个依赖：Adapter 在第一阶段训练后冻结，后续所有 head 基于冻结的 DINOv2 + 冻结的 Adapter 输出的固定特征独立训练。

```text
阶段 0: 训练 Adapter（与 Depth Head 一起，训完后冻结 Adapter）
        ↓
        DINOv2 (frozen) + Adapter (frozen) = 固定特征基座
        ↓
阶段 1-4: 各 head 独立训练，互不依赖，可并行
```

### 10.2 训练阶段

| 阶段                 | 训练内容                               | DINOv2 | Adapter | 数据                                   |
| ------------------ | ---------------------------------- | ------ | ------- | ------------------------------------ |
| 0: Adapter + Depth | Adapter + DPT Depth Head           | frozen | **训练**  | RGB-D / LiDAR / synthetic depth      |
| 1: Detection Head  | Object Decoder + Top-50 class head | frozen | frozen  | COCO / Objects365 / LVIS 映射到 Top-50  |
| 2: Mask Head       | Bbox-based Mask Head               | frozen | frozen  | COCO Panoptic / LVIS / SAM 伪标签       |
| 3: Room Head       | Room Semantics MLP                 | frozen | frozen  | ScanNet / HM3D / Matterport3D / 合成数据 |

阶段 0-3 之间**无依赖关系**，可以并行训练。唯一的前置条件是阶段 0 完成（Adapter 冻结）。

Mask Head 在推理时依赖 Detection Head 的 bbox 输出，但训练时可以用 ground truth bbox 独立训练。

### 10.3 数据需求

| 模块 | 起步数据量 | 较好效果数据量 | 监督 |
|---|---|---|---|
| Adapter + Depth Head | 50-100 万帧 | 200-500 万帧 | depth、valid mask |
| Detection Head | 10-30 万张 | 50-100 万张 | boxes、Top-50 class |
| Bbox Mask Head | 5-20 万张 | 30-100 万张 | instance masks / SAM 伪标签 |
| Room Head | 5-20 万帧 | 20-80 万帧 | room type labels |

### 10.4 总体量级

| 目标 | 数据量级 | 预期效果 |
|---|---|---|
| 原型可用 | 30-80 万帧/图 | 常见家居物品和深度可用 |
| 支撑 VLN | 100-300 万帧/图 | ASM 稳定，Top-50 高召回 |
| 工程成熟 | 300-1000 万帧/图 | 深度、检测、分割、房间都较稳 |

---

## 11. 时间规划

### 11.1 训练时间估算

以 8×A100 (80GB) 为基准，使用 ViT-B/14 backbone，按"原型可用"数据量级估算：

| 阶段  | 训练内容                 | 数据量       | 预计训练时间 | 前置条件    |
| --- | -------------------- | --------- | ------ | ------- |
| 0   | Adapter + Depth Head | 50-100 万帧 | 5-10 天 | 无       |
| 1   | Detection Head       | 10-30 万张  | 5-10 天 | 阶段 0 完成 |
| 2   | Mask Head            | 5-20 万张   | 3-7 天  | 阶段 0 完成 |
| 3   | Room Head            | 5-20 万帧   | 1-3 天  | 阶段 0 完成 |

阶段 0-3 无依赖关系，可在多台机器上并行训练。

### 11.2 里程碑

| 里程碑              | 预计时间点   | 交付物                       | 验收标准                        |
| ---------------- | ------- | ------------------------- | --------------------------- |
| M0: 数据就绪         | 第1周末    | 各任务训练数据准备完成               | 数据格式验证通过，dataloader 可跑通     |
| M1: Depth 可用     | 第3周末    | Adapter + Depth Head 训练完成 | 室内场景 depth 定性可用，边界清晰        |
| M2: Detection 可用 | 第4-5周   | Detection Head 训练完成       | Top-50 常见物品召回率 >70%         |
| M3: 全链路打通        | 第6周末    | 四个 Head 全部就绪              | 单帧输入 → 感知输出 → ASM 合成 全流程跑通  |
| M4: ASM 集成验证     | 第 7-8 周 | 感知模型 + SLAM + ASM 联调      | 多帧序列 ASM 构建稳定，物品 3D 位置误差可接受 |

### 11.3 资源需求

| 资源     | 最低配置        | 推荐配置        | 说明                               |
| ------ | ----------- | ----------- | -------------------------------- |
| 训练 GPU | 4×A100 40GB | 8×A100 80GB | 阶段 0 对显存要求最高（Adapter + Depth 联训） |
| 并行训练   | 1 台         | 2-3 台       | 阶段 1-4 可在不同机器上并行，缩短总周期           |

---

## 12. 部署配置

### 12.1 模型规格

| 规格 | Backbone | 输入分辨率 | 用途 |
|---|---|---|---|
| v0.3-S | DINOv2 ViT-S/14 | 392-504 | 端侧实验 |
| v0.3-B | DINOv2 ViT-B/14 | 504-518 | 平衡版，推荐起步 |
| v0.3-L | DINOv2 ViT-L/14 | 518-704 | 服务器训练，高质量 ASM |

### 12.2 参数量与模型大小

DINOv2 Backbone 参数量（冻结，不参与训练）：

| Backbone | 层数  | 隐藏维度 | 参数量   |
| -------- | --- | ---- | ----- |
| ViT-S/14 | 12  | 384  | ~21M  |
| ViT-B/14 | 12  | 768  | ~86M  |
| ViT-L/14 | 24  | 1024 | ~300M |

各 Head 参数量（可训练部分，近似估算）：

| 模块 | 参数量 | 说明 |
|---|---|---|
| Multi-scale Feature Adapter | ~3-5M | 随 backbone 隐藏维度略有变化 |
| DPT Depth Head | ~10-15M | DPT fusion blocks + output conv |
| Detection Head | ~8-12M | 6 层 Transformer Decoder + box/class/objectness heads |
| Bbox-based Mask Head | ~2-4M | RoIAlign + 轻量卷积 |
| Room Semantics Head | ~0.3-0.5M | MLP classifier |
| **Heads 合计** | **~25-35M** | |

整体模型大小：

| 规格     | Backbone | Heads | 总参数量      | 模型文件大小 (FP16) |
| ------ | -------- | ----- | --------- | ------------- |
| v0.3-S | ~21M     | ~28M  | **~50M**  | ~100 MB       |
| v0.3-B | ~86M     | ~30M  | **~115M** | ~230 MB       |
| v0.3-L | ~300M    | ~33M  | **~330M** | ~660 MB       |

### 12.3 推理显存占用

以下为 batch_size=1、FP16 推理的 GPU 显存估算（含模型权重 + 推理激活）：

| 规格     | 分辨率     | 仅 Depth（每帧） | 全部 Heads（关键帧） |
| ------ | ------- | ----------- | ------------- |
| v0.3-S | 504×504 | ~0.5 GB     | ~0.8 GB       |
| v0.3-B | 518×518 | ~1.0 GB     | ~1.5-2.0 GB   |
| v0.3-L | 518×518 | ~2.0 GB     | ~3.0-4.0 GB   |
| v0.3-L | 704×704 | ~3.5 GB     | ~5.0-6.0 GB   |

说明：

- 所有 Head 共享一次 DINOv2 forward pass，不重复计算 backbone；
- 实时导航时每帧只跑 Depth Head，显存按"仅 Depth"列计；
- 关键帧触发时额外运行 Detection + Mask + Room，显存按"全部 Heads"列计；
- 实际显存受 PyTorch 内存分配策略、CUDA context 等因素影响，上表为净推理估算值。

### 12.4 运行模式

```text
实时导航:
  depth: every frame (10-20Hz)
  detection / segmentation / room: SLAM keyframes only (1-5Hz)

离线建图:
  depth: every frame
  detection / segmentation / room: every keyframe or dense sampled frames
```



vlm-4b
dinov2持续探索-空间智能模型-实现可能性
细化整体计划
