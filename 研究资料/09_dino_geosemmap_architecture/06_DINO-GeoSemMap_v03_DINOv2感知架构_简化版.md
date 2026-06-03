# DINO-GeoSemMap v0.3：DINOv2 语义几何感知架构（简化版）

> 迭代自 [[05_DINO-GeoSemMap_v03_DINOv2感知架构]]  
> 版本：v0.3.1  
> 日期：2026-05-27  
> 定位：不再预测位姿；由外部 SLAM 提供相机位姿和坐标系；本模型专注训练 **深度、目标检测（家居 Top-50）、基于 bbox 的目标分割、房间语义分类** 四类视觉感知能力。  
> 参考：RF-DETR 的 DINOv2 backbone + DETR-style detection 设计；DPT depth head 思路。

---

## 0. 本次迭代结论

当前版本建议从 v0.2 的"几何 + 语义 + 位姿统一模型"收敛为一个更清晰的系统边界：

```text
SLAM 模块
  负责：相机位姿、局部/全局坐标系、稀疏地图、关键帧管理

DINO-GeoSemMap v0.3 感知模型
  负责：深度、目标检测（家居 Top-50）、基于 bbox 的目标分割、房间语义分类

下游 ASM 合成
  负责：基于感知结果（bbox / mask / depth）+ SLAM 位姿，解算物品 3D 位置，合成 Agent Semantic Map

VLN / Qwen3-VL 模块
  负责：语言目标理解、地图问答、决策规划
```

也就是说，v0.3 不再把位姿作为模型学习目标。位姿头、Camera token、LingBot-Map 式长期 KV-cache 都不是主路径必需项。这样做的好处是：

- 训练目标更聚焦；
- 感知模型不再和 SLAM 的定位职责重叠；
- 深度、检测、分割可以直接使用成熟视觉监督数据；
- 检测头聚焦家居 Top-50 目标物品，使用闭集分类训练，保证高频物品检测稳定；
- 分割只需基于检测 bbox 裁剪目标，不需要复杂的 query-mask 架构；
- 房间语义分类为 ASM 提供场景级标签；
- Qwen3-VL 不作为 dense perception backbone，而作为语义层 / VLN 层使用。

---

## 1. 与 v0.2 的关键差异

| 维度 | v0.2 设计 | v0.3 设计 |
|---|---|---|
| 位姿 | 模型内置 Camera Head 预测 pose | 移除；由外部 SLAM 提供 pose |
| 地图坐标系 | 模型通过流式 attention 维护几何上下文 | SLAM 维护坐标系和关键帧 |
| 主干 | DINOv2 + LingBot-Map 48 层交替注意力 | DINOv2 ViT backbone + lightweight multi-scale adapter |
| 长期 KV-cache | Anchor / Window / Trajectory Memory | 默认不需要；可选短时 feature cache 做稳定性 |
| 输出 | pose / depth / detection / segmentation / room type | depth / boxes / masks / Top-50 labels / room type |
| 检测类别 | 未明确 | 家居 Top-50 闭集检测 |
| 分割 | 独立 query-mask 分割头 | 基于检测 bbox 的轻量目标分割 |
| 房间类型 | Room Readout 直接分类 | Room Semantics Head；基于 CLS token + 全局特征 |
| Qwen3-VL 角色 | 可作为 VLN 基模候选 | 不做 dense backbone；做 VLN 决策 |

---

## 2. 总体架构

### 2.1 系统级数据流

```text
RGB_t
  │
  ├──────────────────────────────────┐
  │                                  │
  ▼                                  ▼
DINO-GeoSemMap v0.3              SLAM 模块
  │                                  │
  │ depth / boxes / masks /          │ pose_t / intrinsics
  │ top50_labels / room_type         │
  │                                  │
  └────────────────┬─────────────────┘
                   ▼
           ASM 合成（下游）
                   │
                   │  bbox + mask + depth + pose
                   │    → 解算物品 3D 位置
                   │    → 合成 Agent Semantic Map
                   │
                   ▼
              VLN / VLA
```

### 2.2 模型内部结构

```text
RGB frame / keyframe
  │
  ▼
DINOv2 ViT-B/L backbone
  │
  ├── intermediate features: layer [4, 11, 17, 23]
  │
  ▼
Multi-scale Feature Adapter
  │
  ├── DPT Depth Head
  │     └── depth_t, depth_uncertainty_t, valid_mask_t
  │
  ├── RF-DETR-style Object Decoder (Top-50)
  │     └── boxes_t, objectness_t, top50_labels_t
  │
  ├── Bbox-based Mask Head
  │     └── instance_masks_t（基于检测 bbox 裁剪分割）
  │
  └── Room Semantics Head
        └── room_type_t, room_confidence_t
```

### 2.3 推荐默认配置

| 配置项 | 推荐值 | 说明 |
|---|---|---|
| Backbone | DINOv2 ViT-L/14 | 精度优先；如果端侧部署则换 ViT-B/14 或 ViT-S/14 |
| 输入分辨率 | 518×392 或 518×378 | 与 DINOv2 patch=14 兼容；也可使用 518×518 pad 模式 |
| Patch size | 14 | DINOv2 ViT 默认 |
| 检测 queries | 100 到 300 | 家居场景通常不需要太多 query |
| 深度输出 | 原图分辨率 / 1× 上采样 | 输出 dense depth + confidence |
| mask 输出 | bbox 内区域 | 基于检测框裁剪后做前景/背景分割 |
| 推理频率 | depth 10-20Hz；det/seg/room 1-5Hz | 语义头建议关键帧运行 |

---

## 3. 输入输出定义

### 3.1 输入

模型只需要 RGB 图像作为必需输入：

```text
I_t: 当前 RGB 图像
```

可选输入：

```text
K_t: 相机内参，由 SLAM / 标定模块提供
T_wc_t: 当前相机位姿，由 SLAM 提供
P_sparse_t: SLAM 稀疏点，可用于深度尺度校准或训练监督
```

注意：`K_t` 和 `T_wc_t` 不进入主干训练也可以工作，但在下游 ASM 合成阶段必须使用。

### 3.2 输出

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

## 4. Backbone：DINOv2 ViT

### 4.1 为什么继续用 DINOv2

DINOv2 更适合作为 v0.3 主干，原因是：

- patch tokens 保留规则空间网格，适合 depth / mask 这类 dense prediction；
- 自监督视觉特征对几何、边界、物体区域有较强先验；
- RF-DETR 官方实现已经验证了 DINOv2 backbone + transformer detector 的检测 / 分割路线；
- 相比 Qwen3-VL，DINOv2 更轻、更稳定、更容易做高频推理。

### 4.2 输出特征

从 DINOv2 中取多层特征：

```text
F_4, F_11, F_17, F_23
```

每层特征形状：

```text
F_l ∈ R^{H_p × W_p × C}
H_p = H / 14
W_p = W / 14
C = 1024  (ViT-L)
```

这些特征分别用于：

- 浅层：边缘、纹理、小物体；
- 中层：局部结构、轮廓；
- 深层：语义、物体区域、全局上下文。

### 4.3 Multi-scale Feature Adapter

DINOv2 原始输出是 patch grid，不是 FPN。需要一个轻量 adapter 把不同层特征转成多尺度 feature pyramid：

```text
[F_4, F_11, F_17, F_23]
  → layer norm + linear projection
  → reshape to 2D feature maps
  → upsample / downsample
  → P2, P3, P4, P5
```

推荐输出：

| 特征层 | 分辨率 | 用途 |
|---|---|---|
| P2 | 1/4 | mask 边界、bbox 内分割 |
| P3 | 1/8 | detection decoder cross-attention |
| P4 | 1/16 | 全局语义、房间分类 |
| P5 | 1/32 | 大目标、场景上下文 |

---

## 5. Depth Head

### 5.1 设计目标

Depth Head 负责输出当前帧 dense depth，下游用于把 2D bbox / mask 结合 SLAM 位姿解算出物品的 3D 位置。

因为位姿由 SLAM 提供，深度头不需要承担相机运动估计，只需要关注：

- 单帧深度结构；
- 边界准确性；
- 尺度稳定性；
- 不确定性估计。

### 5.2 结构

推荐使用 DPT-style decoder：

```text
P2, P3, P4, P5
  → DPT fusion blocks
  → upsample
  → depth_map
  → depth_uncertainty / confidence
```

输出：

```text
depth_t ∈ R^{H×W}
confidence_t ∈ R^{H×W}
valid_mask_t ∈ {0,1}^{H×W}
```

### 5.3 深度尺度

有两种路线：

| 路线 | 说明 | 推荐 |
|---|---|---|
| metric depth | 直接预测米制深度 | 有 RGB-D / LiDAR / 合成数据时推荐 |
| relative depth + SLAM scale alignment | 模型预测相对深度，由 SLAM 稀疏点或尺度校正到米制 | 数据复杂、跨域更稳 |

对机器人语义地图来说，推荐：

```text
训练时尽量使用 metric depth
推理时允许用 SLAM sparse points 做尺度校正
ASM 合成时使用 confidence 过滤低质量区域
```

### 5.4 深度损失

```text
L_depth =
  λ_l1   · valid_L1(depth_pred, depth_gt)
+ λ_grad · gradient_L1(depth_pred, depth_gt)
+ λ_si   · scale_invariant_log_loss
+ λ_conf · uncertainty_weighted_loss
```

如果有 SLAM 稀疏点：

```text
L_sparse =
  L1(sample(depth_pred, sparse_uv), sparse_depth)
```

---

## 6. RF-DETR-style 家居 Top-50 Detection Head

### 6.1 参考 RF-DETR 的部分

RF-DETR 官方描述为一个基于 DINOv2 vision transformer backbone 的实时 transformer 检测和实例分割架构，支持 detection 和 instance segmentation，并提供多个模型尺寸。

在 v0.3 中，我们不直接照搬 RF-DETR 的全部实现，而是借鉴其核心范式：

```text
DINOv2 backbone
  → transformer object decoder
  → object queries
  → boxes + objectness + class labels
```

### 6.2 检测头结构

```text
Object Queries Q_obj ∈ R^{Nq × C}
  │
  ├── cross-attend to multi-scale features P2-P5
  │
  ▼
Transformer Decoder layers
  │
  ├── box head
  │     → [cx, cy, w, h]
  │
  ├── objectness head
  │     → object / no-object
  │
  └── Top-50 class head
        → 家居场景 Top-50 目标物品分类
```

### 6.3 家居场景 Top-50 检测类别设计

v0.3 的检测头以 **家居场景 Top-50 目标物品** 作为核心检测对象。这 50 类物品是从家居、室内导航、服务机器人等场景中，按出现频率和导航/交互价值筛选出的最具代表性的目标物品集合，覆盖家具、家电、厨卫用品、日常用品、门窗通道等关键类别。

设计思路：

- **以家居场景为中心**：不追求通用目标检测的类别覆盖面，而是聚焦机器人在家居环境中最常遇到、最需要感知的物品；
- **Top-50 作为闭集分类**：这 50 类使用标准 CE 分类训练，确保高频物品检测的稳定性和高召回；
- **Top-50 是可迭代的**：随着部署场景和数据积累，可以调整具体类别，但总量控制在 50 类左右，保持检测头轻量。

### 6.4 检测损失

```text
L_det =
  λ_box · L1(box_pred, box_gt)
+ λ_giou · GIoU(box_pred, box_gt)
+ λ_obj · focal_loss(objectness)
+ λ_cls · CE(top50_class)
```

匹配方式：

```text
Hungarian matching:
  cost = box cost + giou cost + class cost + objectness cost
```

---

## 7. Bbox-based Mask Head

### 7.1 设计目标

分割头的目标很简单：基于检测到的 bbox，在框内区域把目标物体从背景中分割出来。下游结合 depth 和 SLAM 位姿就能解算出物品的 3D 位置。

不需要完整的 query-mask 实例分割架构，只需要一个轻量的 bbox 内前景/背景分割。

### 7.2 结构

```text
检测到的 bbox_i
  │
  ├── 从 feature map (P2) 裁剪 bbox 区域的特征
  │
  ▼
轻量分割网络（几层卷积）
  │
  └── mask_i: bbox_h × bbox_w 的前景/背景 mask
```

具体流程：

1. 检测头输出 bbox；
2. 用 bbox 从高分辨率特征图（P2）上做 RoI 裁剪（RoIAlign）；
3. 裁剪后的特征通过几层卷积输出二值 mask；
4. mask 表示 bbox 内哪些像素属于目标物体。

这种设计类似 Mask R-CNN 的 mask branch，但更轻量，因为不需要处理复杂的跨实例关系。

### 7.3 与检测头的关系

检测和分割共用检测结果：

```text
检测头输出 →  box_i, objectness_i, top50_label_i
                │
                ▼
           Bbox-based Mask Head
                │
                └── mask_i
```

每个检测到的实例最终包含：

- 2D bbox；
- bbox 内的前景 mask；
- Top-50 类别标签。

### 7.4 分割损失

```text
L_mask =
  λ_bce  · BCE(mask_pred, mask_gt)
+ λ_dice · Dice(mask_pred, mask_gt)
```

如果训练数据只有 box 没有 mask，可以用 SAM / SAM2 生成伪标签做弱监督。

---

## 8. Room Semantics Head

### 8.1 设计目标

Room Semantics Head 负责对当前帧做场景级别的房间类型分类，输出"当前相机看到的是什么房间"。这个信息对 ASM 至关重要：

- ASM 中需要有房间/区域级别的语义标签；
- VLN 导航指令经常包含房间级别的参照（"去厨房"、"找到卧室里的床"）；
- 房间类型可以作为物品检测的上下文先验（厨房里更可能出现冰箱和微波炉）。

### 8.2 结构

Room Semantics Head 是一个轻量的场景分类头，基于全局特征：

```text
DINOv2 CLS token + Global Average Pooled features (from P4 or P5)
  │
  ▼
MLP classifier
  │
  ├── room_type: 家居常见房间类别
  └── room_confidence: 分类置信度
```

因为房间分类是场景级任务，不需要 object-level 的空间分辨率，用 CLS token 和全局池化特征就足够了。几乎不增加推理开销。

### 8.3 房间类别

房间类别覆盖家居场景中的常见房间类型，例如：

```text
客厅、卧室、厨房、卫生间、书房、餐厅、
阳台、走廊、玄关、储物间、车库、楼梯间 ...
```

具体类别数量控制在 15-25 类左右，覆盖绝大多数家居空间。对于不确定或过渡区域（如走廊到客厅的过渡），允许输出低置信度或多标签 soft prediction。

### 8.4 训练

```text
L_room = λ_room · CE(room_pred, room_gt)
```

可选加 label smoothing，因为房间边界本身就是模糊的。

训练数据来源：

- ScanNet、HM3D、Matterport3D 等室内数据集自带房间标注；
- 合成数据（ProcTHOR、AI2-THOR）可以生成大量带房间标签的数据；
- 也可以用 VLM 对无标注数据做伪标签。

---

## 9. 训练数据需求

### 9.1 按任务拆分

| 模块 | 起步数据量 | 比较好效果数据量 | 监督 |
|---|---|---|---|
| Depth Head | 50 万到 100 万帧 | 200 万到 500 万帧 | depth、valid mask、可选 sparse depth |
| Detection Head (Top-50) | 10 万到 30 万张 | 50 万到 100 万张 | boxes、Top-50 class、objectness |
| Bbox Mask Head | 5 万到 20 万张 | 30 万到 100 万张 | instance masks（或 SAM 伪标签） |
| Room Semantics Head | 5 万到 20 万帧 | 20 万到 80 万帧 | room type labels |
| Navigation-specific Fine-tune | 2 万到 10 万关键帧 | 10 万到 50 万关键帧 | 室内目标、门、可通行区域等 |

### 9.2 推荐训练阶段

#### 阶段 A：Depth warm-up

目标：先让 DINOv2 + DPT 学会稳定深度。

```text
数据：RGB-D / LiDAR depth / synthetic depth / MVS pseudo depth
训练：DINOv2 frozen，训练 adapter + DPT head
时间：8×A100 上约 3 到 7 天起效
```

#### 阶段 B：Detection pretrain

目标：学习稳定 object proposal 和 Top-50 类别检测。

```text
数据：COCO / Objects365 / LVIS / OpenImages，映射到 Top-50 类别
训练：DINOv2 frozen 或 LoRA，训练 object decoder + Top-50 class head
时间：8×A100 上约 3 到 7 天起效
```

#### 阶段 C：Mask + Room 联训

目标：让 bbox mask head 和 room semantics head 同时起效。

```text
数据：COCO Panoptic / LVIS / ScanNet / HM3D / 合成数据
训练：检测 decoder + mask head + room head 联训
时间：8×A100 上约 5 到 14 天
```

#### 阶段 D：Navigation fine-tuning

目标：让模型更适合家居场景和 VLN。

```text
数据：机器人关键帧、SLAM pose、室内目标标注、家居常见物体
训练：全 heads 小学习率联训，DINOv2 LoRA 可开
时间：8×A100 上约 3 到 10 天
```

### 9.3 总体量级

| 目标 | 数据量级 | 预期效果 |
|---|---|---|
| 原型可用 | 30 万到 80 万帧/图 | 常见家居物品和深度可用 |
| 可支撑 VLN 地图输入 | 100 万到 300 万帧/图 | ASM 稳定，Top-50 物品高召回 |
| 工程成熟版本 | 300 万到 1000 万帧/图 | 深度、检测、分割、房间分类都较稳 |

---

## 10. 推理流程

### 10.1 高频深度

```text
for each RGB frame:
  RGB → DINOv2 → DPT Head → depth + confidence
```

频率目标：

```text
server GPU: 10-20Hz
edge GPU: 3-10Hz
```

### 10.2 关键帧检测 / 分割 / 房间分类

```text
if SLAM keyframe or semantic trigger:
  RGB → DINOv2 → Object Decoder → boxes / Top-50 labels
  RGB → DINOv2 → Bbox Mask Head → instance masks
  RGB → DINOv2 → Room Semantics Head → room_type
```

触发条件：

- SLAM 产生新关键帧；
- 位移 / 旋转超过阈值；
- VLN 请求目标搜索。

### 10.3 ASM 合成（下游）

```text
for each keyframe with detection results:
  bbox + mask + depth + SLAM pose + intrinsics
    → 解算每个物品的 3D 位置
    → 写入 ASM
```

---

## 12. 参数量和部署建议

### 12.1 三种模型规格

| 规格 | Backbone | 输入 | 用途 |
|---|---|---|---|
| v0.3-S | DINOv2 ViT-S/14 | 392/448/504 | 端侧实验，低算力 |
| v0.3-B | DINOv2 ViT-B/14 | 504/518 | 平衡版，推荐起步 |
| v0.3-L | DINOv2 ViT-L/14 | 518/704 | 服务器训练与高质量地图 |

### 12.2 推荐运行模式

```text
实时导航:
  depth: every frame
  detection/segmentation/room (Top-50): SLAM keyframes only

离线建图:
  depth: every frame
  detection/segmentation/room (Top-50): every keyframe or dense sampled frames
```

---

## 13. 风险与待验证点

1. **Top-50 类别覆盖度**
   - 50 类能否覆盖实际家居场景中绝大多数需要感知的物品，需要通过数据统计验证。
   - 后续可根据实际部署数据调整类别列表。

2. **深度和 SLAM 尺度一致性**
   - 单目 depth 可能存在尺度漂移。
   - ASM 合成时必须用 SLAM sparse points / 关键帧尺度做校正或置信度过滤。

3. **bbox-based mask 精度**
   - 基于 bbox 的分割在遮挡和密集场景中可能不如 query-mask 方案精细。
   - 对于 3D 位置解算来说，bbox 内的粗略 mask 通常足够；如果精度不够，可以后续升级为更精细的分割头。

4. **房间边界模糊性**
   - 过渡区域的房间分类可能不稳定，需要在 ASM 合成层做时序投票或置信度过滤。

5. **DINOv2 单帧感知缺少时序稳定性**
   - SLAM keyframe association 可以解决一部分。
   - 也可以增加轻量 temporal smoothing。

---

## 14. 最终推荐方案

推荐把 v0.3 命名为：

```text
DINO-GeoSemMap v0.3: DINOv2 Geo-Semantic Perception
```

核心定义：

```text
输入：
  RGB frame / keyframe
  optional SLAM sparse depth for scale alignment

模型：
  DINOv2 backbone
  DPT depth head
  RF-DETR-style detection decoder（家居 Top-50）
  Bbox-based mask head
  Room semantics head

下游 ASM 合成：
  detection result + depth + SLAM pose → 物品 3D 位置 → Agent Semantic Map

输出：
  depth + confidence
  boxes + objectness + Top-50 labels
  bbox-based instance masks
  room type + room confidence
```

一句话总结：

> v0.3 是一个面向 ASM 和 VLN 的 DINOv2 视觉感知底座：用 SLAM 提供位姿，用 DINOv2 做 dense depth，用 RF-DETR-style decoder 做家居 Top-50 检测，用 bbox-based mask 分割目标，用 Room Semantics Head 分类房间，下游结合 depth + SLAM pose 解算物品 3D 位置合成 Agent Semantic Map。

---

## 参考

- RF-DETR official repository: https://github.com/roboflow/rf-detr
- RF-DETR documentation: https://rfdetr.roboflow.com/develop/
- [[05_DINO-GeoSemMap_v03_DINOv2感知架构]]
- [[03_DINO-GeoSemMap_v02_模型架构]]
