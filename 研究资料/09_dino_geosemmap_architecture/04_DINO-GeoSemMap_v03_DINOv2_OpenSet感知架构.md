# DINO-GeoSemMap v0.3：DINOv2 语义几何感知架构

> 迭代自 [[03_DINO-GeoSemMap_v02_模型架构]]  
> 版本：v0.3.0  
> 日期：2026-05-27  
> 定位：不再预测位姿；由外部 SLAM 提供相机位姿和坐标系；本模型专注训练 **深度、开集目标检测、实例分割、房间语义分类** 四类视觉感知能力。  
> 参考：RF-DETR 的 DINOv2 backbone + DETR-style detection / segmentation 设计；LingBot-Map / VGGT 系的 DPT depth head 思路。

---

## 0. 本次迭代结论

当前版本建议从 v0.2 的“几何 + 语义 + 位姿统一模型”收敛为一个更清晰的系统边界：

```text
SLAM 模块
  负责：相机位姿、局部/全局坐标系、稀疏地图、关键帧管理

DINO-GeoSemMap v0.3 感知模型
  负责：深度、目标检测、实例分割、房间语义分类、开放词表语义 embedding

语义地图模块
  负责：把 depth + box + mask + SLAM pose 融合成 3D object observations

VLN / Qwen3-VL 模块
  负责：语言目标理解、开放词表查询、地图问答、目标重命名、决策规划
```

也就是说，v0.3 不再把位姿作为模型学习目标。位姿头、Camera token、LingBot-Map 式长期 KV-cache 都不是主路径必需项。这样做的好处是：

- 训练目标更聚焦；
- 感知模型不再和 SLAM 的定位职责重叠；
- 深度、检测、分割可以直接使用成熟视觉监督数据；
- 检测头以家居 Top-50 目标物品为闭集基线，结合 RF-DETR-style query + region-text alignment 实现开集扩展；
- Qwen3-VL 不作为 dense perception backbone，而作为语义层 / VLN 层使用。

---

## 1. 与 v0.2 的关键差异

| 维度 | v0.2 设计 | v0.3 设计 |
|---|---|---|
| 位姿 | 模型内置 Camera Head 预测 pose | 移除；由外部 SLAM 提供 pose |
| 地图坐标系 | 模型通过流式 attention 维护几何上下文 | SLAM 维护坐标系和关键帧 |
| 主干 | DINOv2 + LingBot-Map 48 层交替注意力 | DINOv2 ViT backbone + lightweight multi-scale adapter |
| 长期 KV-cache | Anchor / Window / Trajectory Memory | 默认不需要；可选短时 feature cache 做稳定性 |
| 输出 | pose / depth / detection / segmentation / room type | depth / boxes / masks / object embeddings / Top-50 labels / open-vocab labels / room type |
| 房间类型 | Room Readout 直接分类 | Room Semantics Head；基于 CLS token + 全局特征的场景级房间分类 |
| 开集检测 | 未明确 | 家居 Top-50 闭集检测 + RF-DETR-style open-vocab alignment |
| Qwen3-VL 角色 | 可作为 VLN 基模候选 | 不做 dense backbone；做语义教师、开放词表验证和 VLN 决策 |

---

## 2. 总体架构

### 2.1 系统级数据流

```text
RGB_t
  │
  ├──────────────────────────────┐
  │                              │
  ▼                              ▼
DINO-GeoSemMap v0.3          SLAM 模块
  │                              │
  │ 输出 depth / boxes / masks    │ 输出 pose_t / intrinsics / keyframe graph
  │                              │
  └──────────────┬───────────────┘
                 ▼
        语义地图融合模块
                 │
                 ├── 3D object observations
                 ├── instance memory
                 ├── local traversability evidence
                 ├── object / zone / room semantic nodes
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
  ├── RF-DETR-style Object Decoder (Top-50 + open-set)
  │     └── boxes_t, objectness_t, top50_labels_t, query_embeddings_t
  │
  ├── RF-DETR-Seg-style Mask Head
  │     └── instance_masks_t
  │
  ├── Open-Vocabulary Alignment Head
  │     └── region_embeddings_t, text_similarity_t, open_vocab_labels_t
  │
  └── Room Semantics Head
        └── room_type_t, room_confidence_t
```

### 2.3 推荐默认配置

| 配置项 | 推荐值 | 说明 |
|---|---|---|
| Backbone | DINOv2 ViT-L/14 | 精度优先；如果端侧部署则换 ViT-B/14 或 ViT-S/14 |
| 输入分辨率 | 518×392 或 518×378 | 与 LingBot-Map / DINOv2 patch=14 兼容；也可使用 518×518 pad 模式 |
| Patch size | 14 | DINOv2 ViT 默认 |
| 检测 queries | 100 到 300 | 导航场景通常不需要太多 query；公共场景可提高 |
| 深度输出 | 原图分辨率 / 1× 上采样 | 输出 dense depth + confidence |
| mask 输出 | 1/4 到 1/2 分辨率后上采样 | 实时优先；关键帧可高分辨率 refine |
| 推理频率 | depth 10-20Hz；det/seg 1-5Hz | 语义头建议关键帧运行 |

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
text_vocab: 当前任务 / 地图需要关注的开放词表类别
```

注意：`K_t` 和 `T_wc_t` 不进入主干训练也可以工作，但在地图融合阶段必须使用。

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
      top50_label: optional string,
      top50_score: optional float,
      open_vocab_label: optional string,
      open_vocab_score: optional float,
      region_embedding: D,
      instance_mask: H × W or H/4 × W/4,
      mask_score: float
    }
  ],

  room: {
    room_type: string,
    room_confidence: float,
    room_logits: optional [N_room_classes]
  },

  frame_embedding: optional D,
  dense_features: optional H' × W' × C
}
```

其中 `region_embedding` 是后续开放词表检索、跨帧关联、地图节点合并的关键字段。

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
| P2 | 1/4 | mask 边界、细粒度分割 |
| P3 | 1/8 | detection decoder cross-attention |
| P4 | 1/16 | 全局语义 |
| P5 | 1/32 | 大目标、场景上下文 |

---

## 5. Depth Head

### 5.1 设计目标

Depth Head 负责输出当前帧 dense depth，用于把 2D mask / box 投影成 3D object observations。

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
地图融合时使用 confidence 过滤低质量区域
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

## 6. RF-DETR-style 家居 Top-50 + Open-Set Detection Head

### 6.1 参考 RF-DETR 的部分

RF-DETR 官方描述为一个基于 DINOv2 vision transformer backbone 的实时 transformer 检测和实例分割架构，支持 detection 和 instance segmentation，并提供多个模型尺寸。公开 benchmark 中 RF-DETR-N/S/M/L 使用 384/512/576/704 等分辨率，Seg 版本也提供 Nano 到 2XL 等配置。

在 v0.3 中，我们不直接照搬 RF-DETR 的全部实现，而是借鉴其核心范式：

```text
DINOv2 backbone
  → transformer object decoder
  → object queries
  → boxes + objectness + masks
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
  ├── Top-50 class head
  │     → 家居场景 Top-50 目标物品分类
  │
  └── region embedding head
        → z_obj
```

### 6.3 家居场景 Top-50 检测类别设计

v0.3 的检测头以 **家居场景 Top-50 目标物品** 作为核心检测对象。这 50 类物品是从家居、室内导航、服务机器人等场景中，按出现频率和导航/交互价值筛选出的最具代表性的目标物品集合，覆盖家具、家电、厨卫用品、日常用品、门窗通道等关键类别。

设计思路：

- **以家居场景为中心**：不追求通用目标检测的类别覆盖面，而是聚焦机器人在家居环境中最常遇到、最需要感知的物品；
- **Top-50 作为闭集基线**：这 50 类使用标准 CE 分类训练，确保高频物品检测的稳定性和高召回；
- **超出 Top-50 的物品交给开集机制**：不在 Top-50 之内的长尾物品，由 region embedding + open-vocab alignment 或 VLM 低频验证来处理；
- **Top-50 是可迭代的**：随着部署场景和数据积累，可以调整具体类别，但总量控制在 50 类左右，保持检测头轻量。

这样的设计保证了两个核心能力：

1. 常见家居物品有稳定可靠的检测结果，地图模块可以直接使用；
2. 不在列表中的物品不会被丢弃，而是进入 unknown / open-vocab 通路等待后续识别。

### 6.4 开集检测设计

在 Top-50 闭集检测的基础上，开集能力通过两层机制实现：

#### 第一层：class-agnostic object proposal

检测器先回答一个与类别无关的问题：

```text
这里是不是一个可独立成图的物体 / 区域？
```

输出：

```text
box_i, objectness_i, query_embedding_i
```

这一层不区分物品是否在 Top-50 之内，只负责"发现物体"。

#### 第二层：Top-50 分类 + region-text alignment

对每个 proposal，同时走两条路：

```text
路径 A：Top-50 class head
  直接分类到 50 个家居物品类别之一，或判定为 "不在 Top-50 中"

路径 B：region-text alignment
  z_i = MLP(query_i)
  e_c = TextEncoder(class_name_or_phrase)
  score(i, c) = cosine(z_i, e_c) / temperature
```

两条路的关系：

- 如果 Top-50 分类得分高，直接使用闭集标签，稳定可靠；
- 如果 Top-50 分类得分低但 objectness 高，说明检测到了一个不在 Top-50 中的物品，此时依赖 region-text alignment 或 VLM 给出标签。

TextEncoder 可选：

| Text encoder | 角色 | 说明 |
|---|---|---|
| CLIP / SigLIP text encoder | 实时开放词表分类 | 轻量、工程稳定 |
| Qwen3-VL / VLN 基模 | 低频语义验证 / 重命名 | 不建议每帧跑 |
| 自建类别 prompt encoder | Top-50 之外的扩展词表 | 最轻量 |

推荐默认：

```text
Top-50 内物品：直接使用 closed-set class head
Top-50 外物品：SigLIP/CLIP text embedding 实时匹配
难例 / unknown：Qwen3-VL 低频验证和重命名
```

### 6.5 检测损失

```text
L_det =
  λ_box · L1(box_pred, box_gt)
+ λ_giou · GIoU(box_pred, box_gt)
+ λ_obj · focal_loss(objectness)
+ λ_cls · CE(top50_class)
+ λ_ov  · contrastive_loss(region_embedding, text_embedding)
```

匹配方式：

```text
Hungarian matching:
  cost = box cost + giou cost + class cost + objectness cost
```

---

## 7. Instance Segmentation Head

### 7.1 为什么优先做实例分割

语义地图需要沉淀的是“一个个物体实例”，不是单纯语义像素：

```text
这把椅子
那张桌子
这个电梯门
这个水杯
这个可移动物品
```

因此 v0.3 的分割头优先输出 instance mask，而不是只做 semantic segmentation。

### 7.2 结构

推荐 RF-DETR-Seg / Mask2Former 风格的 query mask 设计：

```text
object query_i
  → mask embedding m_i

pixel feature map P2
  → pixel embedding map E

mask_i = sigmoid(m_i · E)
```

也可以加轻量 mask refinement：

```text
coarse mask
  → boundary refinement block
  → final mask
```

### 7.3 与检测头的关系

检测和分割共用 object queries：

```text
query_i
  ├── box_i
  ├── objectness_i
  ├── class / open-vocab embedding_i
  └── mask_i
```

这样每个实例天然带有：

- 2D box；
- 2D mask；
- 类别 / 文本 embedding；
- 地图关联特征。

### 7.4 分割损失

```text
L_mask =
  λ_bce  · BCE(mask_pred, mask_gt)
+ λ_dice · Dice(mask_pred, mask_gt)
+ λ_edge · boundary_loss(optional)
```

如果训练数据只有 box，没有 mask，可以加弱监督：

```text
box-supervised mask loss
SAM / SAM2 pseudo mask distillation
```

---

## 8. Room Semantics Head

### 8.1 设计目标

Room Semantics Head 负责对当前帧做场景级别的房间类型分类，输出"当前相机看到的是什么房间"。这个信息对语义地图至关重要：

- 地图中的 room / zone 节点需要有明确的房间类型标签；
- VLN 导航指令经常包含房间级别的参照（"去厨房"、"找到卧室里的床"）；
- 房间类型可以作为物品检测的上下文先验（厨房里更可能出现冰箱和微波炉）。

### 8.2 结构

Room Semantics Head 是一个轻量的场景分类头，基于全局特征而不是局部 object query：

```text
DINOv2 CLS token + Global Average Pooled features (from P4 or P5)
  │
  ▼
MLP classifier
  │
  ├── room_type: 家居常见房间类别
  └── room_confidence: 分类置信度
```

因为房间分类是场景级任务，不需要 object-level 的空间分辨率，用 CLS token 和全局池化特征就足够了。这使得 Room Semantics Head 非常轻量，几乎不增加推理开销。

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

可选：加 label smoothing，因为房间边界本身就是模糊的（站在厨房和客厅交界处，两个标签都合理）。

训练数据来源：

- ScanNet、HM3D、Matterport3D 等室内数据集自带房间标注；
- 合成数据（ProcTHOR、AI2-THOR）可以生成大量带房间标签的数据；
- 也可以用 VLM 对无标注数据做伪标签。

### 8.5 与语义地图的关系

Room Semantics Head 的输出在语义地图中用于：

```text
每个关键帧输出 room_type_t
  → 语义地图根据关键帧位置和 room_type 投票
  → 划分 room / zone 区域
  → 为 zone 内的 object node 提供房间上下文
```

房间分类在关键帧频率运行即可（1-5Hz），不需要每帧都跑。

---

## 9. Open-Vocabulary 训练策略

### 8.1 开集检测不是只加一个分类层

要尽可能接近开集检测，需要同时具备：

1. **objectness 泛化**：没见过的物体也能被 proposal 出来；
2. **region-text 对齐**：proposal 能和文本类别或描述对齐；
3. **unknown 机制**：不确定时输出 unknown，而不是强行归到错类；
4. **低频 VLM 验证**：地图沉淀前可用 Qwen3-VL 对关键 unknown 实例做重命名。

### 8.2 推荐数据组合

| 数据类型 | 用途 | 示例 |
|---|---|---|
| closed-set detection | 学稳定 box 和基础类别 | COCO、Objects365、LVIS、OpenImages |
| instance segmentation | 学 mask | COCO Panoptic、LVIS、ADE20K、ScanNet、Cityscapes |
| phrase grounding | 学 region-text 对齐 | RefCOCO、Visual Genome、Flickr30k Entities、GRIT 类数据 |
| open-vocab detection | 学长尾类别 | LVIS rare classes、ODinW、Roboflow 多域数据 |
| navigation-specific labels | 学地图可用目标 | ScanNet、HM3D、Matterport3D、InteriorGS、自采室内数据 |
| pseudo labels | 扩大开放词表 | GroundingDINO / OWL-ViT / Qwen3-VL / SAM2 伪标签 |

### 8.3 类别组织

不要把全部类别硬塞成一个巨大 closed-set softmax。建议分三层：

```text
Layer A: 家居 Top-50 闭集类别
  用 CE 训练，覆盖家居场景最常见的 50 类目标物品，要求稳定高召回

Layer B: 开放词表 embedding
  用 contrastive / matching loss 训练，覆盖 Top-50 以外的长尾物品

Layer C: 地图语义标签
  由 VLN / Qwen3-VL 低频确认，可随任务动态变化
```

### 8.4 unknown 策略

每个实例输出：

```text
unknown_score = objectness - max(top50_score, open_vocab_similarity)
```

地图沉淀规则：

```text
如果 objectness 高但 top50_score 和 open_vocab_score 都低:
  先写入 unknown_object_i
  保留 crop / mask / embedding
  等待 Qwen3-VL 或人工 / 后续观察重命名
```

这样比强行分类更适合长期语义地图。

---

## 10. 与 SLAM 的接口

### 10.1 SLAM 提供什么

```text
SLAM_t = {
  T_wc_t: 当前相机位姿,
  K_t: 相机内参,
  keyframe_id: 当前关键帧编号,
  sparse_points_t: 可选稀疏点,
  local_map_t: 可选局部地图,
  tracking_status: good / weak / lost
}
```

### 10.2 感知模型提供什么

```text
Perception_t = {
  depth_t,
  depth_confidence_t,
  boxes_t,
  masks_t,
  object_embeddings_t,
  labels_t,
  scores_t,
  room_type_t,
  room_confidence_t
}
```

### 10.3 融合为 3D Object Observations

```text
for each instance i:
  mask_i + depth_t
    → visible point cloud_i in camera frame
    → use T_wc_t to transform to world frame
    → compute 3D centroid / bbox / confidence
    → associate with existing map node
```

输出给语义地图：

```text
ObjectObservation = {
  instance_id_local,
  class_label,
  open_vocab_embedding,
  world_centroid,
  world_bbox,
  mask_area,
  evidence_count,
  confidence,
  source_keyframe_id
}
```

### 10.4 SLAM 状态对融合的影响

| SLAM 状态 | 地图写入策略 |
|---|---|
| tracking good | 正常写入 object observation |
| tracking weak | 只缓存 2D 观测，不立即沉淀到 3D |
| tracking lost | 暂停地图更新，等待重定位 |
| loop closure 后位姿修正 | 重投影 / 更新已有 object node 的世界位置 |

---

## 11. 训练数据需求

### 11.1 按任务拆分

| 模块 | 起步数据量 | 比较好效果数据量 | 监督 |
|---|---|---|---|
| Depth Head | 50 万到 100 万帧 | 200 万到 500 万帧 | depth、valid mask、可选 sparse depth |
| Detection Head | 10 万到 30 万张 | 50 万到 100 万张 | boxes、class、objectness |
| Instance Seg Head | 5 万到 20 万张 | 30 万到 100 万张 | instance masks、panoptic masks |
| Open-Vocab Alignment | 10 万到 50 万 region-text pairs | 100 万到 500 万 region-text pairs | region-text matching、phrase grounding |
| Room Semantics Head | 5 万到 20 万帧 | 20 万到 80 万帧 | room type labels（ScanNet / HM3D / Matterport3D / 合成数据） |
| Navigation-specific Fine-tune | 2 万到 10 万关键帧 | 10 万到 50 万关键帧 | 室内/室外导航目标、门、电梯、路口、可通行区域等 |

### 11.2 推荐训练阶段

#### 阶段 A：Depth warm-up

目标：先让 DINOv2 + DPT 学会稳定深度。

```text
数据：RGB-D / LiDAR depth / synthetic depth / MVS pseudo depth
训练：DINOv2 frozen，训练 adapter + DPT head
时间：8×A100 上约 3 到 7 天起效
```

#### 阶段 B：RF-DETR-style detection pretrain

目标：学习稳定 object proposal 和基础类别检测。

```text
数据：COCO / Objects365 / LVIS / OpenImages / Roboflow 多域检测数据
训练：DINOv2 frozen 或 LoRA，训练 object decoder
时间：8×A100 上约 3 到 7 天起效
```

#### 阶段 C：Segmentation pretrain

目标：让 object query 同时输出可用 instance mask。

```text
数据：COCO Panoptic / LVIS / ADE20K / ScanNet / 自建 mask 数据
训练：检测 decoder + mask head 联训
时间：8×A100 上约 5 到 14 天
```

#### 阶段 D：Open-vocab alignment

目标：让 region embedding 能和文本类别 / 短语对齐。

```text
数据：region-text pairs、phrase grounding、VLM 伪标签
训练：region projection head + text contrastive loss
时间：8×A100 上约 3 到 10 天
```

#### 阶段 E：Navigation / map fine-tuning

目标：让模型更适合语义地图和 VLN。

```text
数据：机器人关键帧、SLAM pose、室内/室外目标标注、导航常见物体、可通行区域
训练：全 heads 小学习率联训，DINOv2 LoRA 可开
时间：8×A100 上约 3 到 10 天
```

### 11.3 总体量级

| 目标 | 数据量级 | 预期效果 |
|---|---|---|
| 原型可用 | 30 万到 80 万帧/图 | 常见物体和深度可用，开集较弱 |
| 可支撑 VLN 地图输入 | 100 万到 300 万帧/图 + 100 万级 region-text pairs | 语义地图稳定，长尾类别可初步识别 |
| 工程成熟版本 | 300 万到 1000 万帧/图 + 300 万到 1000 万 region-text pairs | 深度、检测、分割和开集标签都较稳 |

---

## 12. 推理流程

### 12.1 高频深度

```text
for each RGB frame:
  RGB → DINOv2 → DPT Head → depth + confidence
```

频率目标：

```text
server GPU: 10-20Hz
edge GPU: 3-10Hz
```

### 12.2 关键帧检测 / 分割 / 房间分类

```text
if SLAM keyframe or semantic trigger:
  RGB → DINOv2 → Object Decoder → boxes / embeddings
  RGB → DINOv2 → Mask Head → instance masks
  RGB → DINOv2 → Room Semantics Head → room_type / room_confidence
  optional: text_vocab similarity
```

触发条件：

- SLAM 产生新关键帧；
- 位移 / 旋转超过阈值；
- VLN 请求目标搜索；
- 当前画面出现高 objectness unknown；
- 地图模块需要补充语义证据。

### 12.3 Qwen3-VL 低频验证

```text
if unknown object is important or queried by VLN:
  crop + mask + local context
    → Qwen3-VL / VLM
    → label / description / relation
    → update map node
```

Qwen3-VL 不参与每帧 dense prediction，只参与低频语义确认。

---

## 13. 参数量和部署建议

### 13.1 三种模型规格

| 规格 | Backbone | 输入 | 用途 |
|---|---|---|---|
| v0.3-S | DINOv2 ViT-S/14 | 392/448/504 | 端侧实验，低算力 |
| v0.3-B | DINOv2 ViT-B/14 | 504/518 | 平衡版，推荐起步 |
| v0.3-L | DINOv2 ViT-L/14 | 518/704 | 服务器训练与高质量地图 |

### 13.2 推荐运行模式

```text
实时导航:
  depth: every frame
  detection/segmentation: SLAM keyframes only
  room semantics: SLAM keyframes only
  open-vocab text matching: keyframes or queried objects
  Qwen3-VL: low-frequency verification only

离线建图:
  depth: every frame
  detection/segmentation: every keyframe or dense sampled frames
  room semantics: every keyframe
  open-vocab matching: all instances
  Qwen3-VL: unknown / high-value nodes
```

---

## 14. 为什么不把 Qwen3-VL 当主 backbone

Qwen3-VL 适合：

- 理解用户目标；
- 做视觉问答；
- 解释地图节点；
- 开放词表重命名；
- 处理复杂语义关系；
- 作为 VLN / VLA 决策模型。

但它不适合作为 v0.3 的 dense perception 主干：

| 问题 | 说明 |
|---|---|
| 深度输出不自然 | Qwen3-VL 的视觉 token 面向语言推理，不适合直接恢复 dense depth |
| mask 输出成本高 | 像素级分割需要规则网格特征，ViT dense features 更合适 |
| 实时性差 | Qwen3-VL 适合低频推理，不适合每帧高频 dense prediction |
| 训练复杂 | 同时训练 LLM、视觉编码器和 dense heads 成本高，稳定性差 |
| 工程边界不清 | 容易把感知、语义、决策混到一个模型里，难以调试 |

因此推荐：

```text
DINOv2: 感知主干
SLAM: 位姿与坐标系
Qwen3-VL: 语义解释和 VLN 决策
```

---

## 15. 风险与待验证点

1. **开集检测的 objectness 泛化**
   - 即使 region-text alignment 做得好，如果 detector 没有 proposal 出未知物体，开集能力仍会失败。
   - 需要用 LVIS rare classes、ODinW、多域数据和 VLM 伪标签补 objectness。

2. **深度和 SLAM 尺度一致性**
   - 单目 depth 可能存在尺度漂移。
   - 地图融合时必须用 SLAM sparse points / 关键帧尺度做校正或置信度过滤。

3. **mask 边界对 3D 物体沉淀影响很大**
   - mask 粗糙会直接污染 3D object bbox。
   - 关键节点可以用高质量 mask refinement 低频修正。

4. **开放词表标签不应过早写死**
   - 地图节点应保存 embedding、crop、历史观测和 label history。
   - 不确定类别先写为 unknown，后续再由 VLN / Qwen3-VL 验证。

5. **DINOv2 单帧感知缺少时序稳定性**
   - SLAM keyframe association 可以解决一部分。
   - 也可以增加轻量 temporal smoothing，不必恢复 LingBot-Map 的完整 GCA。

---

## 16. 最终推荐方案

推荐把 v0.3 命名为：

```text
DINO-GeoSemMap v0.3: DINOv2 Open-Set Geo-Semantic Perception
```

核心定义：

```text
输入：
  RGB frame / keyframe
  optional text vocabulary
  optional SLAM sparse depth for scale alignment

模型：
  DINOv2 backbone
  DPT depth head
  RF-DETR-style detection decoder (Top-50 + open-set)
  RF-DETR-Seg-style mask head
  open-vocab region-text alignment head
  room semantics head

外部模块：
  SLAM provides pose and coordinate frame
  semantic map fuses observations
  Qwen3-VL verifies long-tail semantics and serves VLN

输出：
  depth + confidence
  boxes + objectness
  instance masks
  region embeddings
  家居 Top-50 labels
  open-vocab labels / unknown objects
  room type + room confidence
```

一句话总结：

> v0.3 不再做”端到端 SLAM 替代”，而是做一个面向语义地图和 VLN 的 DINOv2 视觉感知底座：用 SLAM 保证几何坐标，用 DINOv2 保证 dense perception，用 RF-DETR-style decoder + 家居 Top-50 闭集分类保证检测/分割的基线稳定性，用 Room Semantics Head 提供场景级房间语义，用 open-vocab alignment 和 Qwen3-VL 保证开放语义扩展能力。

---

## 参考

- RF-DETR official repository: https://github.com/roboflow/rf-detr
- RF-DETR documentation: https://rfdetr.roboflow.com/develop/
- [[03_DINO-GeoSemMap_v02_模型架构]]
- [[VGGT]]
- [[VGGT-Omega]]
- [[ABot-N0]]

