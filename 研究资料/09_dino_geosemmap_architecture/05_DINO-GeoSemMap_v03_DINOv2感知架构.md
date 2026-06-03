# DINO-GeoSemMap v0.3：DINOv2 语义几何感知架构

> 迭代自 [[03_DINO-GeoSemMap_v02_模型架构]]  
> 版本：v0.3.0  
> 日期：2026-05-27  
> 定位：不再预测位姿；由外部 SLAM 提供相机位姿和坐标系；本模型专注训练 **深度、目标检测、实例分割** 三类视觉感知能力。  
> 参考：RF-DETR 的 DINOv2 backbone + DETR-style detection / segmentation 设计；DPT depth head 思路。

---

## 0. 本次迭代结论

当前版本建议从 v0.2 的“几何 + 语义 + 位姿统一模型”收敛为一个更清晰的系统边界：

```text
SLAM 模块
  负责：相机位姿、局部/全局坐标系、稀疏地图、关键帧管理

DINO-GeoSemMap v0.3 感知模型
  负责：深度、目标检测（家居 Top-50）、实例分割

语义地图模块
  负责：把 depth + box + mask + SLAM pose 融合成 3D object observations

VLN / Qwen3-VL 模块
  负责：语言目标理解、地图问答、决策规划
```

也就是说，v0.3 不再把位姿作为模型学习目标。位姿头、Camera token、LingBot-Map 式长期 KV-cache 都不是主路径必需项。这样做的好处是：

- 训练目标更聚焦；
- 感知模型不再和 SLAM 的定位职责重叠；
- 深度、检测、分割可以直接使用成熟视觉监督数据；
- 检测头聚焦家居 Top-50 目标物品，使用闭集分类训练，保证高频物品检测稳定；
- Qwen3-VL 不作为 dense perception backbone，而作为语义层 / VLN 层使用。

---

## 1. 与 v0.2 的关键差异

| 维度          | v0.2 设计                                             | v0.3 设计                                                                       |
| ----------- | --------------------------------------------------- | ----------------------------------------------------------------------------- |
| 位姿          | 模型内置 Camera Head 预测 pose                            | 移除；由外部 SLAM 提供 pose                                                           |
| 地图坐标系       | 模型通过流式 attention 维护几何上下文                            | SLAM 维护坐标系和关键帧                                                                |
| 主干          | DINOv2 + LingBot-Map 48 层交替注意力                      | DINOv2 ViT backbone + lightweight multi-scale adapter                         |
| 长期 KV-cache | Anchor / Window / Trajectory Memory                 | 默认不需要；可选短时 feature cache 做稳定性                                                 |
| 输出          | pose / depth / detection / segmentation / room type | depth / boxes / masks / Top-50 labels |
| 房间类型        | Room Readout 直接分类                                   | 不作为核心 head；由语义地图 + VLN / VLM 低频判断                                             |
| 检测类别        | 未明确                                                 | 家居 Top-50 闭集检测                           |
| Qwen3-VL 角色 | 可作为 VLN 基模候选                                        | 不做 dense backbone；做 VLN 决策                                        |

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
  ├── RF-DETR-style Object Decoder (Top-50)
  │     └── boxes_t, objectness_t, top50_labels_t
  │
  └── RF-DETR-Seg-style Mask Head
        └── instance_masks_t
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
      top50_label: string,
      top50_score: float,
      instance_mask: H × W or H/4 × W/4,
      mask_score: float
    }
  ],

  frame_embedding: optional D,
  dense_features: optional H' × W' × C
}

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

## 6. RF-DETR-style 家居 Top-50 Detection Head

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
  ├── top50_class_i
  └── mask_i
```

这样每个实例天然带有：

- 2D box；
- 2D mask；
- Top-50 类别标签。

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

## 8. 与 SLAM 的接口

### 8.1 SLAM 提供什么

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

### 8.2 感知模型提供什么

```text
Perception_t = {
  depth_t,
  depth_confidence_t,
  boxes_t,
  masks_t,
  top50_labels_t,
  top50_scores_t
}
```

### 8.3 融合为 3D Object Observations

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
  top50_label,
  world_centroid,
  world_bbox,
  mask_area,
  evidence_count,
  confidence,
  source_keyframe_id
}
```

### 8.4 SLAM 状态对融合的影响

| SLAM 状态 | 地图写入策略 |
|---|---|
| tracking good | 正常写入 object observation |
| tracking weak | 只缓存 2D 观测，不立即沉淀到 3D |
| tracking lost | 暂停地图更新，等待重定位 |
| loop closure 后位姿修正 | 重投影 / 更新已有 object node 的世界位置 |

---

## 9. 训练数据需求

### 9.1 按任务拆分

| 模块 | 起步数据量 | 比较好效果数据量 | 监督 |
|---|---|---|---|
| Depth Head | 50 万到 100 万帧 | 200 万到 500 万帧 | depth、valid mask、可选 sparse depth |
| Detection Head (Top-50) | 10 万到 30 万张 | 50 万到 100 万张 | boxes、Top-50 class、objectness |
| Instance Seg Head | 5 万到 20 万张 | 30 万到 100 万张 | instance masks、panoptic masks |
| Navigation-specific Fine-tune | 2 万到 10 万关键帧 | 10 万到 50 万关键帧 | 室内/室外导航目标、门、电梯、路口、可通行区域等 |

### 9.2 推荐训练阶段

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

### 9.3 总体量级

| 目标 | 数据量级 | 预期效果 |
|---|---|---|
| 原型可用 | 30 万到 80 万帧/图 | 常见家居物品和深度可用 |
| 可支撑 VLN 地图输入 | 100 万到 300 万帧/图 | 语义地图稳定，Top-50 物品高召回 |
| 工程成熟版本 | 300 万到 1000 万帧/图 | 深度、检测、分割都较稳 |

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

### 10.2 关键帧检测 / 分割

```text
if SLAM keyframe or semantic trigger:
  RGB → DINOv2 → Object Decoder → boxes / Top-50 labels
  RGB → DINOv2 → Mask Head → instance masks
```

触发条件：

- SLAM 产生新关键帧；
- 位移 / 旋转超过阈值；
- VLN 请求目标搜索；
- 地图模块需要补充语义证据。

---

## 11. 参数量和部署建议

### 11.1 三种模型规格

| 规格 | Backbone | 输入 | 用途 |
|---|---|---|---|
| v0.3-S | DINOv2 ViT-S/14 | 392/448/504 | 端侧实验，低算力 |
| v0.3-B | DINOv2 ViT-B/14 | 504/518 | 平衡版，推荐起步 |
| v0.3-L | DINOv2 ViT-L/14 | 518/704 | 服务器训练与高质量地图 |

### 11.2 推荐运行模式

```text
实时导航:
  depth: every frame
  detection/segmentation (Top-50): SLAM keyframes only

离线建图:
  depth: every frame
  detection/segmentation (Top-50): every keyframe or dense sampled frames
```

---

## 12. 为什么不把 Qwen3-VL 当主 backbone

Qwen3-VL 适合：

- 理解用户目标；
- 做视觉问答；
- 解释地图节点；
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

## 13. 风险与待验证点

1. **Top-50 类别覆盖度**
   - 50 类能否覆盖实际家居场景中绝大多数需要感知的物品，需要通过数据统计验证。
   - 后续可根据实际部署数据调整类别列表。

2. **深度和 SLAM 尺度一致性**
   - 单目 depth 可能存在尺度漂移。
   - 地图融合时必须用 SLAM sparse points / 关键帧尺度做校正或置信度过滤。

3. **mask 边界对 3D 物体沉淀影响很大**
   - mask 粗糙会直接污染 3D object bbox。
   - 关键节点可以用高质量 mask refinement 低频修正。

4. **DINOv2 单帧感知缺少时序稳定性**
   - SLAM keyframe association 可以解决一部分。
   - 也可以增加轻量 temporal smoothing，不必恢复 LingBot-Map 的完整 GCA。

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
  RF-DETR-style detection decoder (家居 Top-50)
  RF-DETR-Seg-style mask head

外部模块：
  SLAM provides pose and coordinate frame
  semantic map fuses observations
  Qwen3-VL serves VLN decision

输出：
  depth + confidence
  boxes + objectness
  instance masks
  家居 Top-50 labels
```

一句话总结：

> v0.3 不再做”端到端 SLAM 替代”，而是做一个面向语义地图和 VLN 的 DINOv2 视觉感知底座：用 SLAM 保证几何坐标，用 DINOv2 保证 dense perception，用 RF-DETR-style decoder + 家居 Top-50 闭集分类保证检测/分割的稳定性。

---

## 参考

- RF-DETR official repository: https://github.com/roboflow/rf-detr
- RF-DETR documentation: https://rfdetr.roboflow.com/develop/
- [[03_DINO-GeoSemMap_v02_模型架构]]
- [[VGGT]]
- [[VGGT-Omega]]
- [[ABot-N0]]

