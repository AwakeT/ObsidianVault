# DINO-GeoSemMap 架构方案

> 基于 LingBot-Map、VGGT / VGGT-Omega 与 DINOv2 / DINOv3 的流式 RGB 语义建图方案  
> 版本：v0.1  
> 日期：2026-05-20  
> 阶段：方案设计，不涉及具体实现

---

## 1. 方案定位

本方案面向室内、家庭、办公等真实机器人导航场景，目标是在仅输入连续 RGB 视频流的条件下，构建一套能够同时输出检测、分割、深度、位姿，并在线生成 VLN 可用语义地图的统一模型架构。

模型需要完成的核心任务包括：

1. **检测**：输出目标类别、边界框、置信度；
2. **分割**：输出实例 mask 或语义 mask；
3. **深度**：输出稠密深度、有效区域和不确定性；
4. **位姿**：输出连续相机位姿和轨迹；
5. **地图更新**：将模型输出投影并融合到全局语义地图；
6. **VLN 支撑**：为后续视觉语言导航模块提供结构化地图查询接口。

本方案不是传统的“检测器 + 分割器 + 深度估计器 + SLAM + 地图融合”的松耦合组合，而是一个以共享视觉基础特征和共享时序几何上下文为核心的统一架构。

核心思想：

> 使用 DINOv2 / DINOv3 提供强通用视觉表征，使用 LingBot-Map 式 GCA 维护长程流式几何上下文，使用 VGGT / VGGT-Omega 式多任务几何预测统一输出 pose / depth / point-level geometry，再叠加检测与分割头，将 RGB 视频直接转化为可供 VLN 使用的结构化语义地图。

---

## 2. 总体架构

整体架构分为六层：

```text
Streaming RGB Input
        │
        ▼
[1] Visual Foundation Encoder
    DINOv2 / DINOv3 Backbone
        │
        ▼
[2] Spatio-Temporal Geometry Aggregator
    LingBot-style GCA
    Anchor Context + Sliding Window + Trajectory Memory
        │
        ▼
[3] Multi-Task Prediction Heads
    Pose Head
    Depth Head
    Detection Head
    Segmentation Head
    Map Token Head
        │
        ▼
[4] Frame-Level Perception Output
    pose_t, depth_t, boxes_t, masks_t, semantic tokens_t
        │
        ▼
[5] Online Semantic Mapping
    Occupancy Map
    Semantic Instance Map
    Topological Map
    Trajectory Memory
        │
        ▼
[6] VLN Map Interface
    Object Query
    Region Query
    Frontier Query
    Target Belief
    Navigation Context
```

从功能上看，系统由三条主线组成：

1. **视觉编码主线**：DINOv2 / DINOv3 从 RGB 图像中提取 dense patch features；
2. **流式几何主线**：LingBot-Map 式 GCA 在长序列中维护尺度、位姿和局部几何一致性；
3. **语义建图主线**：检测、分割、深度和位姿结果被投影到全局地图，形成可供 VLN 查询的语义场景表示。

---

## 3. 设计目标

### 3.1 感知目标

模型需要在每个关键帧或流式帧上提供以下信息：

| 输出 | 内容 | 用途 |
|---|---|---|
| Detection | 物体类别、bbox、置信度 | 目标识别、VLN 指令 grounding |
| Segmentation | 实例 mask、语义 mask | 精确 3D 投影、实例级地图 |
| Depth | 稠密深度、有效区域、不确定性 | 障碍物、自由空间、3D 反投影 |
| Pose | 相机位姿、尺度、置信度 | 地图坐标对齐、轨迹估计 |
| Map Tokens | 当前帧压缩语义/几何摘要 | 长程记忆、VLN 上下文压缩 |

### 3.2 建图目标

在线维护一张面向 VLN 的场景地图，而不只是点云或 occupancy grid。

地图应包含：

1. **Metric Layer**：占据栅格、自由空间、障碍物、未知区域；
2. **Semantic Layer**：物体实例、类别、位置、可见性、置信度；
3. **Topological Layer**：房间、门、通道、frontier、区域连通关系；
4. **Trajectory Layer**：机器人历史轨迹、已探索区域、已搜索区域；
5. **Language-facing Layer**：可供 VLN 模块查询和摘要的结构化描述。

### 3.3 VLN 目标

地图需要支持后续 VLN 模块完成：

1. 目标物体查询；
2. 当前区域语义理解；
3. 最近可探索 frontier 查询；
4. 房间/区域级路径规划；
5. 已搜索区域和未搜索区域管理；
6. 长程任务中的搜索恢复；
7. 目标置信度更新；
8. 语言指令与地图实体对齐。

---

## 4. 核心模块一：DINOv2 / DINOv3 视觉基础编码器

### 4.1 Backbone 角色

DINOv2 或 DINOv3 作为统一视觉基础编码器，负责从 RGB 图像中提取高质量 patch-level dense features。

输入：

```text
RGB image I_t ∈ R^{H×W×3}
```

输出：

```text
Patch tokens F_t ∈ R^{N×C}
Multi-scale features {F_t^l}
```

其中，多尺度特征用于深度和分割，patch tokens 用于时序几何聚合。

### 4.2 选择 DINO 系列的原因

选择 DINOv2 / DINOv3 的原因包括：

1. **强通用视觉表征**：能同时支撑语义、几何和密集预测任务；
2. **dense feature 质量高**：适合深度、分割、mask、匹配等 dense tasks；
3. **与 VGGT / LingBot-Map 结构兼容**：两者都基于 DINOv2 风格 ViT 特征；
4. **适合多任务共享 backbone**：比为 detection、segmentation、depth、pose 分别堆模型更经济；
5. **自监督特征具备隐式几何先验**：对深度、法线、几何匹配有天然帮助。

### 4.3 DINOv2 与 DINOv3 的方案角色

建议分阶段定位：

| 阶段 | Backbone | 角色 |
|---|---|---|
| P0 | DINOv2 ViT-L | 主验证 backbone，结构最接近 VGGT / LingBot-Map |
| P1 | DINOv3 ViT-L/H | 替代 backbone 或 teacher，用于增强 dense feature |
| P2 | DINOv3 teacher → DINOv2/Small student | 部署蒸馏路线 |

方案上可以把 DINOv3 定义为增强型 backbone，而不是立即作为唯一基座。这样能避免方案过早依赖一个尚未在当前体系中充分验证的 backbone。

---

## 5. 核心模块二：特殊 Token 系统

借鉴 LingBot-Map 和 VGGT，在每帧 patch tokens 前增加若干特殊 tokens，用于承载几何、尺度、语义和地图状态。

### 5.1 Token 组成

每帧 token 序列：

```text
[camera token]
[register tokens]
[scale token]
[semantic token]
[object token]
[free-space token]
[change token]
[patch tokens...]
```

### 5.2 Token 职责

| Token | 职责 |
|---|---|
| camera token | 聚合当前帧几何信息，用于位姿预测 |
| register tokens | 稳定 attention，吸收冗余注意力 |
| scale token | 区分锚定帧/普通帧，承载尺度信息 |
| semantic token | 当前帧场景语义摘要 |
| object token | 当前帧可见实例摘要 |
| free-space token | 当前帧自由空间/障碍物摘要 |
| change token | 动态变化、遮挡、移动物体、异常观测摘要 |
| patch tokens | 局部视觉、语义、几何 dense 信息 |

### 5.3 Token 层级记忆

这些特殊 token 不只是用于单帧预测，还用于长期记忆：

```text
patch tokens:
    短期保留，仅用于 sliding window

camera / object / semantic / map tokens:
    长期保留，形成 trajectory memory

anchor frame tokens:
    任务或地图初始化阶段长期保留，作为世界坐标锚
```

这使模型能在资源有限的情况下保留长程信息，而不是无限缓存完整图像特征。

---

## 6. 核心模块三：LingBot-style 流式几何上下文聚合器

### 6.1 模块定位

该模块是整个方案的时序中枢。它解决的问题是：

> 如何在流式 RGB 输入下，让当前帧同时感知初始坐标锚、近期局部几何和长期轨迹历史。

### 6.2 三层上下文结构

采用 LingBot-Map 的三层记忆思想：

```text
Geometric Context Attention
├── Anchor Context
│   └── 初始关键帧，全量 token，提供尺度和世界坐标锚
│
├── Sliding Window Context
│   └── 最近 k 帧，全量 patch token，提供局部几何匹配
│
└── Trajectory Memory
    └── 历史所有关键帧的 special/map tokens，提供长程一致性
```

### 6.3 Anchor Context

Anchor Context 由启动阶段的若干关键帧构成。

作用：

1. 建立初始世界坐标系；
2. 提供尺度参考；
3. 稳定后续位姿估计；
4. 为长期地图融合提供一致坐标基准；
5. 类似 VLN 中任务初始化的全局上下文。

### 6.4 Sliding Window Context

Sliding Window 保存最近若干帧的完整 patch tokens。

作用：

1. 局部深度稳定；
2. 局部相机运动估计；
3. 物体短期跟踪；
4. 遮挡恢复；
5. 连续 mask / instance 融合；
6. 减少单帧误检对地图的影响。

### 6.5 Trajectory Memory

Trajectory Memory 不保存完整历史图像，而保存每帧压缩后的 special / map tokens。

作用：

1. 长程位姿漂移抑制；
2. 场景重访识别；
3. 历史语义回忆；
4. 已探索区域压缩表示；
5. 给 VLN 提供时序化历史摘要；
6. 支撑搜索恢复和目标 belief 更新。

### 6.6 为什么不是全量历史缓存

全量保留历史 patch tokens 会导致：

1. 显存/内存线性增长；
2. 注意力计算成本上升；
3. 长距离视觉重叠低，冗余信息反而成为噪声；
4. 不利于流式实时部署。

因此方案采用：

```text
近期详细，远期压缩，初始锚定
```

这与 LingBot-Map 的设计逻辑一致，也适合 VLN 的长期任务。

---

## 7. 核心模块四：多任务预测头

聚合后的时序 token 被送入多个任务头。各 head 共享 backbone 和 GCA 输出，但任务解码相互独立。

### 7.1 Pose Head

输入：

```text
camera tokens + trajectory memory + GCA output
```

输出：

```text
T_wc: camera-to-world pose
R, t: rotation and translation
K or FOV: camera intrinsic estimate
pose confidence
tracking status
```

作用：

1. 为深度点云投影提供外参；
2. 为地图融合提供全局坐标；
3. 为机器人轨迹层提供路径；
4. 为 VLN 当前状态提供空间位置。

设计原则：

Pose Head 采用 VGGT / LingBot-Map 风格的 camera token 迭代精炼结构，而不是传统 SLAM 优化作为核心。必要时可以在系统层叠加轻量后端优化，但主模型应能前馈输出 pose。

### 7.2 Depth Head

输入：

```text
multi-scale DINO features + GCA-enhanced features
```

输出：

```text
dense depth map
depth uncertainty
valid depth mask
optional point map
```

作用：

1. 生成局部点云；
2. 更新障碍物和自由空间；
3. 将 2D mask 投影为 3D object；
4. 支撑可通行区域判断；
5. 支撑地图尺度稳定。

设计原则：

Depth Head 采用 DPT 风格多尺度 dense prediction。深度输出建议带 uncertainty，因为地图融合时不能把所有深度像素等权处理。

### 7.3 Detection Head

输入：

```text
GCA-enhanced visual tokens
semantic / object tokens
optional text / category prototypes
```

输出：

```text
object boxes
category labels
objectness confidence
optional open-vocabulary similarity
```

作用：

1. 找到 VLN 指令中可能出现的目标物；
2. 为实例分割提供 object queries；
3. 为语义地图提供候选实例；
4. 为 target belief 更新提供观测证据。

设计原则：

检测头分两层能力：

1. **固定室内类别检测**：高频、稳定、低成本；
2. **开放词汇检测**：低频、keyframe 或 VLN 请求触发。

固定类别可以覆盖常见导航实体，如：

```text
chair, table, bed, sofa, plant, door, window,
tv, monitor, desk, cabinet, shelf, refrigerator,
sink, toilet, stairs, doorway
```

开放词汇能力可作为增强能力，不建议作为 P0 阶段的唯一检测路径。

### 7.4 Segmentation Head

输入：

```text
multi-scale visual features
object queries from detection head
semantic tokens
```

输出：

```text
instance masks
semantic masks
mask confidence
```

作用：

1. 提供比 bbox 更精确的 3D 投影区域；
2. 支撑实例级地图；
3. 减少背景深度污染；
4. 支撑物体 footprint 估计；
5. 为 VLN 目标定位提供精确地图坐标。

设计原则：

检测和分割应强绑定，形成：

```text
object query → class + box + mask
```

这样每个实例在 2D、3D、地图层中都有一致 ID 线索。

### 7.5 Map Token Head

输入：

```text
semantic token
object token
free-space token
change token
trajectory memory
```

输出：

```text
frame-level semantic summary
local free-space summary
object visibility summary
scene-change score
keyframe score
```

作用：

1. 判断当前帧是否值得进入长期地图；
2. 生成可写入 trajectory memory 的压缩摘要；
3. 给 VLN 提供轻量历史状态；
4. 发现动态变化和地图冲突；
5. 控制低频语义头触发。

这是本方案相对常规多任务网络的关键扩展：模型不只是输出单帧感知结果，还显式输出用于地图和记忆更新的摘要信号。

---

## 8. 模型输出定义

每个时间步 `t` 的模型输出可以抽象为：

```text
Y_t = {
    geometry: {
        pose_t,
        depth_t,
        depth_uncertainty_t,
        optional_pointmap_t
    },

    semantics: {
        boxes_t,
        classes_t,
        scores_t,
        masks_t,
        semantic_logits_t
    },

    memory: {
        camera_token_t,
        semantic_token_t,
        object_token_t,
        free_space_token_t,
        change_token_t,
        keyframe_score_t
    }
}
```

这些输出分成两类：

1. **frame-level outputs**：用于当前帧感知；
2. **map-level update signals**：用于长期地图融合和 VLN 记忆。

---

## 9. 在线语义地图构建架构

模型输出进入在线地图系统。地图系统不是模型 head 的一部分，而是消费模型输出的状态维护层。

### 9.1 地图系统输入

```text
pose_t
depth_t
depth_uncertainty_t
boxes_t
masks_t
classes_t
scores_t
map tokens_t
```

### 9.2 地图系统输出

```text
Semantic Scene Map
Agent Trajectory
Object Database
Topological Graph
VLN Query State
```

### 9.3 地图构建流程

```text
RGB_t
  │
  ▼
Model Inference
  │
  ├── pose_t
  ├── depth_t
  ├── boxes/masks/classes
  └── map tokens
       │
       ▼
Geometry Projection
  depth + pose → local point cloud → global coordinates
       │
       ▼
Metric Map Update
  occupancy / free-space / explored / obstacle
       │
       ▼
Semantic Projection
  masks + depth + pose → 3D object observations
       │
       ▼
Instance Fusion
  associate observations with existing object landmarks
       │
       ▼
Topological Update
  rooms / regions / doors / frontiers / connectivity
       │
       ▼
VLN Interface Update
  queryable map state + navigation context
```

---

## 10. 地图表示设计

### 10.1 Metric Layer

Metric Layer 负责几何可通行性：

```text
Metric Layer
├── occupancy grid
├── free-space grid
├── obstacle height map
├── explored area mask
├── unknown area mask
└── local traversability map
```

它服务于路径规划和局部避障。

### 10.2 Semantic Instance Layer

Semantic Instance Layer 负责对象级语义地图：

```text
Object Landmark
├── object_id
├── category
├── aliases / language labels
├── 3D centroid
├── 3D bbox / footprint
├── confidence
├── observed_count
├── missing_count
├── last_seen_time
├── source_keyframes
├── visual embedding
└── state
```

对象状态建议包括：

```text
active
uncertain
missing
moved
dynamic
merged
duplicate_candidate
```

这比只记录“某处有一个 chair”更适合真实 VLN，因为真实环境中物体会被遮挡、移动、误检或重复观测。

### 10.3 Topological Layer

Topological Layer 负责把 metric map 抽象为 VLN 更容易使用的结构：

```text
Topological Layer
├── room nodes
├── region nodes
├── doorway nodes
├── frontier nodes
├── object anchor nodes
└── connectivity edges
```

典型边关系：

```text
room --contains--> object
room --connected_to--> room
region --visible_from--> viewpoint
frontier --leads_to--> unknown area
object --near--> object
object --on/under/beside--> object
```

### 10.4 Trajectory and Search Memory Layer

用于长程导航任务：

```text
Trajectory Memory
├── agent poses
├── visited regions
├── searched regions
├── failed search attempts
├── target belief updates
├── key observations
└── temporal summaries
```

每条历史记录应带时间信息：

```json
{
  "step": 18,
  "time": 42.5,
  "region": "living_room",
  "action": "inspect_table_area",
  "visible_objects": ["table", "chair", "plant"],
  "target_status": "not_found",
  "confidence_change": -0.12
}
```

这继承了 LingBot-Map 中“轨迹记忆需要时序编码”的思想。

---

## 11. 语义实例融合策略

### 11.1 单帧观测到 3D 实例

对于每个检测/分割实例：

```text
mask_i + depth_t + pose_t
    → 3D points_i
    → centroid_i
    → footprint_i
    → bbox3d_i
    → observation_i
```

每个 observation 包含：

```text
category
confidence
3D centroid
3D bbox
mask quality
depth uncertainty
viewpoint pose
timestamp
visual embedding
```

### 11.2 多帧关联

新观测需要与历史 object landmarks 关联。

关联依据：

```text
association_score =
    category similarity
  + 3D center distance
  + 3D bbox / footprint overlap
  + visual embedding similarity
  + temporal consistency
  + viewpoint visibility consistency
```

如果关联成功，则更新已有 landmark。

如果关联失败，则创建新 landmark。

如果多个 landmark 高度相似，则标记为 duplicate candidate，等待后续合并。

### 11.3 地图更新原则

语义地图不应“看到一次就永久相信”，而应使用置信度生命周期：

```text
detected repeatedly:
    confidence rises
    state = active

not detected but outside current FOV:
    no penalty

not detected while expected visible:
    missing_count rises

missing_count exceeds threshold:
    state = missing / uncertain

same category appears far from old position:
    possible moved object
```

这对真实环境非常重要，尤其是家庭场景中的动态物体。

---

## 12. 流式运行机制

### 12.1 多频率感知

建议模型和地图系统采用多频率设计：

| 模块 | 频率 | 说明 |
|---|---:|---|
| pose / depth | 10-20 Hz | 高频几何更新 |
| occupancy update | 5-10 Hz | 几何地图更新 |
| detection / segmentation | 1-3 Hz | 语义较低频即可 |
| object DB update | 1-3 Hz | 跟随语义帧 |
| topological update | 0.5-1 Hz | 区域级变化较慢 |
| VLN decision | 0.25-1 Hz | 根据任务复杂度调整 |

不建议每帧都跑完整检测分割。实际导航中，pose / depth 需要高频，语义可以 keyframe 化。

### 12.2 Keyframe 策略

语义 keyframe 触发条件：

```text
位移超过阈值
旋转超过阈值
进入新区域
接近 doorway / frontier
VLN 请求检查目标
map token 判断变化显著
检测到潜在目标
当前地图置信度不足
```

keyframe 的作用：

1. 运行完整 detection + segmentation；
2. 写入 object DB；
3. 更新 topological graph；
4. 写入长期 trajectory memory。

### 12.3 非关键帧策略

非关键帧只做：

```text
pose / depth prediction
local occupancy update
trajectory pose update
short-term tracking
```

可不做完整语义实例更新，避免计算浪费和语义噪声累积。

---

## 13. VLN 接口设计

地图最终需要通过接口提供给 VLN，而不是把原始地图全部塞给模型。

### 13.1 查询接口

推荐支持以下查询：

```text
query_objects(category, region, confidence)
query_nearest_object(category, current_pose)
query_visible_objects(current_pose, fov)
query_region_summary(region_id)
query_frontiers(current_pose)
query_unsearched_regions(target)
query_path(start, goal)
query_target_belief(target_description)
query_obstacle_local_map(radius)
query_topological_graph()
```

### 13.2 VLN 当前状态摘要

VLN 每次决策可获得结构化状态：

```json
{
  "agent": {
    "pose": [1.2, -0.5, 1.57],
    "region": "living_room",
    "heading": "toward_doorway"
  },
  "visible_objects": [
    {
      "category": "sofa",
      "distance": 1.8,
      "direction": "front-left",
      "confidence": 0.91
    },
    {
      "category": "table",
      "distance": 2.3,
      "direction": "front",
      "confidence": 0.88
    }
  ],
  "nearby_regions": [
    {
      "region": "kitchen",
      "direction": "right",
      "reachable": true
    }
  ],
  "frontiers": [
    {
      "id": "frontier_03",
      "direction": "front-right",
      "distance": 2.1
    }
  ],
  "target_belief": [
    {
      "region": "living_room",
      "score": 0.52
    },
    {
      "region": "kitchen",
      "score": 0.31
    }
  ],
  "search_memory": [
    {
      "step": 12,
      "region": "bedroom",
      "result": "target_not_found"
    }
  ]
}
```

### 13.3 VLN 使用方式

VLN 可以基于地图执行：

1. “去最近的 chair”；
2. “检查 table 附近是否有 mug”；
3. “前往尚未探索的 kitchen”；
4. “避开当前 blocked 区域”；
5. “重新搜索上次低置信度区域”；
6. “根据 target belief 选择下一个房间”。

---

## 14. 与 LingBot-Map 的关系

本方案继承 LingBot-Map 的部分：

| LingBot-Map | 本方案中的作用 |
|---|---|
| DINOv2 backbone | 作为 P0 视觉编码器 |
| camera / register / scale tokens | 作为几何特殊 token 基础 |
| GCA 三层上下文 | 作为流式时序记忆核心 |
| anchor frames | 用于地图坐标和尺度锚定 |
| sliding window | 用于局部几何和短期实例稳定 |
| trajectory memory | 扩展为几何 + 语义 + VLN 历史摘要 |
| cache / memory 思想 | 方案层面体现为短期详细、长期压缩 |

主要扩展：

1. 从纯几何扩展到几何 + 语义；
2. 增加 detection 和 segmentation heads；
3. 增加 map tokens；
4. 输出不只是 pose / depth，而是直接服务 semantic map；
5. 引入 VLN-facing map interface。

---

## 15. 与 VGGT / VGGT-Omega 的关系

本方案继承 VGGT / VGGT-Omega 的部分：

| VGGT 思想 | 本方案中的作用 |
|---|---|
| DINO tokenizer | 视觉基础特征 |
| alternating attention | 帧内 + 跨帧交互思想 |
| camera head | 位姿预测 |
| DPT dense head | 深度 / point map |
| over-complete prediction | pose / depth / point map 相互约束 |
| 多任务联合训练 | 几何任务共同提升 |
| feed-forward geometry | 尽量减少对传统后端优化依赖 |

主要扩展：

1. 从多视图几何预测扩展为流式在线建图；
2. 从相机 / 深度 / 点云扩展到检测 / 分割 / 语义实例；
3. 从图像集合输出扩展到长期地图状态；
4. 从几何 reconstruction 扩展到 VLN navigation memory。

---

## 16. 推荐阶段路线

### 16.1 Phase 0：方案验证版

目标：验证统一模型输出能否稳定构建语义地图。

配置：

```text
DINOv2 ViT-L
LingBot-style GCA
Pose Head
Depth Head
Detection Head
Segmentation Head
Basic Semantic Map
```

重点验证：

1. pose 是否足够稳定；
2. depth 是否可用于障碍地图；
3. mask 投影后的 object landmark 是否稳定；
4. object DB 是否能支撑 VLN 查询。

### 16.2 Phase 1：语义地图增强版

目标：提升开放词汇、动态更新和拓扑表达。

增加：

```text
Open-vocabulary detection adapter
Map Token Head
Scene change detection
Object lifecycle management
Topological graph construction
Target belief update
```

重点验证：

1. 指令目标词能否匹配地图对象；
2. 家具/物体移动后地图能否更新；
3. 拓扑图是否能帮助 VLN 规划；
4. 搜索失败时能否恢复。

### 16.3 Phase 2：DINOv3 / Teacher-Student 增强版

目标：提升 dense feature 与部署效率。

增加：

```text
DINOv3 backbone ablation
DINOv3 teacher distillation
Lightweight student backbone
Edge deployment profile
```

重点验证：

1. DINOv3 是否提升 mask / depth / semantic consistency；
2. DINOv3 特征是否有利于地图级稳定性；
3. 蒸馏后是否保留足够导航性能；
4. 端侧实时性是否满足机器人需求。

---

## 17. 评估指标

### 17.1 单帧感知指标

```text
Detection: mAP, recall, small-object recall
Segmentation: mask mIoU, instance AP
Depth: AbsRel, RMSE, δ1
Pose: ATE, RPE translation, RPE rotation
```

### 17.2 地图指标

```text
occupancy IoU
free-space precision / recall
object localization error
object duplicate rate
object missing false rate
semantic map consistency
long-term drift
```

### 17.3 VLN 指标

```text
success rate
SPL
target found rate
search recovery success
wrong-room rate
object query accuracy
frontier selection quality
planning latency
```

### 17.4 系统指标

```text
pose / depth FPS
semantic FPS
map update latency
memory growth rate
keyframe count
model compute cost
```

---

## 18. 关键风险

### 18.1 RGB-only 深度尺度不稳定

影响：

```text
地图尺度漂移
物体位置错误
VLN 距离判断错误
```

缓解方向：

```text
anchor frames
scale token
相机高度约束
地面平面约束
轮速 / IMU / 里程计弱融合
```

### 18.2 检测/分割误差累积进地图

影响：

```text
虚假物体
重复实例
目标误定位
错误搜索记忆
```

缓解方向：

```text
object lifecycle
multi-view confirmation
confidence decay
visibility-aware missing update
keyframe-level semantic update
```

### 18.3 DINO 特征缺少语言对齐

影响：

```text
开放词汇目标识别不足
VLN 指令 grounding 受限
```

缓解方向：

```text
固定类别 P0
开放词汇 adapter P1
Grounding-DINO / VLM teacher distillation
CLIP / text prototype alignment
```

### 18.4 多任务互相干扰

影响：

```text
pose 变差
depth 边界变糊
mask 质量下降
```

缓解方向：

```text
分阶段训练
任务 head 解耦
loss 动态权重
backbone 冻结 / 半冻结
geometry 与 semantic 分频输出
```

### 18.5 语义拓扑抽象不稳定

影响：

```text
房间划分错误
frontier 错误
VLN 规划不稳定
```

缓解方向：

```text
先 metric + object map
再逐步引入 room / topology
使用轨迹和门洞作为拓扑锚
VLN 反馈修正 target belief
```

---

## 19. 最终推荐架构

推荐的最终方案如下：

```text
DINO-GeoSemMap

Input:
    Streaming RGB

Backbone:
    DINOv2 ViT-L for P0
    DINOv3 as P1 / P2 teacher or enhanced backbone

Temporal Core:
    LingBot-style GCA
    Anchor Context
    Sliding Window
    Trajectory Memory

Special Tokens:
    camera token
    register tokens
    scale token
    semantic token
    object token
    free-space token
    change token

Heads:
    Pose Head
    Depth Head
    Detection Head
    Segmentation Head
    Map Token Head

Map:
    Metric occupancy / free-space map
    Semantic instance object map
    Topological region graph
    Trajectory / search memory
    Target belief state

VLN Interface:
    object query
    region query
    frontier query
    target belief query
    local map query
    topological graph query
```

一句话总结：

> DINO-GeoSemMap 以 DINOv2 / DINOv3 为统一视觉特征基座，以 LingBot-Map 的流式三层几何记忆为时序核心，以 VGGT / VGGT-Omega 的多任务几何预测为 pose / depth 基础，再叠加检测、分割和 map token 输出，最终把连续 RGB 输入转化为 VLN 可直接查询和推理的动态语义场景地图。

