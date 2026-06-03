# DINO-GeoSemMap v0.2 模型架构

> 纯架构文档，聚焦模型结构、数据流和计算规格  
> 版本：v0.2.1  
> 日期：2026-05-21  
> 详细方案文档：[[02_DINO-GeoSemMap统一单模型架构]]  
> 拓扑系统参考：[[语义与VLM房间分区技术规划]] §6

---

## 1. 架构总览

DINO-GeoSemMap v0.2 是一个流式多任务视觉模型，以 DINOv2 ViT-L 为唯一 backbone，以 LingBot-Map 的 24 组交替注意力（Frame Attention + GCA）为模型主体，通过扩展 token 序列实现几何与语义的统一处理。

模型专注于**感知**（pose / depth / detection / segmentation / room type），地图构建由下游工程拓扑模块完成。

核心设计：

- **单一 backbone**：DINOv2 ViT-L（frozen / LoRA）
- **单一注意力主体**：48 个 Transformer Block（24 组 Frame + GCA 交替）
- **单一 KV-cache**：分页三级缓存（Anchor / Window / Trajectory Memory）
- **4 个 Readout Head**：Camera / DPT / Object / Room

```text
RGB_t
  │
  ▼
DINOv2 ViT-L Backbone (frozen / LoRA)
  │
  │  构建 token 序列:
  │  [camera, scene_reg×16, scale, OBJ×50, patch×M]
  ▼
24 组交替注意力层（共 48 Block）
  ┌──────────────────┐   ┌──────────────────────────────┐
  │ Frame Attention   │ → │ GCA: Full(×18) / RegAttn(×6) │  × 24 组
  │ 帧内全连接自注意力 │   │ 跨帧因果注意力                │
  │ + 2D RoPE        │   │ + 三级 KV-cache               │
  │                  │   │ + 3D Video RoPE               │
  └──────────────────┘   └──────────────────────────────┘
  在第 [4, 11, 17, 23] 组提取多尺度特征
  │
  ├── camera token ──→ Camera Head    → pose_t
  ├── scene_reg×16 ──→ Room Readout   → room_type_t
  ├── patch tokens ──→ DPT Head       → depth_t, uncertainty_t
  └── OBJ tokens ────→ Object Readout → boxes_t, masks_t, classes_t
              │
              ▼
      工程拓扑模块（L1-L4 语义记忆系统）
```

### 1.1 各阶段输入输出（中文描述版）

当前架构可以理解为一条从“相机 RGB 帧”到“可供拓扑系统和 VLN 使用的结构化观测”的流水线。模型本体负责把图像变成位姿、深度、物体、房间类型和场景 token；工程拓扑模块负责把这些观测沉淀成长期语义地图。

### 阶段一：接收当前 RGB 帧

**输入是什么**

- 当前相机看到的一帧 RGB 图像。
- 当前帧的时间戳。
- 当前帧在系统里的角色：是初始化阶段的锚定帧，还是后续普通流式帧。
- 当前帧是否需要做完整语义识别：关键帧做完整识别，非关键帧只做位姿和深度。

**输出是什么**

- 一张已经完成尺寸处理和归一化的图像，送入 DINOv2。
- 一个帧角色标记，用来决定特殊 token 使用“锚定帧版本”还是“普通帧版本”。
- 一个运行模式标记，用来决定后面是否运行物体检测、分割和房间类型识别。

**这一阶段的作用**

这一阶段不做视觉理解，只做输入整理和调度。它决定后续模型是按“初始化建坐标系”的方式处理，还是按“流式追踪当前帧”的方式处理。

---

### 阶段二：DINOv2 提取图像 patch 特征

**输入是什么**

- 阶段一输出的标准化 RGB 图像。

**输出是什么**

- 一组图像 patch 特征。可以理解为：模型把整张图切成很多小块，每个小块变成一个 1024 维视觉特征。
- 这些 patch 特征保留了图像局部的纹理、边缘、物体外观和隐式几何信息。

**这一阶段的作用**

DINOv2 是整个模型唯一的视觉 backbone。后续所有任务，包括位姿、深度、检测、分割和房间判断，都基于这一组 patch 特征展开，不再为不同任务重复跑多个视觉编码器。

---

### 阶段三：构建当前帧 token 序列

**输入是什么**

- DINOv2 输出的图像 patch 特征。
- camera token：用于后续预测当前帧位姿。
- scene registers：用于聚合场景级信息，并在跨帧注意力中传递全局上下文。
- scale token：用于区分锚定帧和普通帧，并承载尺度相关信息。
- OBJ queries：用于后续检测和分割潜在物体实例。

**输出是什么**

当前帧完整 token 序列：

```text
[camera, scene_reg×16, scale, OBJ×50, patch×M]
```

其中：

- `camera` 负责几何位姿；
- `scene_reg×16` 负责场景级表达；
- `scale` 负责尺度和帧类型；
- `OBJ×50` 负责潜在物体实例；
- `patch×M` 负责图像局部细节。

**这一阶段的作用**

这一阶段把“图像特征”和“任务查询 token”放到同一个序列里，让后面的 48 层交替注意力同时处理几何、语义和场景信息。

---

### 阶段四：帧内注意力，整合当前帧内部信息

**输入是什么**

- 当前帧完整 token 序列。

**输出是什么**

- 更新后的当前帧 token。此时 camera token、scene registers、OBJ queries 和 patch tokens 已经在当前帧内部充分交换信息。

**这一阶段的作用**

帧内注意力只看当前帧，不看历史帧。它负责让不同类型 token 在单帧内部互相理解：

- camera token 从图像 patch 中读取几何线索；
- OBJ queries 从 patch 中读取物体线索；
- scene registers 从整帧图像中聚合场景信息；
- patch tokens 也能从特殊 token 中获得任务上下文。

---

### 阶段五：跨帧注意力，融合历史几何和场景上下文

**输入是什么**

- 当前帧经过帧内注意力后的 tokens。
- 锚定帧缓存：初始化阶段保留下来的完整 token，用于提供坐标系和尺度参考。
- 滑动窗口缓存：最近若干帧的完整 token，用于局部几何匹配和短期稳定。
- 轨迹记忆缓存：历史所有关键 token 的压缩记忆，用于长期漂移抑制和场景记忆。

**输出是什么**

- 融合历史上下文后的当前帧 tokens。
- 更新后的 KV-cache，包括窗口缓存和轨迹记忆。

**这一阶段的作用**

这一阶段是 LingBot-Map 思路的核心。它让当前帧不是孤立预测，而是同时参考：

- 最开始建立坐标系的锚定帧；
- 最近几帧的局部视觉连续性；
- 更长历史中的相机轨迹和场景摘要。

其中大部分层使用 Full GCA，让所有 token 都可以跨帧交互；部分层使用 Register Attention，只让 scene registers 等少量特殊 token 跨帧交互，用更低成本传递全局信息。

---

### 阶段六：提取多尺度特征

**输入是什么**

- 交替注意力中第 4、11、17、23 组的中间 patch 特征。

**输出是什么**

- 四组不同深度的视觉特征。
- 浅层特征更偏纹理和边缘，中层特征更偏结构，深层特征更偏语义和全局几何。

**这一阶段的作用**

DPT Head 和 Object Readout 的 mask 预测都需要多尺度信息。深度估计需要局部边界和全局结构，分割也需要同时知道物体细节和语义上下文。

---

### 阶段七：多任务 Readout 输出当前帧感知结果

这一阶段把前面得到的 token 和多尺度特征，读出为具体任务结果。

| Readout | 读取什么 | 输出什么 | 运行频率 |
|---|---|---|---|
| Camera Head | camera token | 当前帧位姿、旋转、平移、相机内参、位姿置信度 | 高频，每帧运行 |
| DPT Head | 多尺度 patch 特征 | 当前帧深度图、深度不确定性、可选点云图 | 高频，每帧运行 |
| Object Readout | OBJ queries + 多尺度 patch 特征 | 物体框、类别、置信度、实例 mask | 低频，关键帧运行 |
| Room Readout | scene registers | 房间类型概率、房间判断置信度 | 低频，关键帧运行 |

**关键帧输出**

关键帧会输出完整感知结果：

```text
位姿
深度
深度不确定性
可选点云图
物体框
物体类别
物体置信度
实例 mask
房间类型
房间置信度
scene registers
```

**非关键帧输出**

非关键帧只输出高频几何结果：

```text
位姿
深度
深度不确定性
可选点云图
scene registers
```

说明：

- 非关键帧仍然会让 OBJ tokens 参与主干注意力，但默认不运行 Object Readout。
- 这样做的目的是让语义 token 持续吸收上下文，但不在每帧都付出检测和分割的计算成本。
- scene registers 每帧都可以输出，既能写入轨迹记忆，也能作为 VLN/VLA 的空间 token。

---

### 阶段八：模型输出交给工程拓扑模块

**输入是什么**

- 模型输出的位姿和深度。
- 关键帧上的物体框、类别、置信度和实例 mask。
- 关键帧上的房间类型和房间置信度。
- 每帧 scene registers。

**输出是什么**

- 3D 物体观测：某个物体在世界坐标中的位置、大小、类别和置信度。
- 房间观测：当前位置附近可能属于哪个房间类型。
- 空间切换事件：机器人可能从一个房间进入另一个房间。
- 局部可通行或障碍 evidence：由深度和位姿投影得到。

**这一阶段的作用**

模型本体不直接维护长期语义地图。它只提供当前帧或关键帧的结构化观测。工程拓扑模块负责把这些观测转成 L2 动态事件。

---

### 阶段九：L2 动态事件沉淀到 L1 长期语义结构

**输入是什么**

- 多帧连续的 L2 观测事件。
- 物体在不同帧中的 3D 位置。
- 房间类型的多次预测结果。
- 位姿邻近关系。
- 语义一致性和 evidence_count。

**输出是什么**

- Room Node：稳定房间节点，包括房间类型、空间范围和邻接关系。
- Zone Node：房间内部区域，如沙发区、餐桌区、床边区域。
- Anchor Object：大型稳定家具，如床、沙发、桌子、柜子。
- Movable Object：可移动物品，如杯子、遥控器、书包等。
- Room neighbors：房间之间的连通关系。

**这一阶段的作用**

L2 是短期观测，L1 是长期记忆。系统不会因为单帧看到一个物体或单帧判断一个房间，就立刻写入长期地图，而是要等多次观测一致后再沉淀。

这样可以避免：

- 单帧误检污染长期地图；
- 房间类型短时抖动导致错误建图；
- 物体临时遮挡或移动造成错误删除；
- 位姿漂移导致语义节点重复。

---

### 阶段十：提供给 VLN/VLA 使用

**输入是什么**

- L1-L4 语义记忆系统中的稳定地图结果。
- 当前帧的位姿、可见物体和房间上下文。
- 可选的 scene registers 空间 token。

**输出是什么**

- 当前所在房间或区域。
- 当前可见物体列表。
- 目标物体可能所在区域排序。
- 已搜索区域和未搜索区域。
- 房间之间的拓扑关系。
- 可供 VLN/VLA 模型使用的空间 token。

**这一阶段的作用**

VLN/VLA 不直接消费原始深度图、mask 和所有 patch tokens，而是消费整理后的语义地图、拓扑结构和必要的空间 token。这样输入更稳定、可解释，也更适合长程导航决策。

---

## 2. Backbone：DINOv2 ViT-L

### 2.1 规格

| 项目 | 值 |
|---|---|
| 模型 | DINOv2 ViT-Large |
| Patch Size | 14×14 |
| Hidden Dim | 1024 |
| Attention Heads | 16 |
| 权重状态 | Frozen（P0）/ LoRA 最后 4 层（P1+） |
| 参数量 | ~307M |

### 2.2 输入输出

```text
输入:
  RGB image I_t ∈ R^{H×W×3}

输出:
  patch_tokens ∈ R^{M × 1024}
  其中 M = (H/14) × (W/14)
  例如 518×294 → 37×21 = 777 个 patch tokens
```

### 2.3 多尺度特征

不通过额外 FPN，而是从交替注意力的第 [4, 11, 17, 23] 组直接提取中间层输出：

```text
每组输出: frame_feat (1024d) ⊕ global_feat (1024d) = 2048d
4 个尺度拼接后用于 DPT Head 和 Object Readout 的 mask 预测
```

---

## 3. Token 序列

每帧构建以下 token 序列，所有 tokens 一起参与 48 层交替注意力：

```text
┌────────┬───────────────────────────┬───────┬──────────────────┬────────────────┐
│ camera │ scene_reg_0 ... _15       │ scale │ OBJ_0 ... OBJ_49 │ patch_0 ... _M │
│  ×1    │         ×16               │  ×1   │       ×50        │      ×M        │
└────────┴───────────────────────────┴───────┴──────────────────┴────────────────┘
  ←──── 18 个几何/场景特殊 tokens ──→  ← 50 个语义 tokens →  ←── M patch tokens ──→
```

### 3.1 Token 定义

| Token | 数量 | 初始化 | 功能 |
|---|---|---|---|
| camera | 1 | `nn.Parameter`, std=1e-6, 双变体 | 聚合帧级几何信息，输入 Camera Head 做位姿预测 |
| scene registers | 16 | `nn.Parameter`, std=1e-6, 双变体 | 注意力稳定 + 场景级空间表征 + Register Attention 跨帧信息载体 + Room Readout 输入 |
| scale | 1 | `nn.Parameter`, std=1e-6, 双变体 | 区分锚定帧/普通帧，承载尺度信息 |
| OBJ queries | 50 | `nn.Parameter`, std=1e-6, 双变体 | 每个 query 检测一个潜在实例，输出 box + class + mask |
| patch tokens | M | DINOv2 编码输出 | 局部视觉、语义、几何 dense 特征 |

### 3.2 双变体机制

所有特殊 tokens（camera / scene_reg / scale / OBJ）均有两个变体：

- **scale frame 变体**：用于锚定帧（任务初始化阶段的关键帧）
- **normal frame 变体**：用于后续流式帧

通过 `nn.Parameter` 学习各自初始化，推理时根据帧类型选择。

### 3.3 Token 数量

```text
每帧 token 总量:
  特殊 tokens:  1 + 16 + 1 = 18
  语义 tokens:  50
  patch tokens: ~500（取决于分辨率）
  ────────────────────
  总计:         ~568 tokens/帧
```

---

## 4. 交替注意力（48 Block）

模型主体是 24 组交替注意力层，每组包含 1 个 Frame Attention Block + 1 个 GCA Block，共 48 个 Transformer Block。

```text
Block 0 (Frame) → Block 1 (GCA) → Block 2 (Frame) → Block 3 (GCA) → ... → Block 46 (Frame) → Block 47 (GCA)
  组 0                               组 1                                      组 23
```

### 4.1 Frame Attention

每个 Frame Attention Block 在当前帧内部执行全连接自注意力，所有 tokens 参与：

```text
帧 t 的全部 ~568 tokens:
[camera, scene_reg×16, scale, OBJ×50, patch×M]
          ↕ 全连接自注意力（所有 token 互相关注）↕
```

位置编码：2D RoPE

```text
camera:        (0, 0)
scene_reg_0:   (1, 1)
...
scene_reg_15:  (16, 16)
scale:         (17, 17)
OBJ_0:         (18, 18)
...
OBJ_49:        (67, 67)
patch_0:       (patch_start + row, patch_start + col)
```

### 4.2 Full GCA Block（18 组）

在 24 组中的 18 组，GCA 执行全量跨帧因果注意力——所有 tokens 参与：

```text
当前帧的所有 tokens (Q)
    attend to:
    ├── Anchor KV:     锚定帧全量 tokens（永不驱逐）
    ├── Window KV:     最近 k 帧全量 tokens（FIFO 回收）
    └── Trajectory KV: 历史帧 special tokens（只追加）
```

位置编码：3D Video RoPE（帧间时序 + 帧内空间）

### 4.3 Register Attention Block（6 组）

在第 [6, 12, 18, 23] 等位置（共 6 组，占 25%），GCA 替换为 Register Attention——仅 registers 跨帧交互：

```text
scene registers (Q) → attend to 历史帧 registers (KV)       ✓ 跨帧
camera / scale  (Q) → attend to 历史帧 camera/scale (KV)     ✓ 跨帧
OBJ tokens      (Q) → 不参与跨帧 attention                   ✗
patch tokens    (Q) → 不参与跨帧 attention                   ✗
```

信息流：registers 在跨帧注意力中聚合全局信息 → 下一个 Frame Attention 中分发给本帧所有 tokens。

计算收益（来自 VGGT-Omega 消融）：

| 配置 | Point Error | FLOPs | 显存 |
|---|---|---|---|
| 100% Full GCA | 0.071 | 100% | 100% |
| 75% Full + 25% Register Attention | 0.073 | -23% | -16% |

### 4.4 三级分页 KV-cache

```text
Patch Page Pool（可回收）:
  ├── Scale frame pages:  锚定帧 patch KV，永不驱逐
  ├── Window pages:       最近 k 帧 patch KV，FIFO 回收
  └── Free pages:         空闲页池

Special Token Page Pool（只追加）:
  └── 所有历史帧的 special token KV（camera + scene_reg×16 + scale + Top-K OBJ）
      永不驱逐 → 轨迹记忆
```

轨迹记忆写入策略（每帧）：

| Token | 写入 | 数量 |
|---|---|---|
| camera | 始终写入 | 1 |
| scene registers | 始终写入 | 16 |
| scale | 始终写入 | 1 |
| OBJ queries | Top-K 高置信度 | ~5 |
| **每帧总量** | | **~23** |

Full GCA blocks 访问全部 pages；Register Attention blocks 只访问 special token pages（跳过 patch pages）。

### 4.5 多尺度特征提取

在第 [4, 11, 17, 23] 组（对应 Block 8, 22, 34, 46）提取 patch tokens 的中间输出：

```text
每组: frame_feat_l (1024d) ⊕ gca_feat_l (1024d) = 2048d
输出: {F^4, F^11, F^17, F^23}，用于 DPT Head 和 Object Readout mask 预测
```

---

## 5. Readout Heads

### 5.1 Camera Head

沿用 LingBot-Map，不做修改。

```text
输入: camera token ∈ R^{1024}（经 48 层交替注意力后）

结构:
  4 次迭代精炼:
    embed(current_pose) → AdaLN 调制 camera token
    → 4 层因果 Transformer Block（独立 KV-cache，3D Video RoPE）
    → MLP → Δpose
    → pose += Δpose

输出: 9D 位姿编码 [T_x, T_y, T_z, q_i, q_j, q_k, q_w, fov_h, fov_w]
  → R, t（外参）
  → K（内参，from FoV）
  → confidence

参数量: ~25M
```

### 5.2 DPT Head

沿用 LingBot-Map，不做修改。

```text
输入: 第 [4, 11, 17, 23] 组的多尺度 patch features（2048d × 4 层）

结构:
  4 个尺度的 project + refinenet
  自底向上融合
  最终卷积 → [depth/xyz, confidence]

输出:
  dense depth map ∈ R^{H×W}
  depth uncertainty ∈ R^{H×W}
  可选: 3D point map ∈ R^{H×W×3} + confidence

参数量: ~10M
```

### 5.3 Object Readout（新增）

```text
输入:
  OBJ tokens ∈ R^{50 × 1024}（经 48 层交替注意力后）
  多尺度 patch features（用于 mask 预测）

结构:
  Classification:  OBJ token → Linear(1024, num_classes) → class logits
  Box Regression:  OBJ token → MLP(1024→512→4) → normalized box [cx, cy, w, h]
  Objectness:      OBJ token → Linear(1024, 1) → objectness score
  Mask Prediction: OBJ token × multi-scale features → dot product → mask logits
                   （Mask2Former 风格，pixel-level 点积）

输出:
  boxes_t     ∈ R^{50 × 4}
  classes_t   ∈ R^{50 × num_classes}
  scores_t    ∈ R^{50}
  masks_t     ∈ R^{50 × H × W}

参数量: ~2M
```

### 5.4 Room Readout（新增，替代 Map Readout）

从 scene registers 读出房间类型预测，不需要额外 token。

```text
输入: scene registers ∈ R^{16 × 1024}（经 48 层交替注意力后）

结构:
  mean_pool(16 × 1024) → 1024d
  → MLP(1024 → 256 → num_room_types) → room_type_logits
  → Linear(256 → 1) → room_confidence

房间类别（对齐语义与VLM房间分区技术规划）:
  living_room / kitchen / corridor / bedroom / bathroom
  / study / balcony / dining_room / other

输出:
  room_type_t      ∈ R^{num_room_types}    房间类型概率分布
  room_confidence_t ∈ R^{1}                 预测置信度

参数量: ~0.3M
```

Room Readout 从 registers 读出的优势：

1. **自带时序上下文**：registers 在 Register Attention blocks 中是唯一跨帧交互的 token，房间判断不是基于单帧而是融合了近期多帧累积的场景理解
2. **零额外 token**：不像 MAP queries 需要在序列中占位置，registers 本身已存在
3. **与 VLN/VLA 复用**：registers 同时可作为下游决策模型的空间 token 输入

---

## 6. 模型输出

每个时间步 `t` 的完整输出：

```text
Y_t = {
    geometry: {
        pose_t               ← Camera Head
        depth_t              ← DPT Head
        depth_uncertainty_t  ← DPT Head
        optional_pointmap_t  ← DPT Head
    },

    semantics: {
        boxes_t              ← Object Readout
        classes_t            ← Object Readout
        scores_t             ← Object Readout
        masks_t              ← Object Readout
    },

    scene: {
        room_type_t          ← Room Readout (from scene registers)
        room_confidence_t    ← Room Readout (from scene registers)
    },

    representations: {
        scene_registers_t    ← 16 × 1024d 场景级空间 token（可供 VLN/VLA 直接消费）
    }
}
```

---

## 7. 模型输出 → 工程拓扑模块

模型专注感知，地图构建由下游工程模块完成。模型输出与 L1-L4 语义记忆系统的对接关系：

### 7.1 数据流

```text
模型输出 Y_t
  │
  ├── pose_t + depth_t ──────────────→ 3D 投影
  │                                     │
  ├── masks_t + classes_t + scores_t ─→ 3D Object Observations
  │     + depth_t + pose_t              │
  │                                     ▼
  ├── room_type_t + room_confidence_t → L2 动态事件层
  │     + pose_t                        ├── 观测事件（可见物品 + 房间类型）
  │                                     ├── 对象状态事件（物品位置更新）
  │                                     └── 空间切换事件（房间转换检测）
  │                                         │
  │                                         ▼ 累积证据沉淀
  │                                     L1 语义结构层
  │                                     ├── Room Node（房间类型、邻接关系）
  │                                     ├── Zone Node（区域、锚点对象）
  │                                     ├── Anchor Object Node（大型家具）
  │                                     └── Movable Object Node（可移动物品）
  │                                         │
  │                                         ▼ 按需组装
  │                                     L3 记忆服务层
  │                                     ├── 找物查询（目标物品 → 优先搜索区域）
  │                                     ├── 找人查询（目标人物 → 活动区域排序）
  │                                     └── 异常检测（当次观测 vs L1 基线）
  │                                         │
  │                                         ▼ 驱动生成
  │                                     L4 用户展示层
  │                                     ├── 房间总览视图
  │                                     ├── 语义布局草图
  │                                     └── 房间详情视图
  │
  └── scene_registers_t ─────────────→ VLN/VLA 决策模型（可选）
```

### 7.2 模型各输出的拓扑消费方式

| 模型输出 | L2 事件类型 | L1 沉淀目标 | 说明 |
|---|---|---|---|
| room_type_t + pose_t | 观测事件 / 空间切换事件 | Room Node | 位姿邻近性 + 语义相似性判断归属房间；位姿跳变 + 房间类型变化触发空间切换 |
| boxes_t + classes_t + masks_t + depth_t + pose_t | 观测事件（可见物品） | Zone Node / Anchor Object / Movable Object | 3D 投影后关联到房间内区域，大型家具沉淀为 Anchor，小物品累积 evidence_count |
| room_confidence_t | — | Room Node（确认条件） | 低置信度观测不沉淀，高置信度多次累积才写入 L1 |
| depth_t + pose_t | — | （可选）Metric Layer | 如需局部避障或可通行性判断，投影为 occupancy grid |

### 7.3 关键帧触发（工程规则）

不再依赖模型学习 keyframe_score，改用确定性规则：

```text
触发条件（满足任一即触发）:
  1. 位移超过阈值（如 0.5m）
  2. 旋转超过阈值（如 30°）
  3. room_type_t 类别发生切换
  4. room_confidence_t 从低变高（进入新区域稳定后）
  5. VLN 模块主动请求

关键帧时运行:
  全部 4 个 Readout Heads

非关键帧时运行:
  仅 Camera Head + DPT Head（pose + depth）
  OBJ tokens 仍参与注意力（持续积累），但不运行 Readout
```

### 7.4 L2 → L1 沉淀规则

直接复用[[语义与VLM房间分区技术规划]] §6.4 的设计：

- **房间节点沉淀**：同一位姿邻域内 room_type 累计 N 次一致 → 创建/确认 L1 Room Node
- **物品记忆沉淀**：某物品在某区域观测 evidence_count 超过阈值 → 写入 Zone Node 的 remembered_objects
- **可移动物品位置更新**：物品状态事件在某房间累计 N 次 → 更新 Movable Object 的 habitual_locations
- **邻接关系强化**：空间切换事件重复出现 → 强化 L1 Room Node 间 neighbors 关系
- **遗忘衰减**：L1 可移动物品长期无 L2 证据支撑 → evidence_count 自然下降

---

## 8. 训练损失

```text
L_total = λ_pose  · L_pose
        + λ_depth · L_depth
        + λ_det   · L_detection
        + λ_seg   · L_segmentation
        + λ_room  · L_room
        + λ_point · L_point
        + λ_match · L_matching
```

### 8.1 几何损失（沿用 LingBot-Map）

```text
L_pose  = absolute_pose_loss + relative_pose_loss
L_depth = weighted_L1(D_hat, D) + gradient_L1(D_hat, D) - α·log(σ_D)
```

### 8.2 语义损失（新增）

```text
L_detection = Hungarian_matching_loss (DETR-style)
L_segmentation = mask_BCE + dice_loss
L_room = CrossEntropy(room_type_pred, room_type_gt)
```

房间类型标注来源：ScanNet / HM3D / Habitat 等室内数据集自带房间标签，或复用 VLM 已验证的房间类型判断结果做弱监督。

### 8.3 辅助几何一致性损失（from VGGT-Omega，仅训练期）

```text
L_point = weighted_L1(P_hat, P) + gradient_L1(P_hat, P) - α·log(σ_P)
  其中 P_hat = unproject(depth_t, pose_t)
  强制 depth 和 pose 在 3D 空间中一致
  推理时零开销（不需要独立 head）

L_matching = BCE(cos_sim(token_i, token_j), is_match)
  正样本: 3D 投影重叠的 patch 对
  负样本: 满足几何约束 + 外观差异
  鼓励最终 patch features 包含可匹配几何信息
  推理时零开销（从最后一层 patch tokens 采样计算）
```

### 8.4 各推理头需要什么训练数据

下面把每个推理头对应的数据类型单独整理出来。这里的思路是：**位姿和深度头吃几何序列数据，检测和分割头吃像素级语义标注数据，房间头吃室内区域级标注数据**。如果希望一个模型同时学会这些能力，最稳妥的方式是“共享视频骨架 + 头部专属监督”的混合数据方案。

| 推理头 | 主要监督信号 | 最适合的数据形态 | 典型数据来源 | 备注 |
|---|---|---|---|---|
| Camera / Pose Head | 绝对位姿、相对位姿、时间顺序、相机内参（可选） | 连续 RGB 序列、多视角序列、机器人轨迹日志 | ScanNet、HM3D、Matterport3D、ARKitScenes、Replica、TUM RGB-D、EuRoC、KITTI、nuScenes、真实机器人日志 | 这类数据最好带锚定帧和连续轨迹，方便训练流式 KV-cache 和相对位姿预测 |
| DPT / Depth Head | Dense depth、valid mask、uncertainty（可选） | RGB-D 序列、LiDAR 投影深度、渲染深度、双目/多视图伪深度 | ScanNet、Replica、Matterport3D、HM3D、ARKitScenes、KITTI、nuScenes、Waymo、3DGS 渲染数据 | 可以混合真值深度和伪深度；训练时最好带稠密有效区域 mask |
| Object Readout / Detection Head | Bounding box、类别、置信度、遮挡/截断属性（可选） | 单帧检测标注、关键帧检测标注、多视角投影检测标注 | COCO、Objects365、LVIS、OpenImages、ScanNet 投影框、HM3D 投影框、InteriorGS 自建标注 | 更适合关键帧训练；如果面向导航，建议加入 door、chair、table、sink、elevator、vending machine、storefront 等常见目标 |
| Object Readout / Segmentation Head | Semantic mask、instance mask、panoptic mask、stuff/thing 标签 | 像素级语义/实例分割、视频 mask propagation、合成渲染 mask | ADE20K、COCO Panoptic、Cityscapes、Mapillary、ScanNet、HM3D、Replica、S3DIS、InteriorGS | 如果最终要服务地图构建，建议额外加入 floor、wall、walkable area、road、crosswalk、doorway 这些导航相关类别 |
| Room Readout / Room Type Head | Room type、zone type、邻接关系、置信度 | 室内区域级标注、房间图、楼层/功能区标签、弱监督房间分类 | ScanNet、HM3D、Matterport3D、Structured3D、ARKitScenes、Habitat、VLM 弱标签 | 这类监督基本只在室内有效；可按 kitchen、bedroom、living room、corridor、bathroom、office、conference room、hallway 等粒度组织 |

#### 建议的数据组织方式

1. **几何主干序列**
   - 一条连续 RGB 序列，带 pose / depth / timestamp；
   - 主要喂给 Camera Head 和 DPT Head；
   - 适合做流式建图的基础训练集。

2. **关键帧语义样本**
   - 从几何序列中抽取关键帧；
   - 叠加 detection / segmentation / room type 标注；
   - 主要喂给 Object Readout 和 Room Readout。

3. **多任务共享样本**
   - 同一批场景同时带 pose、depth、box、mask、room label；
   - 用于统一训练，减少 head 之间的表征割裂；
   - 最适合室内 3D 场景数据，例如 ScanNet、HM3D、ARKitScenes、InteriorGS。

4. **弱监督补充样本**
   - 用 SfM、SLAM、LiDAR 投影、VLM 伪标签补足缺标注场景；
   - 适合扩展室外、长尾场景和开放词表类别；
   - 重点补 pose / depth / room / open-vocab detection 的长尾分布。

#### 一个更实用的训练分配

如果按“先让模型学会走，再让它学会看懂，再让它学会总结成地图”的顺序来排，推荐这样分：

| 阶段 | 训练重点 | 主要数据 |
|---|---|---|
| 阶段 A | 位姿 + 深度 | 连续视频、RGB-D、LiDAR 投影深度、SfM / SLAM 伪标签 |
| 阶段 B | 检测 + 分割 | 单帧/关键帧语义标注、室内家具和户外目标的像素级标注 |
| 阶段 C | 房间类型 + 语义记忆 | 室内房间标签、区域图、VLM 弱监督、连续场景切换序列 |

这套分法的好处是：先把几何骨架学稳，再叠加语义，再把语义压成稳定的房间和区域记忆，最后才交给 VLN / 拓扑系统去消费。

---

## 9. 推理流程

### 9.1 Phase 1：锚定帧处理（双向注意力）

```text
scale_images (前 n 帧)
  → DINOv2 编码
  → 每帧构建完整 token 序列 [camera, scene_reg×16, scale, OBJ×50, patches]
  → forward（双向注意力，所有 scale 帧作为一个 block）
  → 初始化 Anchor KV-cache（含全部 token 的 KV）
  → 输出 anchor 帧的 pose / depth / detection / segmentation / room_type
```

### 9.2 Phase 2：流式推理（逐帧因果）

```text
for each frame t:
  frame_t → DINOv2 编码
  → 构建 token 序列 [camera, scene_reg×16, scale, OBJ×50, patches]
  → forward（因果注意力，使用三级 KV-cache）
  → Camera Head    → pose_t
  → DPT Head       → depth_t
  → Room Readout   → room_type_t                    (关键帧运行)
  → Object Readout → boxes_t, masks_t, classes_t     (关键帧运行)
  → 更新 KV-cache:
      Window: FIFO 回收旧帧 patch pages
      Trajectory: 追加当前帧 ~23 special tokens
  → 输出 Y_t → 工程拓扑模块
```

### 9.3 多频率运行

| 输出 | 频率 | 运行内容 |
|---|---|---|
| pose + depth | 10-20 Hz | 所有 tokens 参与注意力，仅运行 Camera Head + DPT Head |
| detection + segmentation + room_type | 1-3 Hz | 关键帧时运行 Object Readout + Room Readout |

高频帧中 OBJ tokens 仍参与注意力（持续积累信息），但不运行 Readout。

---

## 10. 参数与计算量估算

### 10.1 参数量

| 组件 | 参数量 | 来源 |
|---|---|---|
| DINOv2 ViT-L | ~307M | Frozen |
| 48 层交替注意力 | ~400M | LingBot-Map 预训练 |
| Camera Head | ~25M | LingBot-Map 预训练 |
| DPT Head | ~10M | LingBot-Map 预训练 |
| 特殊 token embeddings | ~0.14M | 新增 (18+50 tokens × 1024d × 2 变体) |
| Object Readout | ~2M | 新增 |
| Room Readout | ~0.3M | 新增 |
| **总计** | **~744M** | |
| **新增参数** | **~2.44M** | **占总量 ~0.33%** |

### 10.2 每帧 token 计算量

```text
Frame Attention:
  ~568 tokens × 568 tokens × 1024d × 16 heads
  = 每帧每 Frame Block 约 0.33G FLOPs

Full GCA Block (18 组):
  Query: ~568 tokens
  KV:    anchor patches + window patches + trajectory specials
         ≈ 5000 + 23T tokens (T = 历史帧数)

Register Attention Block (6 组):
  Query: ~18 (special) tokens
  KV:    trajectory specials ≈ 18T tokens
  节省: 跳过 patch-to-patch 跨帧计算（~550 × 5000+ 省略）
```

### 10.3 KV-cache 显存估算

```text
Anchor KV:  n_anchor × 568 tokens × 1024d × 2 (K+V) × 2 bytes
            2 帧 × 568 × 1024 × 4 ≈ 4.7 MB

Window KV:  k_window × 568 × 1024 × 4
            8 帧 × 568 × 1024 × 4 ≈ 18.7 MB

Trajectory KV (per 10000 frames):
            10000 × 23 × 1024 × 4 ≈ 0.94 GB

总计 (10000 帧): ~0.96 GB
```

### 10.4 训练到“比较好效果”大致需要多少数据和时间

下面这组数字是**工程估算**，不是论文实测值。假设条件是：

- backbone 使用已经预训练好的 DINOv2 / LingBot-Map 权重，通常冻结或只做 LoRA 微调；
- 训练硬件约等于 **8×A100 80GB** 或同级别算力；
- 输入分辨率在 **224 到 384** 之间；
- 数据已经完成清洗、对齐和格式转换；
- “比较好效果”指的是：模型已经能稳定用于流式建图和下游 VLN，不追求绝对 SOTA。

如果从零训练，这些时间通常还要再乘 **3 到 5 倍**。如果只有单机 1 到 2 张消费级 GPU，时间也可以近似按算力线性放大。

| 头部 / 任务               | 起步可用的数据量                    | 比较好效果的数据量                   | 大致训练时长              |
| --------------------- | --------------------------- | --------------------------- | ------------------- |
| Camera / Pose Head    | 20 万到 50 万帧，或 1 千到 5 千段连续轨迹 | 100 万到 300 万帧，或 1 万到 3 万段轨迹 | 2 到 5 天起效，1 到 2 周更稳 |
| DPT / Depth Head      | 50 万到 100 万帧 RGB-D / 伪深度    | 200 万到 500 万帧               | 3 到 7 天起效，1 到 2 周更稳 |
| Object Detection Head | 10 万到 30 万张带框图像             | 50 万到 100 万张带框图像            | 1 到 3 天起效，3 到 7 天更稳 |
| Segmentation Head     | 5 万到 20 万张带 mask 图像         | 30 万到 100 万张带 mask 图像       | 2 到 5 天起效，1 到 2 周更稳 |
| Room Type Head        | 2 万到 5 万张房间级标注关键帧           | 10 万到 30 万张房间级标注关键帧         | 1 到 2 天起效，3 到 7 天更稳 |

#### 一个更实用的整体训练节奏

如果目标是把整套模型训练到“能用于流式语义地图构建”的程度，比较合理的顺序是：

1. **先做几何底座**
   - 数据量：100 万到 300 万帧连续序列
   - 目标：先把 pose 和 depth 训稳
   - 用时：大约 3 到 7 天

2. **再补语义头**
   - 数据量：20 万到 100 万张关键帧，带检测 / 分割 / 房间标注
   - 目标：让物体、房间、可通行区域能被稳定读出
   - 用时：大约 3 到 10 天

3. **最后做联训和时序稳定**
   - 数据量：100 万到 500 万帧的混合流式序列
   - 目标：让检测、分割、深度、位姿在时间上不抖，并且能稳定写入语义地图
   - 用时：大约 1 到 2 周

#### 如果按最终系统目标来估

- **只想做出一个可工作的原型**：大约 **30 万到 80 万** 帧级样本就能看到比较明显的效果，但房间头和分割头会偏弱。
- **想做到可以支撑 VLN 的稳定地图输入**：通常要到 **100 万到 300 万** 帧级样本，外加 **20 万到 100 万** 张有像素级语义标注的关键帧。
- **想做到比较成熟的工程版本**：更稳妥的量级是 **300 万到 1000 万** 帧级样本，混合 RGB-D、投影伪深度、检测、分割和房间标签。

#### 一个经验性的判断

这套模型里，最先出效果的通常不是检测或分割，而是：

1. **位姿 / 深度**
2. **房间类型**
3. **常见物体检测**
4. **细粒度实例分割**

原因很简单：位姿和深度的监督更连续、更密集，先学会之后，后面的语义头才更容易把结果沉淀成稳定地图。检测和分割往往需要更干净的关键帧标注，训练数据虽然可以少一些，但标注质量更重要。
