# DINO-GeoSemMap 统一单模型架构

> 基于 01 方案的单 Backbone 优化版：在 LingBot-Map 的 DINOv2 + GCA 交替注意力框架内扩展语义能力，融合 VGGT-Omega 的 Scene Registers、Register Attention 和多任务训练损失  
> 版本：v0.2  
> 日期：2026-05-20  
> 阶段：方案设计，不涉及具体实现  
> 前置文档：[[01_DINO-GeoSemMap架构方案]]  
> 参考文档：[[VGGT-Omega]]、[[LingBot-Map_Framework_Analysis]]

---

## 0. 与 v0.1 的核心区别

### 0.1 v0.1 的问题

v0.1 将系统描述为六层架构，把 DINOv2 Backbone、GCA 模块、多任务 Heads 描述为三个独立组件。实际上，在 LingBot-Map 的实现中：

1. **GCA 不是一个"独立模块"**——它是 24 组交替注意力层（Frame Attention + GCA）中的一半，与 DINOv2 patch tokens 深度交织，共 48 个 Transformer Block；
2. **GCA 已经基于分页 KV-cache 实现**——三级上下文（Anchor / Window / Trajectory Memory）通过双流 paged KV-cache 管理，patch pages 可回收，special token pages 只追加；
3. **Camera Head 和 DPT Head 是轻量 readout**——Camera Head 是 4 层因果 Transformer + 迭代精炼，DPT Head 是多尺度特征融合上采样，两者都直接消费交替注意力的输出。

因此 v0.1 的"Backbone → GCA → Heads"三层描述是误导性的。真实结构是：

```text
DINOv2 编码 → 48 层交替注意力（Frame + GCA 交替，这就是模型的主体）
                              → Camera token → Camera Head (轻量 readout)
                              → Patch tokens → DPT Head (轻量 readout)
```

### 0.2 v0.2 的核心思路

v0.2 的优化不是"取消 GCA 换一个 decoder"，而是：

> **在 LingBot-Map 已验证的 DINOv2 + GCA 交替注意力框架内，通过扩展 token 词汇表来增加语义能力。**

具体做法：

1. 保留 DINOv2 ViT-L 作为唯一 backbone；
2. 保留 24 组 Frame Attention + GCA 交替注意力结构；
3. 保留分页 KV-cache 的三级上下文管理；
4. **在每帧 token 序列中追加语义任务 tokens**（object queries、map queries），让它们与 camera / register / scale / patch tokens 一起参与全部 48 层交替注意力；
5. 交替注意力输出后，不同 token 送入各自的轻量 readout head。

对比：

| 维度 | v0.1 | v0.2 |
|---|---|---|
| Backbone | DINOv2/v3 分阶段切换 | DINOv2 ViT-L 唯一 |
| 时序核心 | "独立 GCA 模块" | 保留 LingBot-Map 的 24 组交替注意力 + 分页 KV-cache |
| 语义能力 | 独立 Detection Head / Segmentation Head | 在交替注意力内追加 object / map tokens |
| 特殊 Token | 7 种（camera/register/scale + 4 种新增） | camera ×1 + **scene registers ×16**（from VGGT-Ω）+ scale ×1 + OBJ/MAP queries |
| GCA 跨帧注意力 | 所有 tokens 全量跨帧 attention | 引入 **Register Attention**（from VGGT-Ω），部分 GCA block 仅 registers 跨帧交互 |
| 训练损失 | 各 head 独立 loss | 增加 **point loss + matching loss**（from VGGT-Ω）作为辅助几何监督 |
| 任务 Head | 5 个独立 Head | Camera Head + DPT Head（沿用）+ Object Readout + Map Readout（新增轻量层） |
| 模型主体 | 不明确 | 48 层交替注意力就是唯一的模型主体 |

一句话：**在 LingBot-Map 的流式 GCA 框架中"塞入"语义 tokens，同时从 VGGT-Omega 吸收 scene registers、Register Attention 和多任务训练损失。**

---

## 1. 方案定位

与 v0.1 相同：面向室内导航场景，仅输入连续 RGB 视频流，在线输出 pose / depth / detection / segmentation / map tokens，构建 VLN 可用语义地图。

v0.2 的约束：

> **整个感知模型基于 LingBot-Map 的已验证架构做最小扩展。** GCA 的三级上下文结构和交替注意力机制是 pose / depth 稳定性的基础（消融实验：去掉 GCA 组件导致 ATE 退化 1-2 个量级），不应被替换。

---

## 2. 总体架构

### 2.1 LingBot-Map 原始架构（仅几何任务）

```text
RGB_t
  │
  ▼
DINOv2 ViT-L Backbone
  │
  │  每帧前插入 6 个特殊 token:
  │  [camera, reg×4, scale, patch_0, ..., patch_M]
  ▼
24 组交替注意力层（共 48 Block）
  ┌──────────────┐   ┌──────────────────────┐
  │Frame Attention│ → │GCA (Geometric Context │  × 24 组
  │  帧内自注意力  │   │     Attention)        │
  │  + 2D RoPE   │   │  三级因果注意力         │
  └──────────────┘   │  + 3D Video RoPE      │
                     │  + 分页 KV 缓存        │
                     └──────────────────────┘
  │
  ├── camera token → Camera Head (迭代精炼 ×4) → pose
  └── patch tokens → DPT Head (多尺度融合) → depth / point cloud
```

### 2.2 v0.2 扩展架构（几何 + 语义任务 + VGGT-Ω 融合）

```text
RGB_t
  │
  ▼
DINOv2 ViT-L Backbone (frozen / LoRA)
  │
  │  每帧 token 序列:
  │  [camera, scene_reg×16, scale, OBJ_0,...,OBJ_N, MAP_0,...,MAP_K, patch_0,...,patch_M]
  │   └── 原始 3 tokens ──┘            └─── 新增语义 tokens ──┘ └── patch tokens ──┘
  │        ↑ register 从 4 扩展为 16 (from VGGT-Ω)
  ▼
24 组交替注意力层（共 48 Block）
  ┌──────────────┐   ┌──────────────────────────────────────┐
  │Frame Attention│ → │GCA：混合 Full GCA + Register Attention│  × 24 组
  │  帧内自注意力  │   │                                      │
  │  所有 tokens   │   │  75% 组: Full GCA（所有 tokens 跨帧） │
  │  含新增语义     │   │  25% 组: Register Attention           │
  │  + 2D RoPE   │   │    （仅 registers 跨帧，patch 不跨帧） │
  └──────────────┘   │  + 三级 KV-cache + 3D Video RoPE     │
                     └──────────────────────────────────────┘
  在第 [4, 11, 17, 23] 组提取多尺度特征
  │
  ├── camera token ──→ Camera Head (迭代精炼 ×4) → pose_t
  ├── scene registers → 场景级空间表征（可供 VLN/VLA 使用）
  ├── patch tokens ──→ DPT Head (多尺度融合) → depth_t, uncertainty_t
  ├── OBJ tokens ────→ Object Readout (轻量) → boxes_t, masks_t, classes_t
  └── MAP tokens ────→ Map Readout (轻量) → scene_summary_t, keyframe_score_t
              │
              ▼
       Online Semantic Map System → VLN Map Interface
```

### 2.3 扩展的本质

v0.2 的扩展可以用一句话概括：

> **在 LingBot-Map 每帧 token 序列的 special tokens 和 patch tokens 之间，插入 N_obj 个 object query tokens 和 N_map 个 map query tokens，让它们与原有 tokens 一起参与 48 层交替注意力。**

这意味着：

1. 新增的 object / map tokens 在 Frame Attention 中与同帧的 patch tokens 交互，从当前帧视觉特征中提取语义信息；
2. 新增的 tokens 在 GCA 中通过三级 KV-cache 与历史帧交互，获得时序语义上下文；
3. 新增的 tokens 经过 48 层注意力后，已经充分融合了几何和语义信息，只需轻量 readout 即可输出。

---

## 3. 为什么保留 GCA 而不是替换

### 3.1 GCA 消融实验证据

LingBot-Map 的消融实验清楚表明 GCA 各组件对几何精度的贡献：

| 组件 | AUC@3 变化 | ATE 变化 | 核心作用 |
|---|---|---|---|
| Anchor Init. | +3.83 | -0.71 | 坐标系定标，尺度锚定 |
| Context Tokens（轨迹记忆） | +2.12 | -0.42 | 长程漂移校正 |
| Relative Loss | — | -0.79 | 局部旋转一致性 |
| Video RoPE | +0.64 | **-1.48** | 时序感知，最大单项改善 |

去掉 GCA 等于丢掉尺度锚定 + 长程漂移校正 + 时序位置编码三重保障。

### 3.2 GCA 的长程鲁棒性

当序列从 320 帧扩展到 3840 帧：

| 方法 | ATE_320 | ATE_3840 | 退化幅度 |
|---|---|---|---|
| CUT3R（循环压缩） | 18.16 | 32.47 | +78% |
| Wint3R（窗口） | 21.10 | 32.90 | +56% |
| LingBot-Map（GCA） | 6.42 | 7.11 | **+10.7%** |

GCA 的三级上下文结构使其在长序列上远优于替代方案。对于需要长期建图的 VLN 任务，这个特性尤其关键。

### 3.3 GCA 已经是 KV-cache

v0.2 最初考虑"用 decoder KV-cache 替代 GCA"。但实际上 GCA 已经就是 KV-cache 实现：

```text
GCA 的分页 KV-cache 双流设计：

Patch Page Pool (可回收):
  - scale frame pages: 锚定帧 patch，永不驱逐
  - window pages: 滑动窗口 patch，FIFO 回收
  - free pages: 空闲页池

Special Token Page Pool (只追加):
  - 所有历史帧的 6 个特殊 token KV
  - 永不驱逐 → 这就是轨迹记忆
```

这意味着"取消 GCA 改用 KV-cache"本身就是一个伪命题——GCA 的工程实现已经就是分页 KV-cache。

### 3.4 交替注意力的不可替代性

LingBot-Map 的 48 层不是"先跑 24 层 Frame Attention 再跑 24 层 GCA"，而是**交替**执行：

```text
Frame Block 0 → GCA Block 0 → Frame Block 1 → GCA Block 1 → ... → Frame Block 23 → GCA Block 23
```

这意味着每一层的帧内特征都会立即被跨帧上下文修正，然后修正后的特征再参与下一层帧内计算。这种深度交织产生的几何一致性无法被"先编码再 decode"的简单两阶段结构复现。

---

## 4. 从 VGGT-Omega 吸收的三个设计

LingBot-Map 提供了流式框架，但 VGGT-Omega 在三个维度上有已验证的改进，可以直接融入本方案。

### 4.1 为什么选择 LingBot-Map 而不是 VGGT-Omega 作为主框架

| 维度 | LingBot-Map | VGGT-Omega |
|---|---|---|
| 推理模式 | **因果流式**，逐帧处理 | 批量前馈，N帧一次性处理 |
| 注意力约束 | 因果（当前帧只看历史） | 双向（所有帧互相看） |
| 长序列内存 | GCA 三级 KV-cache，每帧仅增 6 token | 全局 attention，显存随帧数²增长 |
| 长序列鲁棒性 | 320→3840 帧仅退化 10.7% | 单 A100 最大 ~1250 帧 OOM |
| 实时性 | ~20 FPS 流式 | 1000 帧批处理 11.7s（非逐帧） |

**VGGT-Omega 不是流式模型。** 它是多帧 feed-forward reconstruction，不支持因果推理和长序列在线建图。对于机器人 VLN 场景（逐帧到达、实时决策、可能跑数千帧），LingBot-Map 的 GCA 是唯一已验证的解决方案。

但 VGGT-Omega 有三个组件被独立验证且可移植到 GCA 框架中。

### 4.2 设计一：Scene Registers ×16（替代 Register Tokens ×4）

**VGGT-Omega 的发现**：16 个 scene registers 不仅是注意力稳定器，还能聚合场景级空间语义信息。实验证据：

- 冻结的 scene registers 拼接进 OpenVLA-OFT，LIBERO 平均成功率从 97.1 提升到 98.5；
- 通过 InfoNCE 可与 VLM/LLM 文本 embedding 对齐，说明 registers 携带了场景级语义信息；
- 对中间 image tokens 做 PCA + k-means，无需 motion labels 即可分离动态物体与静态背景。

**v0.2 的做法**：将 LingBot-Map 的 4 个 register tokens 扩展为 16 个 scene registers。

```text
LingBot-Map 原始:  camera ×1 + register ×4  + scale ×1 = 6 特殊 tokens
v0.2 扩展:         camera ×1 + register ×16 + scale ×1 = 18 特殊 tokens
```

扩展后的 registers 承担双重角色：

1. **注意力稳定**（原始功能）：吸收冗余注意力，防止 patch token 质量下降；
2. **场景级表征**（VGGT-Ω 验证）：聚合跨帧全局信息，可直接作为 VLN/VLA 的空间 token 输入。

与 LingBot-Map 一致，所有 registers 保持**双变体**（scale frame 版本 + normal frame 版本）。

**对轨迹记忆的影响**：每帧 special tokens 从 6 增长到 18（+12），这远小于 OBJ queries 带来的增长，可忽略。

### 4.3 设计二：Register Attention（部分 GCA Block 的轻量替代）

**VGGT-Omega 的发现**：可视化 VGGT 的 global attention map 发现跨帧注意力非常稀疏——大部分信息交换集中在 registers 上，patch-to-patch 跨帧交互在很多层中是冗余的。

实验结果：

| 配置 | Point Error | FLOPs | 显存 |
|---|---|---|---|
| 100% Full Global Attention | 0.071 | 100% | 100% |
| 75% Full + 25% Register Attention | 0.073 | -23% FLOPs | -16% 显存 |
| 100% Register Attention | — | -94% FLOPs | — |

替换 25% 的 global attention 为 register attention 几乎无损（0.071 → 0.073），但 FLOPs 降 23%、显存降 16%。

**v0.2 的做法**：在 24 组 GCA blocks 中，将 6 组（25%）替换为 Register Attention。

```text
24 组 GCA 的注意力模式：

组 0-5:   Full GCA（所有 tokens 跨帧因果注意力）
组 6:     Register Attention（仅 registers 跨帧，patch/OBJ/MAP 不跨帧）
组 7-11:  Full GCA
组 12:    Register Attention
组 13-17: Full GCA
组 18:    Register Attention
组 19-22: Full GCA
组 23:    Register Attention

Register Attention 分布在第 [6, 12, 18, 23] 组
= 6 组 Register Attention / 24 组 = 25%
```

Register Attention 的具体行为：

```text
Full GCA Block:
  所有 tokens (camera + registers + scale + OBJ + MAP + patches)
  的 Q attend to 历史帧所有 tokens 的 KV
  → 完整的跨帧信息交换

Register Attention Block:
  仅 scene registers 的 Q attend to 历史帧 registers 的 KV（跨帧）
  patches / OBJ / MAP tokens 不参与跨帧 attention
  → registers 聚合跨帧全局信息
  → 在下一个 Frame Attention 中，registers 将全局信息分发给本帧 patches
```

**对 GCA KV-cache 的影响**：

Register Attention blocks 不需要访问历史帧的 patch pages，只需要 special token pages。这意味着这些 blocks 的注意力计算量大幅下降：

```text
Full GCA block 的 KV 长度:
  anchor patch + window patch + all special tokens
  ≈ 2×500 + 8×500 + T×18 = 5000 + 18T

Register Attention block 的 KV 长度:
  仅 all special tokens 中的 registers
  ≈ T×16

10000 帧时:
  Full GCA:      5000 + 180000 = 185000 KV tokens
  Register Attn: 160000 KV tokens (且只是 registers，不含 patch)
```

Register Attention 在超长序列中的加速效果更显著。

### 4.4 设计三：Point Loss + Matching Loss 作为辅助训练监督

**VGGT-Omega 的发现**：虽然推理时只输出 depth + camera，但训练时保留 point loss 和 matching loss 带来显著增益。

消融结果：

| 配置 | Point Error |
|---|---|
| 完整 loss（cam + depth + point + matching） | 0.071 |
| 去掉 point + matching losses | 0.078 |

VGGT-Omega 的核心洞察：

> **多任务监督收益不一定要求保留多 dense heads；可以只保留轻量输出头，在训练期通过几何一致性 loss 注入监督。**

**v0.2 的做法**：在训练期增加两个辅助 loss，推理时不保留对应 head。

```text
Point Loss:
  从预测 depth + pose 反投影得到点云 P_hat
  与 GT 点云 P 计算:
    L_point = weighted_L1(P_hat, P) + gradient_L1(P_hat, P) - α·log(confidence)

  不需要独立的 point head，只需要在训练期做 depth + pose → 反投影 → 与 GT 点云对比
  推理时不增加任何计算

Matching Loss:
  取最后一层交替注意力输出的 patch tokens
  构造正负 patch 对（基于 3D 投影重叠）:
    L_match = BCE(cosine_similarity(token_i, token_j), is_match)

  正样本：3D 投影重叠的 patch 对
  负样本：满足几何约束 + 外观差异（排除动态区域和未匹配区域）
  推理时不增加任何计算
```

这两个 loss 的作用：

1. **Point Loss** 强制 depth 和 pose 在 3D 空间中一致——单独的 depth loss 和 pose loss 可能各自"正确"但组合后 3D 投影错位；
2. **Matching Loss** 鼓励交替注意力的最终 patch features 包含可匹配的几何信息——这间接提升了 GCA 跨帧特征匹配的质量，有利于 pose 估计。

---

## 5. DINOv2 ViT-L 作为唯一 Backbone

### 5.1 砍掉 DINOv3 多阶段路线

v0.1 规划了 DINOv2 → DINOv3 → 蒸馏的三阶段切换。v0.2 只保留 DINOv2 ViT-L：

1. LingBot-Map 的全部实验结果（ATE 6.42、F1 98.98、~20 FPS）都基于 DINOv2 ViT-L 获得，已验证充分；
2. DINOv3 在 VGGT / LingBot-Map 体系中尚未验证，引入未经验证的 backbone 风险高于收益；
3. 交替注意力的 48 层是在 DINOv2 特征空间上训练的，换 backbone 等于重新训练整个模型；
4. 如果后续需要增强特征，走 LoRA / adapter 路线在 DINOv2 基础上微调，而不是换 backbone。

```text
唯一 Backbone = DINOv2 ViT-L
DINOv2 权重状态 = frozen（P0）或 LoRA fine-tune（P1+）
不规划 backbone 替换路线
```

### 5.2 Backbone 输出

与 LingBot-Map 完全一致：

```text
输入：RGB image I_t ∈ R^{H×W×3}

DINOv2 编码：
  patch_tokens ∈ R^{(H/14 × W/14) × 1024}

交替注意力中间层提取（第 [4, 11, 17, 23] 组）：
  多尺度特征 {F^l} 用于 DPT Head 和 Object Readout 的 mask 预测
```

多尺度特征不需要额外 FPN adapter——LingBot-Map 直接从交替注意力的中间层拼接 frame_feat 和 global_feat 得到 2048 维多尺度表示。

### 5.3 Backbone 冻结策略

| 阶段 | 策略 | 理由 |
|---|---|---|
| P0 | 完全冻结 DINOv2 | 与 LingBot-Map 一致，验证扩展是否有效 |
| P1 | 冻结 DINOv2 + LoRA（最后 4 层） | 如果语义任务需要更好的特征 |
| P2 | 冻结 DINOv2 + task-specific adapter | 如果特定任务（小物体检测）瓶颈在特征层 |

---

## 6. 扩展 Token 系统

### 6.1 LingBot-Map 原始 Token 序列

```text
每帧 token 序列（6 + M 个 token）：
┌────────┬─────┬─────┬─────┬─────┬───────┬────────┬────────┬───┐
│ camera │reg_0│reg_1│reg_2│reg_3│ scale │patch_0 │patch_1 │...│
│  idx=0 │  1  │  2  │  3  │  4  │   5   │   6    │   7    │   │
└────────┴─────┴─────┴─────┴─────┴───────┴────────┴────────┴───┘
  ← 6 个特殊 token →                      ← M 个 patch tokens →
```

### 6.2 v0.2 扩展后的 Token 序列

register tokens 从 4 扩展为 16 个 scene registers（§4.2），token 序列变为：

```text
每帧 token 序列（18 + N_obj + N_map + M 个 token）：
┌────────┬──────────────────────┬───────┬──────┬──────┬─────┬─────┬────────┬───┐
│ camera │ scene_reg_0 ... _15  │ scale │OBJ_0 │OBJ_1 │ ... │MAP_0│patch_0 │...│
│  idx=0 │  1  ...  16          │  17   │  18  │  19  │     │     │        │   │
└────────┴──────────────────────┴───────┴──────┴──────┴─────┴─────┴────────┴───┘
  ← 18 个几何/场景特殊 tokens  →         ← 语义 tokens →    ← M 个 patch tokens →
```

所有特殊 token 保持**双变体**（scale frame 版本 + normal frame 版本），通过 `nn.Parameter` 学习初始化，与 LingBot-Map / VGGT-Omega 一致。

### 6.3 完整 Token 定义

| Token | 数量 | 来源 | 初始化 | 功能 |
|---|---|---|---|---|
| camera token | 1 | 沿用 LingBot-Map | `nn.Parameter`, std=1e-6, 双变体 | 聚合帧级几何信息，输入 Camera Head |
| scene registers | **16** | **from VGGT-Ω**（原 4 个扩展） | `nn.Parameter`, std=1e-6, 双变体 | 注意力稳定 + 场景级空间表征 + Register Attention 的跨帧信息载体 |
| scale token | 1 | 沿用 LingBot-Map | `nn.Parameter`, std=1e-6, 双变体 | 区分锚定帧/普通帧，承载尺度信息 |
| OBJ queries | N_obj (50-100) | 新增 | `nn.Parameter`, std=1e-6, 双变体 | 每个 query 负责检测一个潜在实例，输出 box + class + mask |
| MAP queries | N_map (3-5) | 新增 | `nn.Parameter`, std=1e-6, 双变体 | 场景摘要、关键帧判断、自由空间感知、变化检测 |
| patch tokens | M | 沿用 LingBot-Map | DINOv2 编码输出 | 局部视觉、语义、几何 dense 信息 |

### 6.4 与 v0.1 Token 系统的对照

| v0.1 设计 | v0.2 处理 |
|---|---|
| camera token | 保留，不变 |
| register tokens ×4 | **扩展为 scene registers ×16**（from VGGT-Ω） |
| scale token | 保留，不变 |
| semantic token（新增） | 合并为 MAP query #1 |
| object token（新增） | 由 N_obj 个 OBJ queries 替代，从单一摘要变为实例级 queries |
| free-space token（新增） | 合并为 MAP query #2 |
| change token（新增） | 合并为 MAP query #3 |

### 6.5 Token 数量估算

```text
每帧 token 总量:
  特殊 tokens:  1 + 16 + 1 = 18
  OBJ queries:  50 (P0)
  MAP queries:  4
  patch tokens: ~500 (取决于分辨率)
  ─────────────────────
  总计:         ~572 tokens/帧

对比 LingBot-Map 原始: 6 + 500 = 506 tokens/帧
增量: +66 tokens/帧 (+13%)
```

---

## 7. 交替注意力中的 Token 行为

### 7.1 Frame Attention 中的行为

Frame Attention 在每帧内部执行全连接自注意力。所有 tokens（含扩展的 registers 和新增的 OBJ/MAP）参与帧内交互：

```text
帧 t 的 tokens:
[camera, scene_reg×16, scale, OBJ_0,...,OBJ_N, MAP_0,...,MAP_K, patch_0,...,patch_M]
          ↕ 全连接自注意力（所有 token 互相关注）↕
```

效果：

- OBJ queries 从当前帧的 patch tokens 中提取目标物体特征；
- MAP queries 从所有 tokens 聚合帧级场景摘要；
- camera token 可以从 OBJ queries 获得物体定位辅助（跨任务信息流）；
- **16 个 scene registers 聚合帧级全局信息**，为后续 Register Attention 做准备。

位置编码：使用 2D RoPE。特殊 tokens 分配对角线位置：

```text
camera:      (0, 0)
scene_reg_0: (1, 1)
...
scene_reg_15:(16, 16)
scale:       (17, 17)
OBJ_0:       (18, 18)
...
MAP_0:       (18+N, 18+N)
...
patch_0:     (patch_start + row, patch_start + col)
```

### 7.2 Full GCA Block 中的行为

在 Full GCA blocks（24 组中的 18 组）中，**所有 tokens** 参与跨帧因果注意力：

```text
当前帧的所有 tokens (作为 Q)
    attend to:
    ├── Anchor frames 的全量 token KV（含 anchor 帧的 registers/OBJ/MAP/patches）
    ├── Window frames 的全量 token KV（含近期帧的全量 tokens）
    └── Trajectory Memory 的 special token KV（含历史帧的压缩 special tokens）
```

效果：

- OBJ queries 可以跨帧匹配历史检测，实现隐式短期跟踪；
- MAP queries 可以从历史 MAP tokens 感知场景变化趋势；
- 所有 tokens 的 GCA 交互自然地将几何一致性传播给语义 tokens。

### 7.3 Register Attention Block 中的行为

在 Register Attention blocks（24 组中的 6 组，分布在第 [6, 12, 18, 23] 等位置）中，**仅 scene registers 参与跨帧注意力**：

```text
Register Attention 的注意力模式：

scene registers (Q) → attend to 历史帧 registers (KV)     ✓ 跨帧
camera / scale (Q)  → attend to 历史帧 camera/scale (KV)   ✓ 跨帧（数量极少）
OBJ / MAP tokens (Q) → 不参与跨帧 attention                 ✗
patch tokens (Q)     → 不参与跨帧 attention                 ✗
```

信息流路径：

```text
Register Attention block: registers 从历史帧 registers 聚合跨帧全局信息
        ↓
下一个 Frame Attention block: registers 在帧内自注意力中将全局信息分发给 patches/OBJ/MAP
        ↓
效果: patches/OBJ/MAP 间接获得跨帧信息，但不需要 patch-to-patch 的昂贵跨帧 attention
```

**为什么 OBJ/MAP tokens 在 Register Attention blocks 中不跨帧也可以？**

1. 它们在 18 个 Full GCA blocks 中已经有充分的跨帧交互；
2. Register Attention 通过 registers 作为信息瓶颈，间接传递跨帧信息；
3. VGGT-Omega 验证了这种混合模式几乎无损（point error 0.071 → 0.073）。

### 7.4 分页 KV-cache 的扩展

原始 LingBot-Map 的 special token 数量 = 6（camera + reg×4 + scale）。v0.2 扩展后：

```text
每帧 special tokens = 18 (camera + reg×16 + scale) + N_obj + N_map

例如 N_obj=50, N_map=4:
  全量 special tokens per frame = 72
```

但轨迹记忆不需要写入全部 72 个 token。采用选择性写入策略：

```text
写入轨迹记忆的 tokens（每帧）:
  camera token ×1         （保留，位姿信息）
  scene registers ×16     （保留，场景级表征 + Register Attention 必需）
  scale token ×1          （保留，尺度信息）
  MAP tokens ×N_map       （保留，场景摘要）
  Top-K OBJ tokens ×K     （只保留置信度最高的 K 个实例 query）

总量 = 18 + N_map + K ≈ 18 + 4 + 5 = 27 tokens/帧
```

对比：

| 配置 | 轨迹记忆 tokens/帧 | 10000 帧总量 | 备注 |
|---|---|---|---|
| LingBot-Map 原始 | 6 | 60K | 基线 |
| v0.2（选择性写入） | 27 | 270K | ×4.5 增长，register 扩展贡献 +12 |
| v0.2（全量写入） | 72 | 720K | 不推荐 |

27 tokens/帧的增长主要来自 16 个 scene registers（比原来的 4 个多 12），这是 Register Attention 在长序列中正常工作的必要条件——如果不保留历史 registers 的 KV，Register Attention blocks 就无法执行跨帧注意力。

### 7.5 Register Attention 对 KV-cache 的额外优化

Register Attention blocks 只需要访问 special token pages，不需要访问 patch pages。这意味着：

```text
Full GCA block 需要的 KV pages:
  anchor patch pages + window patch pages + all special pages

Register Attention block 需要的 KV pages:
  仅 special pages（只需读取 registers 部分）
```

在分页 KV-cache 的 page table 构建中，Register Attention blocks 可以跳过 patch pages，减少 page table 长度和 FlashInfer 的计算量。

---

## 8. Readout Heads

### 8.1 Camera Head（沿用 LingBot-Map）

```text
输入：camera token（经过 48 层交替注意力后）

Camera Head 结构（不修改）：
  4 次迭代精炼
  每次迭代：
    embed(current_pose) → AdaLN 调制 camera token
    → 4 层因果 Transformer Block（独立 KV-cache，3D Video RoPE）
    → MLP → Δpose
    → pose += Δpose

输出：9D 位姿编码 [T_x, T_y, T_z, q_i, q_j, q_k, q_w, fov_h, fov_w]
  → R, t (外参)
  → K (内参, from FoV)
  → confidence
```

Camera Head 完全沿用 LingBot-Map，不做任何修改。

### 8.2 DPT Head（沿用 LingBot-Map）

```text
输入：第 [4, 11, 17, 23] 组的多尺度 patch features（2048 维）

DPT Head 结构（不修改）：
  4 个尺度的 project + refinenet
  自底向上融合
  最终卷积 → [depth/xyz, confidence]

输出：
  dense depth map + uncertainty
  或 3D point map + confidence
```

DPT Head 完全沿用 LingBot-Map，不做任何修改。

### 8.3 Object Readout（新增）

```text
输入：
  OBJ tokens（经过 48 层交替注意力后）∈ R^{N_obj × 1024}
  多尺度 patch features（用于 mask 预测）

结构：
  Classification:  OBJ token → Linear(1024, num_classes) → class logits
  Box Regression:  OBJ token → MLP(1024→512→4) → normalized box [cx, cy, w, h]
  Objectness:      OBJ token → Linear(1024, 1) → objectness score
  Mask Prediction: OBJ token × multi-scale features → dot product → mask logits
                   （Mask2Former 风格，在多尺度特征图上做 pixel-level 点积）

输出：
  boxes_t     ∈ R^{N_obj × 4}
  classes_t   ∈ R^{N_obj × num_classes}
  scores_t    ∈ R^{N_obj}
  masks_t     ∈ R^{N_obj × H × W}
```

Object Readout 是纯轻量层（几个 Linear/MLP），不含 Transformer blocks。这是因为 OBJ tokens 已经在 48 层交替注意力中充分与视觉/几何特征交互过，不需要额外的重型 decoder。

### 8.4 Map Readout（新增）

```text
输入：MAP tokens（经过 48 层交替注意力后）∈ R^{N_map × 1024}

结构：
  MAP token #1 → Linear → scene_summary_embedding ∈ R^{256}
  MAP token #2 → Linear → free_space_summary ∈ R^{256}
  MAP token #3 → Linear → change_score ∈ R^{1}
  Concat(all MAP tokens) → MLP → keyframe_score ∈ R^{1}

输出：
  scene_summary_t
  free_space_summary_t
  change_score_t
  keyframe_score_t
```

### 8.5 Readout 参数量估算

| Head | 来源 | 估计参数量 | 备注 |
|---|---|---|---|
| Camera Head | 沿用 LingBot-Map | ~25M | 4 层因果 Transformer + MLP |
| DPT Head | 沿用 LingBot-Map | ~10M | 多尺度 refinenet + conv |
| Object Readout | 新增 | ~2M | 几个 Linear/MLP |
| Map Readout | 新增 | ~0.5M | 几个 Linear |
| **新增总量** | | **~2.5M** | **仅占模型总量 ~0.5%** |

v0.2 新增的参数量极小，绝大部分模型结构沿用 LingBot-Map。

---

## 9. 模型输出定义

每个时间步 `t` 的模型输出：

```text
Y_t = {
    geometry: {
        pose_t,               ← Camera Head (沿用)
        depth_t,              ← DPT Head (沿用)
        depth_uncertainty_t,  ← DPT Head (沿用)
        optional_pointmap_t   ← DPT Head (沿用)
    },

    semantics: {
        boxes_t,              ← Object Readout (新增)
        classes_t,            ← Object Readout (新增)
        scores_t,             ← Object Readout (新增)
        masks_t               ← Object Readout (新增)
    },

    memory: {
        scene_summary_t,      ← Map Readout (新增)
        free_space_summary_t, ← Map Readout (新增)
        change_score_t,       ← Map Readout (新增)
        keyframe_score_t      ← Map Readout (新增)
    }
}
```

---

## 10. 训练策略

### 10.1 核心思路：从已训练的 LingBot-Map 出发

v0.2 不从零训练，而是从 LingBot-Map 预训练权重出发做增量训练：

```text
已有：LingBot-Map 预训练权重（DINOv2 + 48 层交替注意力 + Camera Head + DPT Head）
新增：OBJ query embeddings + MAP query embeddings + Object Readout + Map Readout
训练：冻结已有权重，只训练新增部分 → 逐步解冻联合微调
```

### 10.2 推荐训练流程

```text
Stage 1: 语义 Token 预热
  冻结：DINOv2 + 48 层交替注意力 + Camera Head + DPT Head
  训练：OBJ query embeddings + MAP query embeddings + Object Readout + Map Readout
  数据：室内场景图像 + detection/segmentation 标注（可以是单帧数据，不需要视频序列）
  Loss: L_detection + L_segmentation + L_map_auxiliary
  目的：让新增的 OBJ/MAP tokens 学会从冻结的交替注意力特征中提取语义信息
  关键：此阶段不应影响 pose/depth 性能

Stage 2: 联合微调（交替注意力层解冻）
  解冻：48 层交替注意力（或其中的后半部分）
  冻结：DINOv2 backbone
  训练：全部 heads + 交替注意力层
  数据：室内导航视频序列 + pose + depth + detection + segmentation 标注
  Loss: L_pose + L_depth + L_detection + L_segmentation + L_map_auxiliary
        + L_point + L_matching  ← VGGT-Ω 辅助 loss，此阶段引入
  目的：建立跨任务信息流，让几何和语义相互促进；point/matching loss 强化几何一致性

Stage 3 (可选): LoRA backbone 微调
  对 DINOv2 最后 4 层加 LoRA
  如果 Stage 2 后语义任务瓶颈在特征层
```

### 10.3 Loss 设计

```text
L_total = λ_pose · L_pose
        + λ_depth · L_depth
        + λ_det · L_detection
        + λ_seg · L_segmentation
        + λ_map · L_map_auxiliary
        + λ_point · L_point           ← from VGGT-Ω (训练期辅助，推理时无开销)
        + λ_match · L_matching        ← from VGGT-Ω (训练期辅助，推理时无开销)

其中：
  L_pose = absolute pose loss + relative pose loss     (沿用 LingBot-Map)
  L_depth = weighted L1 + gradient L1 - uncertainty reg (沿用 LingBot-Map)
  L_detection = Hungarian matching loss (DETR-style)    (新增)
  L_segmentation = mask BCE + dice loss                 (新增)
  L_map_auxiliary = keyframe binary CE + scene summary contrastive loss (新增)

  L_point = weighted_L1(P_hat, P) + gradient_L1(P_hat, P) - α·log(σ)   (from VGGT-Ω, §4.4)
    其中 P_hat = unproject(depth_t, pose_t)，与 GT 点云 P 对比
    强制 depth 和 pose 在 3D 空间中一致
    不需要独立 point head，仅训练期通过 depth+pose 反投影计算

  L_matching = BCE(cos_sim(token_i, token_j), is_match)                 (from VGGT-Ω, §4.4)
    正样本：3D 投影重叠的 patch 对
    负样本：满足几何约束 + 外观差异（排除动态区域和未匹配区域）
    鼓励交替注意力最终输出的 patch features 包含可匹配的几何信息
    不需要额外输出头，仅训练期从最后一层 patch tokens 采样计算
```

Point loss 和 matching loss 是 VGGT-Omega 验证的关键训练技巧（去掉后 point error 从 0.071 退化到 0.078）。它们只在训练期参与梯度计算，推理时零开销。

### 10.4 关键训练约束

**Pose/Depth 不退化原则**：任何训练阶段完成后，都需要验证 pose 和 depth 性能不低于 LingBot-Map 基线。如果退化：

1. 检查 loss 权重，降低语义 loss 权重；
2. 回退交替注意力的解冻范围；
3. 考虑在注意力层中对几何 tokens 和语义 tokens 的梯度做分离。

---

## 11. 流式推理机制

### 11.1 推理流程

完全继承 LingBot-Map 的两阶段推理，扩展 token 集合：

```text
Phase 1: 锚定帧处理（前 n 帧，双向注意力）
  scale_images → DINOv2 → prepend [camera, scene_reg×16, scale, OBJ×N, MAP×K, patches]
  → forward(双向注意力, 所有 scale 帧作为一个 block)
  → 初始化 Anchor KV-cache（含 OBJ/MAP tokens 的 KV）
  → 输出 anchor 帧的 pose/depth/detection/segmentation

Phase 2: 流式推理（逐帧因果）
  for each frame:
    frame → DINOv2 → prepend [camera, scene_reg×16, scale, OBJ×N, MAP×K, patches]
    → forward(因果注意力, 使用三级 KV-cache)
    → Camera Head → pose_t
    → DPT Head → depth_t
    → Object Readout → boxes_t, masks_t, classes_t
    → Map Readout → scene_summary_t, keyframe_score_t
    → 更新 KV-cache（Window FIFO + 选择性 Memory 写入）
    → 输出 Y_t 给地图系统
```

### 11.2 多频率推理

继承 v0.1 的多频率设计，但实现方式是控制 OBJ/MAP tokens 的参与：

```text
高频帧 (10-20 Hz):
  所有 tokens 参与 Frame Attention + GCA
  但只运行 Camera Head + DPT Head
  OBJ/MAP tokens 虽然参与注意力（提供跨任务信息），但不运行 Readout

关键帧 (1-3 Hz):
  所有 tokens 参与 Frame Attention + GCA
  运行全部 Readout heads
  完整输出 Y_t
```

这里有一个重要设计选择：**高频帧中 OBJ/MAP tokens 是否参与注意力。**

方案 A：高频帧中不包含 OBJ/MAP tokens

```text
优点：高频帧计算量与 LingBot-Map 完全一致
缺点：OBJ/MAP tokens 在高频帧间断了，关键帧时需要"冷启动"
```

方案 B：高频帧中包含 OBJ/MAP tokens 但不运行 Readout（推荐）

```text
优点：OBJ/MAP tokens 在 GCA 中持续积累信息，关键帧时 Readout 质量更高
缺点：高频帧 token 数量增加 ~10-20%
计算增量：~500 原始 tokens → ~560 tokens，增加 ~12%
```

推荐方案 B，因为 12% 的计算增量换来的是 OBJ/MAP tokens 的持续时序积累。

### 11.3 Keyframe 触发

除 v0.1 的条件外，v0.2 利用 MAP tokens 提供学习到的 keyframe 判断：

```text
规则触发（与 v0.1 一致）：
  位移/旋转超过阈值
  进入新区域
  接近 doorway / frontier
  VLN 模块请求

学习触发（v0.2 新增）：
  Map Readout 输出的 keyframe_score > θ
  Map Readout 输出的 change_score > θ
```

学习到的 keyframe 判断可以捕捉规则难以定义的情况，如"场景中出现了一个从未见过的物体"。

---

## 12. 与 v0.1 地图系统的兼容性

v0.2 的模型输出格式是 v0.1 的超集。地图系统不需要修改：

```text
v0.1 输出: pose_t, depth_t, boxes_t, masks_t, classes_t, map_tokens_t
v0.2 输出: pose_t, depth_t, boxes_t, masks_t, classes_t, scene_summary_t, keyframe_score_t, ...
           ↑ 与 v0.1 完全兼容                               ↑ 新增字段，地图系统可选消费
```

v0.1 中关于地图表示设计（Metric / Semantic / Topological / Trajectory 层）、语义实例融合策略、VLN 接口设计的内容完全沿用，不重复。

---

## 13. 关键优势

### 13.1 最小化架构风险

v0.2 的核心框架（DINOv2 + GCA 交替注意力 + Camera Head + DPT Head）已经在 LingBot-Map 中被验证，新增内容仅为：

- 50-100 个 OBJ query tokens 的 learnable embeddings
- 3-5 个 MAP query tokens 的 learnable embeddings
- ~2.5M 参数的 Readout layers

这意味着即使语义扩展完全失败，pose/depth 能力仍保留。

### 13.2 自然的跨任务信息流

OBJ/MAP tokens 与 camera/patch tokens 在同一个 48 层交替注意力中交互，信息流是双向的：

```text
camera token ← 从 OBJ tokens 获得物体匹配先验 → 更稳定的位姿
patch tokens ← 从 OBJ tokens 获得语义注意力引导 → 更 task-aware 的特征
OBJ tokens ← 从 camera token 获得几何上下文 → 更准确的 3D 定位
OBJ tokens ← 从 GCA 轨迹记忆获得历史观测 → 跨帧实例关联
MAP tokens ← 从所有 tokens 获得全局摘要 → 更准确的场景理解
```

### 13.3 GCA 时序能力自然传递给语义任务

OBJ tokens 参与 GCA 意味着它们自动获得：

1. **从 Anchor KV 获得初始场景语义参考**；
2. **从 Window KV 获得短期实例跟踪能力**（不需要额外的 tracking 模块）；
3. **从 Trajectory Memory 获得长程语义回忆**。

这些能力在传统"检测器 + 跟踪器"的松耦合架构中需要单独模块实现。

### 13.4 单一训练和部署流程

```text
模型 = DINOv2 (frozen) + 48 层交替注意力 + 4 个 Readout Heads
权重文件 = 1 个 .pt 文件
推理入口 = 1 个 forward() 调用
```

---

## 14. 关键风险与缓解

### 14.1 语义 Token 对几何精度的干扰

风险：新增的 OBJ/MAP tokens 在注意力中"抢占"了几何 tokens 的注意力资源。

严重性：**高**。这是 v0.2 最大的风险。

缓解：

```text
1. 分阶段训练：先冻结已有权重只训练新增部分，确认 pose/depth 不退化后再联合微调
2. OBJ/MAP tokens 初始化为极小值（std=1e-6），与 LingBot-Map 的特殊 token 初始化策略一致
3. 如果退化严重，考虑在注意力中对几何/语义 tokens 做分组：
   - Frame Attention 中所有 tokens 全连接（跨任务交互）
   - GCA 中几何 tokens 和语义 tokens 分别 attend to 各自的历史 KV
     （相当于 GCA 内部有两条 KV-cache 流，降低互相干扰）
4. 监控指标：每个训练阶段完成后对比 LingBot-Map 基线的 ATE / depth AbsRel
```

### 14.2 轨迹记忆的显存增长

风险：每帧 special tokens 从 6 增长到 18（scene registers 从 4 扩展为 16），加上 OBJ/MAP queries 的选择性写入，超长序列显存压力增加。

缓解：

```text
1. 选择性写入：只将 Top-K 高置信度 OBJ tokens 写入轨迹记忆（§7.4 策略）
2. 语义 KV 分池管理：geometric special 永不淘汰，semantic special 可按策略淘汰
3. 非关键帧不写入 OBJ/MAP tokens 的 KV
4. Register Attention blocks 只访问 special pages（§7.5），降低长序列注意力开销
5. 预期影响：采用选择性写入后，每帧轨迹记忆从 6 tokens 增长到 ~27 tokens
   10000 帧序列：6 × 10000 = 60K → 27 × 10000 = 270K tokens
   显存增量：270K × 1024 × 2 × 2 = ~1.1 GB（在 16GB GPU 上可接受）
```

### 14.3 OBJ Queries 数量的选择

风险：OBJ queries 数量过多增加计算量，过少漏检。

缓解：

```text
室内场景典型可见物体数量 = 5-20
考虑到 DETR 经验，N_obj = 50-100 是合理范围

P0: N_obj = 50（保守，优先验证架构可行性）
P1: 根据 recall 调整，如果漏检严重增加到 100
```

### 14.4 Frozen Backbone 的语义表征能力

风险：DINOv2 自监督预训练的特征可能不够支撑检测/分割。

缓解：

```text
1. DINOv2 的特征已被多个下游任务验证（DepthAnything 用于 depth，DINO-det 用于 detection）
2. 48 层交替注意力本身就是一个强大的特征精炼器，可以弥补 backbone 特征的不足
3. 如果确实不够，对 backbone 最后几层加 LoRA（Stage 3）
4. 对于开放词汇检测，后续可在 OBJ Readout 中加 CLIP text encoder adapter
```

### 14.5 RGB-only 深度尺度不稳定

与 v0.1 和 LingBot-Map 相同。GCA 的 anchor KV + scale token 双变体是现有的缓解方案。

---

## 15. 推荐阶段路线

### 15.1 Phase 0：验证 Token 扩展不破坏几何

```text
Backbone: DINOv2 ViT-L (frozen, LingBot-Map 权重)
交替注意力: 48 层 (frozen, LingBot-Map 权重)
Camera Head + DPT Head: frozen, LingBot-Map 权重
新增: OBJ queries ×50 + MAP queries ×4 + Object Readout + Map Readout
训练: 只训练新增部分，室内单帧数据
KV-cache: 与 LingBot-Map 一致

验证目标：
  1. pose ATE 不退化（与 LingBot-Map 基线对比）
  2. depth AbsRel 不退化
  3. 检测 mAP 达到可用水平
  4. 分割 mask IoU 达到可用水平
```

### 15.2 Phase 1：联合微调 + 流式语义

```text
在 Phase 0 基础上：
解冻: 交替注意力后半部分（第 12-23 组）
训练: 全部 heads + 解冻层，室内视频序列数据
KV-cache: 扩展轨迹记忆以包含选择性 OBJ/MAP tokens

验证目标：
  1. 跨任务信息流是否带来 pose/depth 提升
  2. OBJ tokens 在 GCA 中是否实现跨帧实例关联
  3. MAP tokens 的 keyframe_score 是否优于规则触发
  4. 长序列（1000+ 帧）地图构建质量
```

### 15.3 Phase 2：完整地图系统

```text
在 Phase 1 基础上：
接入完整地图系统（Metric + Semantic + Topological layers）
接入 VLN 接口

验证目标：
  1. 端到端从 RGB 到 VLN 可用地图的完整流程
  2. 地图中 object landmark 的稳定性
  3. VLN 查询接口的准确性和延迟
```

### 15.4 Phase 3（可选）：增强与优化

```text
Backbone LoRA 微调
开放词汇检测 adapter
KV-cache 量化
部署优化
```

---

## 16. 评估指标

沿用 v0.1 的所有指标。新增 v0.2 特有的：

| 指标 | 含义 |
|---|---|
| geometry-preservation | 加入语义 tokens 后 pose/depth 相对 LingBot-Map 基线的退化量，**必须 ≤ 5%** |
| cross-task gain | 联合训练 vs 单独训练各任务的性能差异 |
| temporal-semantic-gain | OBJ tokens 在 GCA 中跨帧积累 vs 单帧检测的性能差异（验证 GCA 对语义的贡献） |
| kv-cache-overhead | 扩展 token 后 KV-cache 的显存增量 |
| keyframe-quality | MAP 学习触发 vs 规则触发的 keyframe 选择质量对比 |

---

## 17. 最终推荐架构

```text
DINO-GeoSemMap v0.2 — GCA-Extended Semantic Architecture + VGGT-Ω Fusion

Input:
    Streaming RGB

Backbone:
    DINOv2 ViT-L (frozen / LoRA)
    与 LingBot-Map 完全一致

Token Sequence (per frame):
    [camera ×1, scene_reg ×16, scale ×1] ← scene registers 从 4 扩展为 16 (from VGGT-Ω)
    [OBJ queries ×50, MAP queries ×4]    ← 新增语义 tokens
    [patch tokens ×M]                    ← 沿用 LingBot-Map

Alternating Attention (48 blocks):
    24 组 [Frame Attention → GCA]
    18 组 Full GCA + 6 组 Register Attention (from VGGT-Ω, 25% 替换)
    Register Attention: 仅 registers 跨帧交互，节省 23% FLOPs / 16% 显存
    三级 KV-cache: Anchor + Window + Trajectory Memory
    3D Video RoPE + 2D RoPE

Training Losses:
    L_pose + L_depth + L_detection + L_segmentation + L_map_auxiliary
    + L_point + L_matching (from VGGT-Ω, 训练期辅助，推理时零开销)

Readout Heads:
    Camera Head (沿用)     → pose_t
    DPT Head (沿用)        → depth_t, uncertainty_t
    Object Readout (新增)  → boxes_t, masks_t, classes_t
    Map Readout (新增)     → scene_summary_t, keyframe_score_t

Map System:
    与 v0.1 相同

VLN Interface:
    与 v0.1 相同
```

一句话总结：

> DINO-GeoSemMap v0.2 以 LingBot-Map 的 DINOv2 + GCA 交替注意力为基础框架，融合 VGGT-Omega 的三个已验证设计——16 个 scene registers 替代 4 个 register tokens 以增强场景级表征、25% GCA blocks 采用 Register Attention 以降低计算开销、训练期加入 point loss 和 matching loss 以强化几何一致性——同时在 token 序列中追加 OBJ queries 和 MAP queries，使语义与几何在同一个 48 层交替注意力中深度交互。整个系统仍是一个 backbone、一套交替注意力、一个 KV-cache，新增参数仅 ~2.5M，自然继承 GCA 的三级时序上下文能力。
