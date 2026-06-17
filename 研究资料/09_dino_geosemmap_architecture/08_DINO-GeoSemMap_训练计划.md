# DINO-GeoSemMap 训练计划

> 版本：v0.1
> 日期：2026-06-01
> 关联文档：[07_DINO-GeoSemMap_架构设计.md](07_DINO-GeoSemMap_架构设计.md)

---

## 0. 文档定位

本文是 [07 架构设计](07_DINO-GeoSemMap_架构设计.md) 的训练落地细化，回答"**怎么训出来**"。

**本次训练的核心目标**：把原先 **depth / detection / mask 三个独立模型**，合并到一个 **冻结 DINOv2 + 多独立 head** 的统一框架里，共享一次 backbone forward。

**首轮范围**：打通统一框架**全部四个 head** 的核心链路——深度（本地）+ 检测 / 分割 / 房间（云上并行）。**每一轮都对全部 head 做效果确认**（见 §2.1），不把 Mask / Room 推迟到最后。四个阶段在阶段0 完成后相互独立、可并行。

---

## 1. 训练总原则

```text
DINOv2 ViT (全程冻结，不参与任何梯度更新)
  │
  ▼
Multi-scale Feature Adapter
  │  └── 仅在阶段0 与 Depth Head 联合训练，训完即冻结
  │      → 之后作为所有 head 共享的"固定特征基座"
  ▼
├── Depth Head      ← 阶段0（与 Adapter 联训）
├── Detection Head  ← 阶段1（独立）
├── Mask Head       ← 阶段2（独立，训练用 GT bbox）
└── Room Head       ← 阶段3（独立）
```

三条铁律：

1. **DINOv2 全程冻结**——所有可训练参数都在 Adapter + 各 head 内。
2. **唯一的阶段依赖**：阶段0 必须最先完成（Adapter 冻结），阶段1/2/3 才能基于固定特征基座开训；阶段1/2/3 之间无依赖（多卡可并行；**单卡 PRO6000 下按依赖无序、资源上串行排队执行**）。
3. **Adapter 一旦冻结，整个框架的"相机/焦距适应范围"就锁死**——分辨率固定 896×504（不做多分辨率）；**多焦距覆盖必须在阶段0 喂满**（见 §3.4）。

> 工程含义：阶段1/2/3 因为 backbone + adapter 全冻结，可以**预先缓存 adapter 输出特征**再训 head，大幅加速（见 §8.3）。

---

## 2. 训练阶段总览

| 阶段 | 训练内容 | DINOv2 | Adapter | 数据规模（原型） | 预计时间(1×PRO6000 96G,算力) | 前置 |
|---|---|---|---|---|---|---|
| **0** | Adapter + DPT Depth Head | frozen | **训练** | 50–100 万帧 | ~3–6 天 | 无 |
| **1** | RF-DETR 式 Detection（Top-50） | frozen | frozen | 10–30 万张 | ~1–2 天（缓存特征更快） | 阶段0 |
| **2** | Bbox-based Mask Head | frozen | frozen | 5–20 万张 | ~0.2–0.6 天 | 阶段0 |
| **3** | Room Semantics Head | frozen | frozen | 5–20 万帧 | ~0.05–0.2 天 | 阶段0 |

```text
阶段0 (Adapter+Depth)
   │  训完冻结 Adapter = 固定特征基座
   ▼
   ├── 阶段1 Detection ┐
   ├── 阶段2 Mask      ├── 依赖上无序；
   └── 阶段3 Room      ┘
```

> 单卡训练:总周期 ≈ 各阶段**相加**(非取最大),失去多机并行红利。

### 2.1 训练推进策略：预验证轮 → 正式首轮 → 完整数据轮

正式训练前先做**预验证轮**（小/中量数据），验证效果可行性 + 打磨训练框架，降低在大数据上踩坑的成本。三段推进：

| 轮次        | 数据量      | 目标                     | 各 head 效果确认                   | 算力 / 存储（约）         |
| --------- | -------- | ---------------------- | ----------------------------- | ------------------ |
| **预验证轮**  | 小量 → 中量  | 框架正确性 + 效果可行性 + 调参/调框架 | 深度 / 检测 / 分割 / 房间 **全部跑通看趋势** | 天级 / ~0.02–0.1 TB  |
| **正式首轮**  | 原型（起步）   | 打通统一框架核心链路（M1–M4）      | 深度 / 检测 / 分割 / 房间 **全部达验收**   | ~1–1.5 周 / ~0.3 TB |
| **完整数据轮** | 较好效果（完整） | 四 head 提质              | 深度 / 检测 / 分割 / 房间 **全部提质**    | ~3–4 周 / ~1.5 TB   |

> **每轮全 head 确认（铁律）**：每一轮都对**全部四个 head（深度 / 检测 / 分割 / 房间）**做效果确认——Mask / Room 虽轻（阶段2/3 仅小时级、云上与检测并行），也**不推迟到最后**，每轮都跑一遍训练 + 验收，尽早暴露"冻结 Adapter 特征是否适配各任务"的问题。

> **硬件分配（详见 §8.1）**：**先本地打通、再上云扩量**——预验证轮 + 正式首轮**全程本地 PRO6000**（四 head 串行验证）；待首轮四 head 收敛后，**完整数据轮**再把检测/分割/房间上云并行。

**预验证轮细分两级：**

```text
L0 Smoke（过拟合小集）: 数百–数千样本
   目标: loss 能压到极低 → 证明 模型/损失/梯度通路/dataloader 无 bug
L1 小量: 深度 1–5 万帧 / 检测 1–2 万张
   目标: 指标方向正确、收敛趋势出现
L2 中量: 深度 10–20 万帧 / 检测 3–5 万张
   目标: 接近原型效果的早期信号，定 batch / lr / 梯度累积 / 损失权重
```

**预验证轮必查清单：**

- [ ] **Canonical Transform 数值正确性**——用多个不同 K 的样本，验证 `depth_metric = depth_canon · (f_real/f_c)` 反归一化后与 GT 米制一致（多焦距都对）。**这是多相机能否成立的命门，必须最先单独验证**；
- [ ] **冻结确认**——DINOv2 参数训练前后不变（hash 比对）、`no_grad` 生效（显存符合 §8.4 预期）；
- [ ] **多分辨率不崩**——{392…588} 各档前向正常，指标不因分辨率剧烈漂移；
- [ ] **显存/吞吐实测**——回填校准 §10.1 的吞吐假设（60/80 img/s）与 batch/累积配置；
- [ ] **特征缓存流程**（§8.3）——验证 缓存 → 读取 → 训 head 链路可用；
- [ ] **检测 / 分割 / 房间 head 跑通**——L2 中量上三头训练收敛、指标方向正确（检测召回雏形、分割框内 IoU、房间 Top-1），**确认冻结 Adapter 特征适配各任务**；
- [ ] **checkpoint / resume** 正常。

**毕业判据（进入正式首轮）：** L0 过拟合成功 + Canonical 反归一化误差达标 + L2 中量上深度 AbsRel 明显下降、检测召回出现合理雏形 + 显存/吞吐/调参定稿。

---

## 3. 阶段0：Adapter + Depth Head（重点）

### 3.1 目标

训出一个 **单帧、米制、多相机/多分辨率通用** 的深度头——目标是"**像双目深度估计一样，开箱给米制深度**"，不依赖外部尺度对齐。

用途定位：深度**只用于"检测到物品后，结合 SLAM 位姿解算物品 3D 世界坐标"**。导航/避障深度由硬件双目 + 超声独立负责，不走本头。因此：

- **单帧**即可，无需时序一致性（物体是离散 landmark，不做稠密全局融合）；
- 深度头可**只在关键帧运行**，与 detection/mask/room 共享同一次 forward；
- 精度要求集中在**物体区域**（取物体 mask 前景深度中位数当物体距离，对逐像素噪声鲁棒）；
- 但**训练用全图稠密监督**（DPT 头免费给稠密 + 尺度/正则需要），**消费时稀疏取用**。

### 3.2 多相机方案：Canonical Camera Transform

输入图像统一为**去畸变针孔图**。单目 metric 的根本歧义是"焦距 ↔ 尺度"耦合，用 Metric3D 式的标准相机归一化消除，**不改网络结构**，相机参数只进前/后处理：

```text
常量:  f_c = 标准焦距（按部署相机分布选定，例如 1000px @ 518 工作分辨率）

—— 训练预处理（每样本）——
1. 保持长宽比 resize 原图到工作分辨率（x/y 同一缩放比 r），pad 补齐
     等效焦距  f' = sqrt(fx · fy) · r
2. center-crop/pad 将 principal point 拉到画面中心附近
3. GT 深度缩放到标准空间:   d_gt_canon = d_gt · (f_c / f')
   → 网络只学"标准相机"这一种 metric，焦距歧义消除

—— 推理后处理 ——
   网络输出 depth_canon
   反归一化:   depth_metric = depth_canon · (f_real / f_c)
   （f_real = 当前真实相机在同样 resize 后的等效焦距）
```

> 注意：**保持长宽比 resize + pad**，让 x/y 共用一个 r，f' 才定义干净；各向异性 resize 会让单一 f_c 近似失真。

### 3.3 网络结构与张量 I/O

```text
DINOv2(frozen) layer[4,11,17,23]
  → Multi-scale Feature Adapter（本阶段训练，训完冻结）→ P2,P3,P4,P5
  → DPT refinenet 自底向上融合: path4→path3→path2→path1
  → output_conv → 2 通道
```

```text
网络输入:
  RGB:  [B, 3, H, W]        # 去畸变针孔图，H/W 为 14 的倍数
  (K 不进网络，仅作前/后处理元数据)

网络输出:
  depth_canon: [B, 1, H, W]  # 标准相机空间深度，inv_log 激活 sign(y)·(exp(|y|)-1)
  confidence:  [B, 1, H, W]  # softplus，作为 aleatoric 不确定性

推理产物（后处理后）:
  depth_metric:[B, 1, H, W]
  valid_mask:  [B, 1, H, W]  # (conf > τ) & (d ∈ [d_min, d_max])
```

### 3.4 数据 pipeline（决定整个框架的相机/焦距适应范围）

```text
数据源（每条都需带 K + valid_mask）:
  - ARKitScenes          # 超大、室内、iPhone LiDAR metric，主力
  - ScanNet / ScanNet++  # 室内 metric
  - Hypersim             # 合成，干净 metric + 锐利边界，补边界质量
  - NYUv2 / Matterport3D
  - Habitat 仿真 🎯       # 按机器人真实相机内参(FOV 120×85)+机位高度渲染
                         #   原生宽 FOV + rectilinear 针孔，匹配"去畸变针孔"假设
                         #   补"宽 FOV + 机器人视角"的 in-domain 覆盖（公开数据多为窄 FOV）

固定工作分辨率 896×504 (16:9)，所有输入统一 resize 到此，不做多分辨率。
增强（多相机焦距能力的来源）:
  - 多焦距:   混不同 K 的数据集 + 随机 crop→resize 到 896×504 的焦距增强
              (输出尺寸不变、等效焦距变化；Canonical Transform 保证标签始终一致)
  - 常规:     color jitter / 水平翻转(同步翻 cx)
```

> ⚠️ Adapter 在本阶段训完即冻结、被后续所有 head 共享。分辨率固定 896×504；**多焦距必须在此喂满**，否则后期扩相机焦距需重训 Adapter，牵动全部 head。

### 3.5 损失（全部在 canonical 空间计算）

```text
L_depth = λ_l1   · valid_L1(d_pred_canon, d_gt_canon)      # 米制绝对锚
        + λ_grad · multiscale_gradient_L1                  # 边界锐利
        + λ_si   · SILog(d_pred_canon, d_gt_canon)         # 大量程训练稳定
        + λ_conf · uncertainty_NLL(d_pred, d_gt, conf)     # 学不确定性

所有项用 GT valid_mask 屏蔽无效像素（空洞 / 越界 / 反光）
起步权重（待调）: λ_l1=1.0, λ_grad=0.5, λ_si=0.5, λ_conf=0.2
```

- `SILog` 提供训练稳定性（纯米制 L1 在远距离易抖），但有 `valid_L1` 绝对锚，输出仍是米制，不退化为相对深度。
- `uncertainty_NLL` 用 Laplacian/Gaussian NLL（网络出 log-variance，confidence = exp(-σ)）：反光/玻璃/远处自动降置信，下游 ASM 据此丢弃或降权不可靠物体深度。

### 3.6 超参数（起步）

| 项 | 值 |
|---|---|
| Backbone | DINOv2 ViT-B/14（frozen） |
| 优化器 | AdamW，weight_decay=0.05 |
| 学习率 | base 2e-4，cosine decay，warmup 1–2 epoch |
| 精度 | BF16 混合精度 |
| 有效 batch | 32–64（PRO6000 单卡 + 梯度累积，见 §8.4） |
| 训练量 | ~100–200k iters（50–100 万帧，约 20–30 epoch） |
| 工作分辨率 | **固定 896×504（16:9，2304 tokens）**，不做多分辨率 |

### 3.7 验收 / 冻结 Adapter 判据

| 指标 | 原型目标 | 说明 |
|---|---|---|
| AbsRel ↓ | < 0.10 | 室内 metric 相对误差 |
| δ < 1.25 ↑ | > 0.90 | 内点率 |
| RMSE ↓ | 场景相关 | 米制 |
| 定性 | 边界锐利、无大面积塌陷 | |

验证集指标 plateau 后 **冻结 Adapter**，转入阶段1/2/3。

---

## 4. 阶段1：家居 Top-50 Detection Head（首轮重点）

### 4.1 结构

RF-DETR 式：固定 DINOv2 + Adapter 特征上接 transformer decoder。

```text
Object Queries Q_obj ∈ R^{Nq × C}  (Nq ≈ 300)
  → cross-attend to P2–P5
  → 6 层 Transformer Decoder
  → box head [cx,cy,w,h] / objectness / Top-50 class head
匹配: Hungarian (box + giou + class + objectness cost)
```

### 4.2 数据

| 来源 | 用途 |
|---|---|
| COCO / Objects365 / LVIS | 映射到家居 Top-50 类目 |
| 类目映射表 | 维护一份 {源数据集类 → Top-50} 映射，量控制在 50 类左右 |

50 类覆盖家具、家电、厨卫、日用品、门窗通道等高频物品；类目可随部署场景迭代。

### 4.3 损失与超参

```text
L_det = λ_box · L1(box) + λ_giou · GIoU + λ_obj · focal(objectness) + λ_cls · CE(top50)
起步权重: λ_box=5, λ_giou=2, λ_obj=1, λ_cls=1
优化器: AdamW，lr 1e-4（decoder），cosine，weight_decay 1e-4
训练量: ~50 epoch（DETR 类收敛偏慢；frozen backbone 下可适当缩短并加大 lr）
```

### 4.4 验收

| 指标 | 目标 |
|---|---|
| Top-50 常见物品**召回率** | > 70%（对应 07 文档 M2 里程碑） |
| mAP@[.5:.95] | 记录基线，观察 frozen backbone 上限 |

> ⚠️ 风险：全冻结 backbone 上做 DETR 式检测，mAP 可能受限（见 §11）。若召回达不到 70%，启用 §11 的 LoRA 兜底预案。

---

## 5. 阶段2：Bbox-based Mask Head

### 5.1 结构与训练方式

```text
GT bbox（训练）/ 检测 bbox（推理）
  → RoIAlign 从 P2 裁剪框内特征
  → 几层卷积 → mask: bbox_h × bbox_w 前景/背景二值
```

- **训练用 GT bbox**，与 Detection Head 解耦，可独立/并行训练；
- 推理时才依赖 Detection Head 的 bbox 输出。

### 5.2 数据与损失

```text
数据: COCO Panoptic / LVIS instance masks；无标注区用 SAM/SAM2 伪标签弱监督
L_mask = λ_bce · BCE + λ_dice · Dice   （起步 λ_bce=1, λ_dice=1）
优化器: AdamW lr 1e-4，~30 epoch
```

### 5.3 验收

框内前景 mask IoU 定性可用即可（下游只需 mask 内深度中位数定位物体，不追求实例分割精度）。

---

## 6. 阶段3：Room Semantics Head

> **Scope（重要）**：本头只输出**逐帧房间类型**（替代原先的在线 VLM 房间识别），**不负责"房间分区"**。房间分区（把空间切成有边界的房间实例）是**下游 GSM/地图模块**的职责——用 **门检测（检测头）+ 自由空间（深度）+ 帧级类型投票** 做几何分区（方案 A），见 [11 输入输出与 ASM 对比](11_DINO-GeoSemMap_输入输出与ASM对比.md)。本头是分区的**类型先验**，不是分区本身。

### 6.1 结构

```text
DINOv2 CLS token + GAP(P4/P5)  → MLP classifier → room_type + room_confidence
```

### 6.2 数据与损失

```text
数据: ScanNet / HM3D / Matterport3D / 合成数据，12 类房间（见 10 号数据集文档 §6.1）
L_room = λ_room · CE(room) + label_smoothing(0.1)
优化器: AdamW lr 1e-3（轻量 MLP），~20 epoch
```

过渡区域允许低置信度 / 多标签 soft prediction。

### 6.3 验收

常见房间 Top-1 准确率定性可用；过渡区不强求。

---

## 7. 数据准备总表

| 模块 | 起步数据量 | 较好效果 | 监督信号 | 主要来源 |
|---|---|---|---|---|
| Adapter + Depth | 50–100 万帧 | 200–500 万帧 | metric depth + valid mask + K | ARKitScenes / ScanNet++ / Hypersim / Matterport / **Habitat 仿真** |
| Detection | 10–30 万张 | 50–100 万张 | boxes + Top-50 class | COCO / Objects365 / LVIS |
| Mask | 5–20 万张 | 30–100 万张 | instance masks / SAM 伪标签 | COCO Panoptic / LVIS |
| Room | 5–20 万帧 | 20–80 万帧 | room type | ScanNet / HM3D / Matterport3D |

统一 dataloader 约定（建议）：每条样本 `{image, K, depth?, valid_mask?, boxes?, labels?, masks?, room?}`，按阶段取用对应字段。

### 7.1 存储估算

存储需求（数据集 / 特征缓存 / 模型产物 / 本地↔云分布 / 总量建议）已抽出为独立文档：**[09 存储规划](09_DINO-GeoSemMap_存储规划.md)**。

速查（数据集）：**最低数据量 ~0.3 TB；完整数据量 ~1.2–3.1 TB**（深度占大头）。详见 09。

---

## 8. 训练基础设施

### 8.1 硬件分配（本地 + 云）

**先本地打通、再上云扩量**：预验证轮 + 正式首轮**全程本地**；**待首轮四 head 都收敛后**，完整数据轮再把检测/分割/房间上云并行：

| 轮次 / 阶段 | 硬件 | 说明 |
|---|---|---|
| 预验证轮（L0/L1/L2） | 本地 1×PRO6000(96GB) | 全部 head 在本地验证、打磨框架 |
| **正式首轮（阶段0–3 全部）** | 本地 1×PRO6000(96GB) | 四 head 在本地**串行**训练，端到端打通统一框架；原型数据量，串行也只需天级 |
| 完整数据轮·深度 | 本地 1×PRO6000(96GB) | Adapter 已冻结；深度数据量最大，留本地 |
| 完整数据轮·检测/分割/房间 | 云平台（多卡） | 各 head 独立 → 云上**并行** + 多卡 DDP，压缩完整数据轮耗时 |

**本地 → 云 交接（完整数据轮前，关键）：**

- **门槛**：**首轮四 head 在本地均已收敛、Adapter 冻结**，才上云；
- 导出**冻结特征基座**（DINOv2 + Adapter）权重（~0.2–0.3 GB），连同检测/分割/房间数据集上云；
- ⚠️ **版本钉死**：云端各 head 必须基于与本地**逐字节一致**的冻结基座训练（ckpt hash 比对）——否则特征 ≠ 部署特征，**无损合并失效**（见 §1 铁律 / §11）；
- 云端喂数：① 直接跑 DINOv2+Adapter forward；② 预缓存 Adapter 特征(§8.3)再训 head（见 §7.1）。

**收益：** 首轮全本地 = 低风险端到端验证；完整数据轮检测/分割/房间上云并行，总周期 ≈ **深度(本地) + max(检测,分割,房间)(云)**。

### 8.2 训练配置

- **本地深度**：单卡（无 DDP）；**云端各 head**：视实例可单卡或多卡 DDP；
- 统一 BF16 混合精度（本地 Blackwell 优先 BF16）；
- backbone 冻结 → 无需对 backbone 做梯度 checkpoint，激活内存只来自 adapter + head；
- 用**梯度累积**凑有效 batch；定期 checkpoint + 验证集指标早停。

### 8.3 加速：缓存固定特征

阶段1/2/3 时 backbone + Adapter **全冻结**，可**预计算并缓存 Adapter 输出（P2–P5）**，head 直接在缓存特征上训练，省去每步 DINOv2 forward，大幅提速。

> 注意：分辨率已固定 896×504（无多分辨率），特征尺寸唯一，**缓存可直接复用**（不再有"多分辨率与缓存互斥"的问题），阶段1/2/3 缓存收益更明确。

### 8.4 显存预算（ViT-B/14，BF16）

**峰值出在阶段0（Adapter + Depth）。** 关键前提：DINOv2 全程冻结、以 `no_grad`/eval 当纯特征提取器跑——**backbone 激活不为反向保存，也无 backbone 梯度/优化器状态**，只有 ~20–35M 可训练参数（adapter + head）产生激活、梯度、优化器状态。所以显存大头是 **head 激活**，而非模型本身（与"全量微调 DINOv2"差一个量级）。

**阶段0 单卡显存拆解：**

| 组成 | 显存 | 说明 |
|---|---|---|
| 模型权重 | ~0.2 GB | backbone 86M + heads 20M，BF16 |
| 优化器 + 梯度 | ~0.3 GB | 只算可训练 ~20M（AdamW: FP32 master + m + v） |
| backbone forward | ~0.5–1.5 GB | no_grad，瞬时峰值，用完即释放 |
| **head 激活（主力）** | **~3.4 GB × batch** | DPT 稠密头是真正的开销（@896×504，∝像素） |
| PyTorch/CUDA 开销 | ~1–2 GB | context、内存碎片 |

**按 batch 估算单卡峰值（固定 896×504，2304 tokens ≈ 1.68× of 518²）：**

```text
batch 12:  ≈ 0.5 + 12×3.4 + 2  ≈  ~43 GB
batch 16:  ≈ 0.5 + 16×3.4 + 2  ≈  ~57 GB
batch 24:  ≈ 0.5 + 24×3.4 + 2  ≈  ~84 GB   （96GB 下偏紧）
```

> 分辨率固定，无多分辨率峰值波动；单样本 head 激活 ~3.4GB（896×504）。

**1×RTX PRO 6000（96GB）推荐配置：**

| 项 | 建议 |
|---|---|
| 单卡 batch | 12–16（@896×504），峰值 ~43–57GB，留 ~40GB headroom |
| 有效 batch | 单卡无 DDP；用**梯度累积**凑到有效 batch 32–64 |
| 阶段1/2/3 | 更省（backbone+adapter 全冻结，可缓存特征，见 §8.3），batch 可更大；单卡下**务必用缓存**压缩耗时 |

补充：

- **单卡无跨卡通信开销**——backbone 冻结本就让需同步的梯度只有 ~20M 参数，单卡则连 all-reduce 都省了，整训练全程在一张卡内。
- 96GB 显存对 ViT-B 绰绰有余（峰值 ~60–70GB），**留有上探 ViT-L 的空间**（ViT-L 需相应调小 batch）。
- 上表为基于"冻结 backbone + DPT 头"的标定估算，配 batch 足够；**真实峰值开训前跑一个 trial step `nvidia-smi` 实测确认**（PyTorch 内存分配/碎片会有出入）。

---

## 9. 超参数总表

| 阶段 | 优化器 | LR | Schedule | Epoch/Iter | 关键权重 |
|---|---|---|---|---|---|
| 0 Depth | AdamW (wd 0.05) | 2e-4 | cosine + warmup | ~100–200k it | l1=1, grad=.5, si=.5, conf=.2 |
| 1 Det | AdamW (wd 1e-4) | 1e-4 | cosine | ~50 ep | box=5, giou=2, obj=1, cls=1 |
| 2 Mask | AdamW | 1e-4 | cosine | ~30 ep | bce=1, dice=1 |
| 3 Room | AdamW | 1e-3 | cosine | ~20 ep | ce=1, ls=0.1 |

---

## 10. 时间规划与里程碑

### 10.1 估算口径与吞吐假设

关键事实：**backbone 冻结、只训 ~20–35M 的 head，收敛快**；时间主要由"数据量 × epoch ÷ 吞吐"决定，而非参数量。

| 阶段 | 训练吞吐(单卡, img/s, 在线) | epoch | 说明 |
|---|---|---|---|
| 0 Depth | ~40 | ~30 | 稠密 DPT 头 @896×504（2304 tok），最重 |
| 1 Detection | ~80 | ~50 | DETR 类收敛偏慢 |
| 2 Mask | ~120 | ~30 | RoI 小头 |
| 3 Room | ~300 | ~20 | MLP，极轻 |

```text
时间(天, 纯算力) ≈ 数据量 × epoch ÷ (吞吐 × 86400)
```

- 吞吐为保守中值（含冻结 backbone forward + head fwd/bwd + 增强/IO），实现不同 ±50%；
- **真实 wall-clock 另含调试/调参/重跑，通常 ×1.5–2**；
- 阶段1/2/3 用特征缓存(§8.3, 跳过 backbone forward)吞吐可再 ×2–3。

### 10.2 时间对比：最低数据量 vs 完整数据量

| 阶段 | 最低数据量 | 时间(算力) | 完整数据量 | 时间(算力) |
|---|---|---|---|---|
| 0 Depth | 50 万帧 | ~4–5 天 | 500 万帧 | ~43 天 |
| 1 Detection | 10 万张 | ~0.7 天 | 100 万张 | ~7 天 |
| 2 Mask | 5 万张 | ~0.2 天 | 100 万张 | ~3 天 |
| 3 Room | 5 万帧 | ~0.05 天 | 80 万帧 | ~0.6 天 |
| **合计** | **~5 天**（首轮·全本地串行） | | **~50 天**（完整轮·本地深度+云并行） | |
| 正式首轮（最低，全本地） | | **~5 天**（四阶段串行相加，但 1/2/3 极小） | | — |

> 口径（与 §8.1 硬件分段一致）：
> - **正式首轮 = 最低数据量、全本地 PRO6000 串行** → 合计 = 各阶段相加；因阶段1/2/3 极小，≈ 阶段0 深度 ~5 天；
> - **完整数据轮 = 深度本地 + 检测/分割/房间云并行** → 合计 ≈ **深度(本地) + max(检测,分割,房间)(云)** ≈ 深度主导 ~43–50 天；
> - 深度按 **896×504（吞吐 ~40 img/s）**；含 ×1.5–2 wall-clock 余量：**首轮 ~1.5 周；完整数据轮 ~10–14 周**。

观察：

- **首轮（最低数据、全本地）约一周出结果**（含调试 ~1.5 周）——单卡串行也够快（阶段1/2/3 仅天内/小时级），低风险端到端验证；
- **瓶颈几乎全在阶段0 深度**（本地、896×504 稠密）——完整数据量下 ~43 天，占绝对大头；
- 完整数据轮上云后，检测/分割/房间**并行**且与本地深度训练**重叠**进行（Adapter 首轮已冻结），故几乎不增加总周期；云端多卡 DDP 还能再压。

### 10.3 里程碑（正式首轮：最低/原型数据量；**全本地 PRO6000 串行**）

| 里程碑 | 时间点 | 交付物 | 验收标准 |
|---|---|---|---|
| M0 数据就绪 | 第1–2周 | 各任务数据 + dataloader | 格式校验通过，dataloader 跑通；深度数据带 K（不占 GPU，**数据采集/建库常是真正瓶颈**） |
| M1 Depth 可用 | 第2–3周 | Adapter + Depth 训完 | AbsRel<0.1、δ1>0.9，多分辨率/多相机定性稳定，Adapter 冻结 |
| M2 云端三头可用 | 第3–4周 | Detection / Mask / Room 训完（云上并行） | 检测 Top-50 召回 > 70%；分割框内 IoU 定性可用；房间 Top-1 定性可用 |
| M3 全链路打通 | 第4–5周 | 四个 head 就绪 | 单帧 → 感知输出 → ASM 合成全流程跑通 |
| M4 ASM 集成验证 | 第5–7周 | 感知 + SLAM + ASM 联调 | 多帧序列 ASM 稳定，物品 3D 位置误差可接受 |

> 上表按最低/原型数据量（即"正式首轮"）。**完整数据量（500万帧）下仅阶段0 深度即扩到 ~43 天（算力，@896×504）/ ~10–14 周（含余量）**，M1 及后续相应后移。
> 整体按 §2.1 三段推进：**预验证轮（小/中量，天级，验证框架+效果）→ 正式首轮（原型，~1.5 周，打通链路）→ 完整数据轮（完整，~10–14 周，提质）**。预验证轮的算力/存储很轻，是降低大数据踩坑成本的关键前置。

---

## 11. 风险与缓解

| 风险 | 说明 | 缓解 |
|---|---|---|
| **Frozen backbone 检测上限** | 全冻结 backbone 上 DETR 式检测 mAP 可能受限，70% 召回偏乐观 | 兜底：对 backbone 加 **LoRA / 解冻最后若干层**，仅供 Detection 用（会破坏"完全共享冻结特征"，需评估对其他 head 的影响） |
| **Adapter 被 Depth 任务锚定** | Adapter 在阶段0 随 Depth 训练定型，可能不利于 detection/room 的语义特征 | 监控阶段1/3 收敛；必要时给各 head 配轻量私有 adapter，或阶段0 加入任务无关/多任务约束 |
| **单目 metric 跨相机泛化** | Canonical Transform 依赖 K 准确 + 训练焦距分布覆盖 | 阶段0 强制多焦距/多分辨率覆盖；用双目 teacher 补真实相机域 |
| **多分辨率范围锁死** | Adapter 冻结后扩分辨率需重训 | 阶段0 一次喂满目标分辨率/焦距区间 |
| **SAM 伪标签噪声** | Mask 弱监督质量参差 | 优先真值数据，伪标签做置信度过滤 |

---

## 12. 待确认 / 后续细化

- [ ] `f_c` 标准焦距与工作分辨率档位的具体取值（依实际部署相机分布定）；
- [ ] 深度量程 `[d_min, d_max]`（室内初拟 0.2–8m）与 `valid_mask` 阈值 τ；
- [ ] Top-50 类目映射表（源数据集 → 50 类）定稿；
- [ ] 各阶段损失权重的实测调参。
