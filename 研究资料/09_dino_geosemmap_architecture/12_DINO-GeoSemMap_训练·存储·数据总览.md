
## 1. 关键固定参数

| 项                 | 值                                            |
| ----------------- | -------------------------------------------- |
| Backbone          | DINOv2 ViT-B/14（**全程冻结**；当前选型，可替换）           |
| 工作分辨率             | **固定 896×504**（16:9，2304 tokens），不做多分辨率      |
| 相机                | 120°×85° → 去畸变针孔，**裁剪到 ~90° HFOV（大部分数据集尺寸）** |
| 内参（1280×720）      | `fx=fy=640, cx=640, cy=360`；机位高 **1.15m**    |
| f_c（标准焦距，896×504） | **448 px**（Canonical Transform 反归一化用）        |
| 深度量程              | 0.2–5m（米制，单帧）                                |

---

## 2. 训练

**结构**：DINOv2(冻结) → Adapter → {Depth, Detection, Mask, Room}。Adapter 在阶段0 与 Depth 联训后**冻结**，成为各 head 共享的固定特征基座；各 head 独立训练、可无损合并成单模型。

| 阶段 | 头 | 要点 |
|---|---|---|
| 0 | Adapter+Depth | 单帧 metric，DPT 头，Canonical Transform 多相机 |
| 1 | Detection | RF-DETR 式，家居 30 类起步（可扩 Top-50） |
| 2 | Mask | 框内前景/背景二值，GT bbox 训练 |
| 3 | Room | CLS+GAP→MLP，**12 类房间类型**（仅帧级类型，不做分区） |

**推进（3 轮，每轮确认全部 4 head）**：预验证轮（小/中量，调框架）→ 正式首轮（原型，打通链路）→ 完整数据轮（提质）。

**硬件（先本地打通、再上云扩量）**：预验证轮 + 正式首轮**本地 1×PRO6000(96GB)**（四 head 串行训练验证）；**待首轮四 head 都收敛后**，完整数据轮再 **深度留本地、检测/分割/房间上云并行**（Adapter 已冻结、各 head 独立）。

**显存**（@896×504）：单卡 batch 12–16，峰值 ~43–57GB；深度吞吐 ~40 img/s。

---

## 3. 数据集构成

> 主力 = 公开真实数据；**Habitat（按部署内参从零渲）补 in-domain 宽视角 + 一次出四 head 标签**。

| Head | 主力来源 | in-domain 补充 |
|---|---|---|
| 深度 | ARKitScenes / ScanNet++ / Hypersim / NYU | Habitat ~20% |
| 检测 | COCO/O365/LVIS→30类 | ARKit3D框/SUN RGB-D/ScanNet投影 + Habitat |
| 分割 | SAM/SAM2 伪标签 + COCO/LVIS mask | Habitat 实例 mask |
| 房间 | MP3D(region)/Structured3D/HM3D | Habitat 房型 |

- **映射表**：房间 12 类（MP3D 30 region → 12，见 10 §6.1）；物品 30 类 {COCO/O365/LVIS → 统一类}（10 §6.2）；
- **in-domain eval**：现 Habitat held-out（sim→sim）；中期补**小真机 eval**（sim→real 体检）；

---

## 4. 存储

| 场景 | 本地 PRO6000 | 云 |
| --- | --- | --- |
| 预验证轮 | <0.1 TB（四 head 全部，全本地） | — |
| 正式首轮 | ~0.3–0.65 TB（**四 head 全部，全本地**） | — |
| 完整数据轮 | ~1.0–2.5 TB（深度） | ~0.2–0.6 TB（检测/分割/房间） |

> 与 §2 硬件分段一致：首轮全本地 → 数据全在本地；完整数据轮才把检测/分割/房间数据搬上云。

- **建议**：本地 ~4TB NVMe，云 ~1TB；
- **特征缓存**慎用（P2–P5 ~15MB/图，完整量级达十几 TB，不划算 → 云端直接 re-forward）；
- **模型产物** GB 级：冻结基座导出 ~0.2–0.3GB；最终单模型 **~230MB（v0.3-B FP16）**。

---
