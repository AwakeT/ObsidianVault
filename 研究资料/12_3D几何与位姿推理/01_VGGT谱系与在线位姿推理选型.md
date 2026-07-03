# VGGT 谱系与在线位姿推理选型

> 版本：v0.1（调研 + 选型结论）
> 日期：2026-07-02
> 服务对象：DINO-GeoSemMap v-next.2「轻量流式几何专家」（[[21_DINO-GeoSemMap_v-next_总架构愿景]]、`DINO-GeoSemMap_v-next.2_端侧主干专家架构.drawio`）
> 结论先行：**P0a 现成在线基线 = CUT3R（+ 训练-free 的 TTT3R 增强）**；漂移不行再换 Anchor3R / RayMap3R。

---

## 0. 一句话背景

VGGT（CVPR 2025 Best Paper）把 SfM/BA 换成一个前馈 Transformer，一次前向出相机位姿 + 深度 + 稠密点云 + 点跟踪。但它的三个结构性痛点定义了整条后续脉络：

1. **global attention O(N²)** → 序列一长显存/耗时爆炸（>300 帧 OOM）；
2. **参数大、单次前向** → 部署贵、且**天然离线**（整段集合式，不能随动）；
3. **锚定参考帧** → 对首帧选择敏感、对输入顺序不鲁棒。

> ⚠️ 关键：**VGGT 本体没有在线推理模式**——它必须一次吃完所有帧联合推理。要"伴随机器人运动、逐帧增量出位姿"，得用它的**在线/流式后裔**。

---

## 1. 精度轴：把位姿/几何做得更准更稳

| 工作 | 优化点 | 收益 |
|------|--------|------|
| **π³ (Pi3)** ICLR 2026 · [arXiv:2507.13347](https://arxiv.org/abs/2507.13347) | 去掉参考帧，全 permutation-equivariant；每帧预测 affine-invariant 位姿 + scale-invariant 局部点图 | 对输入顺序近零方差（DTU acc std 0.003 vs VGGT 0.033），位姿/深度/点云全面 SOTA；Pi3X 支持近似 metric scale |
| **DINOv3-VGGT** · [DINOv3](https://arxiv.org/pdf/2508.10104) | 换更强 backbone | 相机位姿/多视深度/视图匹配一致提升 |
| **VGGT-Long** ICRA 2026 · [arXiv:2507.16443](https://arxiv.org/abs/2507.16443) | chunk + overlap 对齐 + 轻量回环 | 公里级无界室外，全局一致 |

> π³ 虽强，但**非因果（整段集合式）**，不能直接在线；其"去参考帧"思想我们只**借用**来提鲁棒。

---

## 2. 效率轴：省显存 / 降算力 / 提速

| 工作 | 手段 | 收益 |
|------|------|------|
| **FastVGGT** ICLR 2026 · [arXiv:2509.02560](https://arxiv.org/abs/2509.02560) | training-free token merging | 1000 帧 4× 加速，缓解长序列误差累积 |
| **FlashVGGT** · [arXiv:2512.01540](https://arxiv.org/pdf/2512.01540) | compressed descriptor attention | 长序列比 CUT3R 快 3.3×+ |
| **QuantVGGT** · [arXiv:2509.21302](https://arxiv.org/html/2509.21302) | 低比特量化 W4A4 | 大幅省显存/算力，几乎不掉精度 |
| **Sparse-VGGT** | block-sparse attention kernel | 约 4× |

> 这些都在优化 VGGT **本体**，仍是离线/整段——是"端侧压缩"阶段（P0b 之后）的工具箱，不是在线基线。

---

## 3. ⭐ 在线 / 流式轴（能随动增量的——P0a 的候选池）

按"如何保留历史"分三派，直接决定精度—显存权衡：

- **循环状态（RNN 式）**：Spann3R、**CUT3R** —— 常量隐藏态，逐帧更新，恒定显存，可长跑。
- **显式指针记忆**：**Point3R** —— 精度更好，但 >700 帧 OOM。
- **因果注意力 + KV cache**：**StreamVGGT**（VGGT 蒸馏）、STream3R —— 近离线精度，但 KV 无界、长序列 OOM。

### 候选对比

| 模型 | 记忆机制 | 在线 | 常量显存 | 在线 metric 深度 | 主要短板 |
|------|---------|:---:|:---:|:---:|------|
| **CUT3R** · [页](https://arxiv.org/html/2509.26645v3) | 隐式循环记忆 | ✅ | ✅ | ✅ | 长序列漂移/灾难性遗忘 |
| **TTT3R** · [arXiv:2509.26645](https://arxiv.org/pdf/2509.26645) | CUT3R + 推理期状态更新规则（训练-free） | ✅ | ✅ | ✅ | 受 CUT3R 上限约束；**~2× 精度、实时** |
| **Point3R** · [arXiv:2507.02863](https://arxiv.org/pdf/2507.02863) | 显式点指针 | ✅ | ~ | ✅ | >700 帧 OOM |
| **StreamVGGT** | KV cache（VGGT 蒸馏） | ✅ | ✗ | 相对深度 | KV 无界、长序列 OOM、延迟高 |
| **Spann3R** | 空间记忆（DUSt3R 系） | ✅ | ✅ | — | 静态场景假设、较早 |
| **WinT3R** · [arXiv:2509.05296](https://arxiv.org/pdf/2509.05296) | 滑窗 + 相机 token 池 | ✅ | ✅ | — | 感受野受限，可能漂移 |
| **Anchor3R**（新，约 1 月）· [arXiv:2606.05035](https://arxiv.org/html/2606.05035) | current-centric 局部锚 | ✅ | ✅ | — | 新、验证少；7Scenes 超 CUT3R/TTT3R/STream3R/StreamVGGT |
| **RayMap3R**（2026）· [arXiv:2603.20588](https://arxiv.org/pdf/2603.20588) | reset 对齐 + 状态平滑 | ✅ | ✅ | — | 专治漂移(ATE)，动态场景 |

> 全场只有 **CUT3R / Point3R / TTT3R 出在线 metric 深度**——这对"深度锚定位姿尺度"很关键。

---

## 4. 对 DINO-GeoSemMap v-next.2 的选型结论

架构已定：语义主干 + **轻量流式几何专家**（单向注入语义），位姿=世界系 T_t 逐帧在线，深度 5–10Hz 锚尺度，P0 两步走（P0a 现成验可用性 → P0b 蒸馏到端侧）。

**P0a 现成基线 = CUT3R（+ TTT3R）**，理由与决策一一对应：

| 决策 | CUT3R/TTT3R |
|------|-------------|
| 逐帧在线 + 常量显存（端侧红线） | ✅ RNN 循环态，可无限长跑 |
| 世界系 T_t | ✅ 原生演化世界系，每帧出相机位姿 + pointmap |
| 深度锚尺度 | ✅ 在线 metric 深度 |
| 几何专家用 CUT3R 循环态（决策 #4） | ✅ P0a 现成件与 P0b 目标形态同构 |

- **TTT3R**：训练-free、套 CUT3R 权重、~2× 精度、保持实时，专压 CUT3R 长序列漂移 → 零成本兜底。
- **漂移 plan B**：Anchor3R（最新在线 SOTA）/ RayMap3R（压 ATE）。
- **P0 闸门需外部参照**：家居序列跑 MASt3R-SLAM / DROID-SLAM 当伪 GT，或用带 GT 位姿数据（ScanNet / TUM RGB-D / 7-Scenes / Habitat 仿真）。→ 对应开放项 C（位姿监督与评测），P0a 判闸前必须先定。

---

## 5. 开放项（待续）

- **C. P0a 的 GT / 评测怎么搭**：伪 GT（SLAM）还是 GT 数据集；VO 漂移用 ATE/RPE 怎么对标；家居数据从哪来。
- **B. 端侧有界记忆机制**：循环态（CUT3R）/ 指针（Point3R）/ 压缩 KV（XStreamVGGT）终选。
- **P0b 蒸馏方向**：收缩 CUT3R vs 以 VGGT 为教师蒸馏一个小因果学生。

---

## 关联

- [[21_DINO-GeoSemMap_v-next_总架构愿景]]
- `../09_dino_geosemmap_architecture/DINO-GeoSemMap_v-next.2_端侧主干专家架构.drawio`
