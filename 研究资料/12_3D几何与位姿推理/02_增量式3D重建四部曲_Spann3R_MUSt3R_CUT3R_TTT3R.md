# 增量式在线 3D 重建四部曲：Spann3R → MUSt3R → CUT3R → TTT3R

> 版本：v0.1（论文笔记 · 提取自 CSDN 博客 + arXiv 原文核实）
> 日期：2026-07-02
> 来源博客：https://blog.csdn.net/m0_60177079/article/details/159048989
> 服务对象：[[01_VGGT谱系与在线位姿推理选型]]、DINO-GeoSemMap v-next.2 几何专家（P0a 现成基线 = CUT3R+TTT3R）

---

## 0. 共同背景：为什么是「增量式」

四篇都长在 **DUSt3R** 范式上（前馈网络直接回归 pointmap，无需相机内参/位姿），要解决 DUSt3R 的两个痛点：

- **成对 → O(N²)**：DUSt3R 只处理图像对，N 张图要两两配 + 离线全局对齐（optimization-based global alignment），规模爆炸。
- **离线 → 不能随动**：全局方法（含 VGGT）要一次吃完所有帧。

**增量式（online / RNN 式）** 的解法：维护一个**定长记忆状态**，逐帧因果处理（不看未来帧），每帧更新状态并直出该帧在**全局坐标系**下的 pointmap/位姿，复杂度约 O(1)/帧、显存恒定，可长跑。代价是**长序列漂移/遗忘**（TTT3R 专治这个）。

| 维度 | 全局（DUSt3R对 / VGGT） | 增量（本文四篇） |
|---|---|---|
| 处理方式 | 一次吃完 / 成对 + 全局对齐 | 逐帧因果、流式 |
| 复杂度 | O(N²) | ~O(1)/帧，恒定显存 |
| 随动在线 | ✗ | ✅ |
| 主要短板 | 显存/延迟、离线 | 长序列漂移/遗忘 |

---

## 1. Spann3R —— 给 DUSt3R 加"外部空间记忆"

| 项目 | 内容 |
|---|---|
| 标题 | **3D Reconstruction with Spatial Memory** |
| 作者/机构 | Hengyi Wang, Lourdes Agapito（UCL） |
| 会议 | 3DV 2025；[arXiv:2408.16061](https://arxiv.org/abs/2408.16061)；[项目页](https://hengyiwang.github.io/projects/spanner) |

**动机**：DUSt3R 每对图出的是**局部坐标系**下的 pointmap，拼全景要跑昂贵的全局对齐优化。Spann3R 想**直接在全局系里逐帧出 pointmap**，去掉全局对齐。

**方法要点**：
- 核心是一个**外部空间记忆（external spatial memory）**：记住此前见过的 3D 信息，新帧来时用 **cross-attention 读记忆**，预测该帧在全局系下的结构。
- 记忆管理：`check_sim` 判相似度避免冗余入库；`memory_prune` 保留 top-k 重要 token（long memory 上限约 4000）。
- 训练：DUSt3R 累积置信度加权损失 + 单边 hinge 尺度损失；复用 DUSt3R 预训练权重再微调。

**结果/定位**：有序图像流可**实时**处理，跨未见数据集泛化尚可。是**最早一批**真·流式方法之一（与 CUT3R 并列），但基于静态场景假设、精度受 DUSt3R 上限约束。

---

## 2. MUSt3R —— DUSt3R 的对称多视 + 工作记忆

| 项目 | 内容 |
|---|---|
| 标题 | **MUSt3R: Multi-view Network for Stereo 3D Reconstruction** |
| 作者/机构 | Cabon, Stoffl, Antsfeld, Csurka, Chidlovskii, Revaud, Leroy（Naver，DUSt3R/MASt3R 同组） |
| 会议 | CVPR 2025；[arXiv:2503.01661](https://arxiv.org/abs/2503.01661) |

**动机**：同样冲着 O(N²) 成对 + 离线全局对齐，但走"**把 DUSt3R 从成对扩到多视**"的路。

**方法要点**：
- **对称架构**：改 DUSt3R 为对称设计，直接为**所有视图**预测共享坐标系下的 3D 结构。
- **多层工作记忆（multi-layer memory）**：把复杂度压下来，可"高帧率推上千张 pointmap"。
- 相对 Spann3R 更简：**单解码器**（Spann3R 用两个），只做一次 cross-attention；对每个非参考帧加 **prefix token** 做因果 cross-attention。
- 训练：log-space 回归损失；增量训练用 N=10 帧序列。
- **离线/在线双模式**，可用于 SfM 与 visual SLAM。

**结果**：在无标定 VO、相对位姿、尺度/焦距估计、3D 重建、多视深度上报 SOTA。定位：**可扩展到大集合**的多视统一网络，兼顾离线与在线。

---

## 3. CUT3R —— 可持续状态 + 生成/想象能力

| 项目 | 内容 |
|---|---|
| 标题 | **Continuous 3D Perception Model with Persistent State**（项目名 CUT3R = Continuous Updating Transformer for 3D Reconstruction） |
| 作者/机构 | Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa（UC Berkeley） |
| 会议 | CVPR 2025；[arXiv:2501.12387](https://arxiv.org/abs/2501.12387)；[项目页](https://cut3r.github.io) |

**动机/亮点**：不止增量重建，还能**推理/想象**——单图 3D、预测未观测区域的 3D 点与颜色、场景补全、动态场景重建。一个统一框架吃多种 3D/4D 任务。

**方法要点（架构）**：
- **可学习状态 s₀（记忆）** + **可学习位姿 token z**；**两个解码器**（State Update 状态更新 / State Readout 状态读出）互换 q/k/v。
- **三个头**：local pointmap、global pointmap、pose。
- 从图像流**在线**出**米制尺度（metric）**的 pointmap，落在共享坐标系，累积成随新帧增长的稠密重建。
- **新视角推理**：用虚拟相机的 ray-map encoder，在未观测视角"探针"出结构。
- 输入灵活：有序视频流 / 无序图集、静态 / 动态皆可。
- 训练：32 个混合数据集、四个渐进阶段（末阶段冻结 image encoder，在 4–64 视图上训）。

**结果/定位**：多项 3D/4D 任务上有竞争力或 SOTA；是**在线 metric 深度**的少数几个之一。已知短板：**训练上下文 ≤64 帧，上千帧会因状态过度适应近期观测而遗忘早期历史 → 位姿漂移/几何破碎**（TTT3R 要修的正是这个）。

---

## 4. TTT3R —— 训练-free 修复 CUT3R 的长序列遗忘

| 项目 | 内容 |
|---|---|
| 标题 | **TTT3R: 3D Reconstruction as Test-Time Training** |
| 作者/机构 | Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, Anpei Chen（Tübingen / Westlake 等） |
| 会议 | ICLR 2026；[arXiv:2509.26645](https://arxiv.org/abs/2509.26645)；[代码](https://github.com/Inception3D/TTT3R) |

**动机**：RNN 式重建虽线性复杂度，但**超出训练上下文长度就明显退化**（弱 length generalization）——CUT3R 在上千帧上漂移。

**方法要点**：
- 把重建重构为 **Test-Time Training / 在线学习**问题：用**记忆状态与新观测的对齐置信度**，导出一个**闭式学习率 β_t**，在推理时自适应调节记忆更新——平衡"保留历史 vs 适应新帧"。
- CUT3R 原更新：`S_t = S_{t-1} + softmax(Q_{S_{t-1}} K_{X_t}ᵀ) V_{X_t}`；TTT3R 用 per-token 自适应率 β_t 门控之。
- **即插即用、无需训练**（套 CUT3R 权重，不改底座、不重训）。

**结果**：全局位姿估计 **~2× 提升**；**20 FPS / 仅 6GB 显存**，可处理上千张图。定位：CUT3R 的**零成本长序列兜底**。

---

## 5. 演进脉络（一句话串起来）

**Spann3R**（DUSt3R + 外部空间记忆，去全局对齐）→ **MUSt3R**（对称多视 + 工作记忆，单解码器更省，可扩展大集合）→ **CUT3R**（可学习循环状态 + 位姿 token + 生成/新视角 + 在线 metric）→ **TTT3R**（训练-free 的推理期状态更新，修 CUT3R 长序列遗忘）。

| 模型 | 记忆机制 | 解码器 | 位姿 | 在线 metric 深度 | 一句话 |
|---|---|---|:---:|:---:|---|
| Spann3R | 外部空间记忆(top-k≤4000) | 2 | 隐式 | — | 最早流式，去全局对齐 |
| MUSt3R | 多层工作记忆 | 1 | ✅(VO/rel-pose) | — | 对称多视、可扩展、离线+在线 |
| CUT3R | 可学习循环状态 s₀+位姿 token z | 2(更新/读出) | ✅(pose 头) | ✅ | 状态化 + 生成/想象 |
| TTT3R | CUT3R + 自适应 β_t 更新 | 承 CUT3R | ✅ | ✅ | 训练-free 治漂移，2×、20FPS/6GB |

---

## 6. 对 DINO-GeoSemMap v-next.2 的意义（选型落点）

- **P0a 现成基线仍是 CUT3R + TTT3R**：CUT3R 提供状态化 + 在线 metric 深度 + 世界系位姿，与我们几何专家「CUT3R 式循环态、世界系 T_t、深度锚尺度」的决策同构；TTT3R 零成本压漂移（20FPS/6GB 也友好）。本笔记补全了它俩的**上游谱系与内部机理**。
- **MUSt3R 值得留意**：单解码器 + 工作记忆的**对称多视**设计，若 P0b 蒸馏要在"多视一致性 vs 端侧轻量"间取舍，MUSt3R 的结构比 CUT3R 更省，可作蒸馏/结构参考。
- **Spann3R 的记忆管理**（check_sim 去冗余、memory_prune top-k）对应我们的**有界记忆红线**（开放项 B）——是"如何把记忆卡在有界预算"的具体范式之一。
- 与 [[01_VGGT谱系与在线位姿推理选型]] 的对比表互补：那篇横向比性能/显存，这篇纵向讲谱系/机理。

---

## 关联

- [[01_VGGT谱系与在线位姿推理选型]]
- [[00_索引]]
- `../09_dino_geosemmap_architecture/DINO-GeoSemMap_v-next.2_端侧主干专家架构.drawio`
