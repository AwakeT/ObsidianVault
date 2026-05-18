---
title: "AgentVLN: Towards Agentic Vision-and-Language Navigation"
method_name: "AgentVLN"
authors: [Zihao Xin, Wentong Li, Yixuan Jiang, Ziyuan Huang, Bin Wang, Piji Li, Jianke Zhu, Jie Qin, Shengjun Huang]
year: 2026
venue: arXiv
tags: [vision-language-navigation, vlm-agent, embodied-ai, posmdp, skill-library, cross-space-mapping, qd-pcot, edge-deployment]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2603.17670v1
created: 2026-05-18
---

# 论文笔记：AgentVLN: Towards Agentic Vision-and-Language Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Nanjing University of Aeronautics and Astronautics, Shandong University, Zhejiang University |
| 日期 | 18 Mar 2026 |
| 项目主页 | N/A |
| 代码 | [GitHub](https://github.com/Allenxinn/AgentVLN) |
| 对比基线 | [[InternVLA-N1]], [[DualVLN]], [[EfficientVLN]], [[StreamVLN]], [[JanusVLN]], [[NaVILA]], [[NaVid]], [[ETPNav]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.17670) / [PDF](https://arxiv.org/pdf/2603.17670) |

---

## 一句话总结

> AgentVLN 将 VLN 建模为 [[POSMDP]]，用轻量 [[Qwen2.5-VL]]-3B 作为高层“大脑”调度感知/规划技能，通过 3D waypoint 到 2D 像素提示的跨空间映射、上下文自纠错和 QD-PCoT 主动感知，在 R2R-CE 与 RxR-CE Val-Unseen 上超过更大的 SOTA 模型，并支持 Jetson 边缘端部署。

---

## 核心贡献

1. **VLM-as-Brain agentic VLN 框架**: 将长程导航从端到端动作预测改为 VLM 调度技能库，解耦高层语义推理和低层感知/控制执行
2. **跨空间表示映射**: 将感知层生成的 3D 拓扑 waypoint 投影到当前图像平面，形成 VLM 能直接理解的像素级视觉提示，缓解 2D VLM 与 3D 几何之间的表示断裂
3. **上下文驱动自纠错与主动探索**: 当当前视野没有可用全局路径提示或出现遮挡/偏航时，VLM 输出细粒度原子动作进行局部恢复
4. **QD-PCoT（Query-Driven Perceptual Chain-of-Thought）**: 遇到空间尺度歧义时，模型主动生成自然语言查询并调用感知技能补充深度/语义信息，再进行目标像素定位
5. **AgentVLN-Instruct 数据集**: 构建带动态阶段路由、技能调用、定位推理和主动问答的指令微调数据，用于对齐高层指令与低层技能调用

---

## 问题背景

### 要解决的问题

VLN 要求智能体在未见环境中，根据自然语言指令和自中心观测完成长程连续导航。现代 [[VLM]] 对 2D 图像语义理解很强，但直接部署到 3D 物理世界时会遇到空间感知不足、尺度歧义、路径误差累积和实时部署成本高等问题。

### 现有方法的局限

- **单系统端到端方法**：用大量视频轨迹数据学习从视觉/语言到动作块的黑箱映射，容易缺少显式几何结构，跨环境泛化受限
- **双系统方法**：用 VLM 做低频全局规划，再用扩散/轨迹模块生成低层路径，但 3D 轨迹和 VLM 的 2D 视觉表示之间存在跨空间断裂
- **额外深度/3D模块成本高**：一些方法引入深度估计或 3D 视觉编码器来补空间信息，但会增加推理延迟、上下文长度和显存负担
- **单目尺度歧义**：仅凭 RGB 难以理解“近处/远处/左前方/桌子旁边”等带空间关系的指令，局部目标定位容易偏移
- **边缘端部署困难**：7B 以上模型与外接几何模块常依赖云端推理，不满足真实机器人高频控制需求

### 本文的动机

与其让 VLM 隐式学习完整 3D 几何和低层控制，不如让 VLM 专注于高层语义-视觉匹配和技能调度；把几何建图、路径规划、深度查询和机器人控制封装为可插拔技能，通过显式的 2D-3D 映射把 VLM 能理解的像素空间与机器人可执行的物理空间连接起来。

---

## 方法详解

### 模型架构

AgentVLN 采用 **VLM-as-Brain + 可插拔技能库** 架构：

- **任务建模**: [[POSMDP|Partially Observable Semi-Markov Decision Process]]
- **VLM 大脑**: [[Qwen2.5-VL]]-3B
- **输入**: 自中心 RGB-D 观测、历史上下文、自然语言导航指令
- **技能库**:
  - 感知级技能 $\mathcal{F}_{percep}$：建图、深度/几何查询、相机处理、语义/几何增强观测
  - 规划级技能 $\mathcal{F}_{plan}$：路径生成、局部避障、机器人运动控制
- **核心桥接**: 3D occupancy/topological waypoint $\rightarrow$ 当前图像上的像素级视觉提示
- **执行流程**: 全局路径导航阶段调用感知技能生成可视路径提示；局部目标定位阶段调用 QD-PCoT 查询缺失空间信息并回投 3D 目标点
- **真实部署**: [[Unitree Go2]] + [[Intel RealSense]] D455 + [[RTAB-Map]] + Jetson 边缘端推理

### 核心模块

#### 模块1: POSMDP 技能调度

**设计动机**: VLN 中的动作并不一定要是低层连续速度或离散方向，可以是持续多个时间步的技能调用。用半马尔可夫过程建模能自然表达“调用一次技能，执行一段闭环控制，再由 VLM 重新决策”。

**具体实现**:

- 状态 $s \in \mathcal{S}$ 编码机器人全局位置与姿态
- 观测 $o_t \in \mathcal{O}$ 来自机载传感器
- 动作空间替换为技能函数集合 $\mathcal{F}=\{f_1,\ldots,f_K\}$
- VLM 根据历史 $\mathcal{H}_{t_k}$、当前观测 $o_{t_k}$ 和指令 $\mathcal{I}$ 选择要调用的技能
- 感知技能执行时间 $\tau=0$，返回增强观测 $\Delta o$
- 规划技能执行时间 $\tau>0$，内部产生低层动作序列，并由终止条件 $\beta_f$ 决定是否回到 VLM 决策周期

#### 模块2: Cross-Space Representation Mapping

**设计动机**: 双系统 VLN 中，规划器输出的是 3D waypoint，而 VLM 主要理解 2D 图像 token。直接让 VLM 选择 3D 路径会造成表示不一致。本文把 3D 可行路径点投影回当前相机图像，用像素级提示让 VLM 在熟悉的 2D 空间做语义选择。

**具体实现**:

- 使用 RGB-D 与相机位姿将像素反投影到世界坐标，增量构建 occupancy grid
- 规划级技能在 3D 地图中生成全局可行路径点
- 将世界坐标中的路径点通过相机外参和内参投影到当前图像平面
- 在图像上用绿色点等视觉 prompt 标注候选路径，让 VLM 根据语言指令选择语义匹配方向
- VLM 选择像素 prompt 后，再由规划技能恢复到 3D waypoint 并执行导航

#### 模块3: Context-Driven Fine-Grained Self-Correction / Active Exploration

**设计动机**: 长程导航中常出现遮挡、盲区、转角、通道狭窄、路径提示不可见等情况。如果强制执行长距离路径，会放大误差。

**具体实现**:

- 当当前视野没有与指令语义相关的全局路径投影时，VLM 不直接盲走
- VLM 根据历史上下文和当前观测输出细粒度原子动作：`Forward`、`Left`、`Right`
- 这些动作用于看向周围、重新捕获可行路径提示、纠正偏航
- 一旦重新发现有效全局路径映射，系统恢复“感知技能 - 规划技能”的高效长程执行

#### 模块4: QD-PCoT 局部目标定位

**设计动机**: 到达目标附近后，导航从“全局路径选择”切换为“局部目标精确 docking”。此时指令中的空间关系往往需要深度和几何线索，纯 RGB VLM 容易产生尺度幻觉。

**具体实现**:

- VLM 检测到空间歧义时，不直接输出目标坐标
- 它先生成自然语言查询，例如“前方椅子距离我多少米？”
- 系统调用对应的感知级技能获取深度/几何/语义反馈
- 反馈作为文本增量提示写回上下文
- VLM 基于多轮感知 CoT 输出符合指令意图的目标像素坐标
- 系统从深度图取该像素深度，反投影到相机坐标，再变换到世界坐标并调用规划技能完成 docking

#### 模块5: AgentVLN-Instruct

**设计动机**: 原始 VLN 数据集通常只提供粗粒度指令-waypoint 对齐，不能直接训练 VLM 在 VLM-as-Brain 架构下学会“何时调用技能、调用什么技能、如何主动提问、如何做局部定位”。

**具体实现**:

- 基于 [[Habitat]] 构建大规模指令微调数据
- 引入 target-visibility-driven dynamic routing：目标不可见时进行全局路径导航，目标可见时切换到局部定位
- 全局阶段训练模型调用感知技能选择专家对齐 waypoint
- 注入轨迹噪声，用细粒度动作序列作为视觉 waypoint 缺失时的 fallback
- 局部阶段构造多轮 perceptual CoT，用主动问答模拟定位推理
- 训练中混合 [[LLaVA]]-Video-178K 以保留通用多模态能力

---

## 关键公式

### 公式1: 技能选择策略

$$
c_k \sim \pi_{\theta}(f \mid \mathcal{H}_{t_k}, o_{t_k}, \mathcal{I}), \quad f \in \mathcal{F}
$$

**含义**: 在第 $k$ 个决策时刻，VLM 根据历史上下文、当前观测和指令选择一个技能函数。

**符号说明**:

- $c_k$: 当前决策周期选择的技能调用
- $\pi_{\theta}$: 由 VLM 参数化的高层策略
- $\mathcal{H}_{t_k}$: 时间 $t_k$ 的历史上下文
- $o_{t_k}$: 当前自中心观测
- $\mathcal{I}$: 自然语言指令
- $\mathcal{F}$: 技能集合

### 公式2: 规划技能的半马尔可夫转移

$$
P(s_{t_k+\tau}, \tau \mid s_{t_k}, f_{c_k})
= \sum_{a_1 \dots a_\tau}
\left(
\prod_{i=0}^{\tau-1}
P(s_{t_k+i+1} \mid s_{t_k+i}, a_{t_k+i})
\pi_{f_{c_k}}(a_{t_k+i})
\right)
\beta_{f_{c_k}}(s_{t_k+\tau})
$$

**含义**: 调用规划技能后，技能内部连续产生 $\tau$ 步低层动作，直到终止条件触发后才回到 VLM 高层决策。

**符号说明**:

- $f_{c_k}$: 当前被选中的技能
- $\tau$: 技能持续执行的时间步数
- $\pi_{f_{c_k}}$: 技能内部的低层控制策略
- $\beta_{f_{c_k}}$: 技能终止条件

### 公式3: RGB-D 像素反投影到世界坐标

$$
\mathbf{P}^{w} = R_t (d \cdot K^{-1}\mathbf{p}_{img}) + \mathbf{t}_t
$$

**含义**: 将当前图像中带深度的像素点转换为世界坐标点，用于增量构建 occupancy grid。

**符号说明**:

- $\mathbf{p}_{img}=[u,v,1]^T$: 图像齐次像素坐标
- $d$: 该像素的深度
- $K$: 相机内参矩阵
- $R_t, \mathbf{t}_t$: 当前相机姿态
- $\mathbf{P}^{w}$: 世界坐标系中的 3D 点

### 公式4: 3D 路径点投影到图像平面

$$
s \cdot \mathbf{p}_{path}^{img}
= K R_t^{-1}(\mathbf{P}_{path}^{w} - \mathbf{t}_t)
$$

**含义**: 将世界坐标中的可行路径点投影到当前图像，生成 VLM 可见的像素级路径 prompt。

**符号说明**:

- $\mathbf{P}_{path}^{w}$: 世界坐标中的 3D 路径点
- $\mathbf{p}_{path}^{img}=[u_{path},v_{path},1]^T$: 图像平面中的路径点
- $s$: 相机坐标系中的深度尺度因子

### 公式5: 细粒度自纠错动作

$$
a_t \sim \pi_{\theta}(a \mid \mathcal{H}_t, o_t, \mathcal{I}), \quad
a \in \{\texttt{Forward}, \texttt{Left}, \texttt{Right}\}
$$

**含义**: 当没有可靠的全局路径提示时，VLM 根据上下文直接输出短程原子动作进行主动探索和路线修正。

### 公式6: 目标像素反投影到相机坐标

$$
\mathbf{P}_{target}^{c}
= d_{target} \cdot K^{-1}\mathbf{p}_{target}^{img}
=
\begin{bmatrix}
\frac{(u_{target}-c_x)\cdot d_{target}}{f_x} \\
\frac{(v_{target}-c_y)\cdot d_{target}}{f_y} \\
d_{target}
\end{bmatrix}
$$

**含义**: QD-PCoT 输出目标像素坐标后，从深度图读取深度并反投影为相机坐标系下的 3D 目标点。

**符号说明**:

- $\mathbf{p}_{target}^{img}=[u_{target},v_{target},1]^T$: 目标像素坐标
- $d_{target}=D_t(u_{target},v_{target})$: 目标像素深度
- $f_x,f_y$: 相机焦距
- $c_x,c_y$: 主点坐标

### 公式7: 目标点从相机坐标变换到世界坐标

$$
\mathbf{P}_{target}^{w} = R_t \mathbf{P}_{target}^{c} + \mathbf{t}_t
$$

**含义**: 将 VLM 在 2D 视觉平面选择的目标位置转换为机器人规划技能可执行的世界坐标目标点。

---

## 关键图表

### Figure 1: Performance Comparison on RxR-CE / 精度-参数量对比

![Figure 1](https://arxiv.org/html/2603.17670v1/x1.png)

**说明**: AgentVLN 在 RxR-CE Val-Unseen 上以 3B 参数量达到更高 SR，相比 7B/8B 级模型有更好的准确率-效率权衡，并强调可在 Jetson 边缘板本地实时推理。

### Figure 2: Overview of AgentVLN / 框架总览

![Figure 2](https://arxiv.org/html/2603.17670v1/x2.png)

**说明**: AgentVLN 以 VLM-as-Brain 方式调度模块化技能。全局路径导航阶段将 3D 路径点投影为像素 prompt；局部目标定位阶段用 QD-PCoT 查询缺失空间线索并完成 3D grounding。

### Figure 3: Navigation Visualization / 导航可视化

![Figure 3](https://arxiv.org/html/2603.17670v1/x3.png)

**说明**: 绿色点表示感知技能生成的视觉 prompt，红圈表示模型选择或预测的导航 waypoint。遇到狭窄通道或严重遮挡时，模型会输出细粒度原子动作进行轨迹微调。

### Figure 4: Real-World Deployment / 真实机器人部署

![Figure 4](https://arxiv.org/html/2603.17670v1/x4.png)

**说明**: AgentVLN 在室内复杂狭窄空间和户外光照变化环境中均能根据自然语言指令规划并执行导航轨迹。硬件平台为 Unitree Go2 + RealSense D455。

### Figure 5: Temporal Context Length Ablation / 历史上下文长度消融

![Figure 5](https://arxiv.org/html/2603.17670v1/x5.png)

**说明**: 历史窗口过短会导致模型缺少运动上下文，过长会引入过期视觉噪声并稀释注意力。论文设置 8 帧上下文时达到最佳 SR/SPL。

### Table 1: R2R-CE Val-Unseen 结果

| 方法 | 参数/输入 | NE↓ | OS↑ | SR↑ | SPL↑ |
|------|----------|-----|-----|-----|------|
| JanusVLN-8.2B | RGB | 4.78 | 65.2 | 60.5 | 56.8 |
| EfficientVLN-4B | RGB | 4.18 | 73.7 | 64.2 | 55.9 |
| DualVLN-7.1B | RGB | 4.05 | 70.7 | 64.3 | 58.5 |
| InternVLA-N1-8.3B | RGB + Depth | 4.83 | 63.3 | 58.2 | 54.0 |
| **AgentVLN-3B** | **RGB + Depth** | **3.88** | **73.5** | **67.2** | **64.7** |

**说明**: AgentVLN 在 R2R-CE Val-Unseen 上以 3B 模型达到最高 SR/SPL。相对 InternVLA-N1，SR 提升 9.0%，SPL 提升 10.7%。

### Table 2: RxR-CE Val-Unseen 结果

| 方法 | 参数/输入 | NE↓ | SR↑ | SPL↑ | nDTW↑ |
|------|----------|-----|-----|------|-------|
| NavFoM-7B | RGB | 5.51 | 57.4 | 49.4 | 60.2 |
| JanusVLN-8.2B | RGB | 6.06 | 56.2 | 47.5 | 62.1 |
| EfficientVLN-4B | RGB | 3.88 | 67.0 | 54.3 | 68.4 |
| DualVLN-7.1B | RGB | 4.58 | 61.4 | 51.8 | 70.0 |
| InternVLA-N1-8.3B | RGB + Depth | 5.91 | 53.5 | 46.1 | 65.3 |
| **AgentVLN-3B** | **RGB + Depth** | **3.92** | **69.5** | **61.3** | **74.6** |

**说明**: AgentVLN 在 RxR-CE 上 SR 比 EfficientVLN 高 2.5%，SPL 高 7.0%，nDTW 也最高。

### Table 3: 模块消融

| VLM-as-Brain | CDFG | QD-PCoT | NE↓ | OS↑ | SR↑ | SPL↑ |
|--------------|------|---------|-----|-----|-----|------|
| - | - | - | 6.53 | 48.50 | 38.60 | 35.10 |
| ✓ | - | - | 4.67 | 65.10 | 59.70 | 55.60 |
| ✓ | ✓ | - | 3.90 | 72.10 | 65.60 | 63.40 |
| ✓ | ✓ | ✓ | **3.88** | **73.50** | **67.20** | **64.70** |

**说明**: VLM-as-Brain + 跨空间映射带来最大跃升；上下文细粒度策略缓解长程误差累积；QD-PCoT 进一步解决局部目标定位中的尺度歧义。

---

## 实验

### 数据集

| 数据集 | 规模/特点 | 用途 |
|--------|-----------|------|
| [[R2R-CE]] | Room-to-Room 连续环境 VLN | Val-Unseen 导航评测 |
| [[RxR-CE]] | 多语言、更长指令、更复杂路径的连续环境 VLN | Val-Unseen 导航评测 |
| [[AgentVLN-Instruct]] | Habitat 中构建的技能调用与定位推理指令微调数据 | 训练 VLM-as-Brain 和 QD-PCoT |
| [[LLaVA]]-Video-178K | 通用视频多模态指令数据 | 混合训练，保持通用多模态能力 |

### 实现细节

- **VLM骨干**: [[Qwen2.5-VL]]-3B
- **训练策略**: 冻结视觉编码器和多模态投影层
- **优化器**: [[AdamW]]
- **Batch size**: 128
- **学习率**: peak learning rate $2\times10^{-5}$，cosine annealing，warmup ratio 0.03
- **输入分辨率**: 640 × 480
- **视场角**: 110°
- **历史上下文窗口**: 8 帧
- **训练资源**: 32 × [[NVIDIA A100]]
- **评估指标**: NE、OS、SR、SPL、nDTW

### 真实部署

| 项目 | 内容 |
|------|------|
| 机器人 | [[Unitree Go2]] 四足机器人 |
| 相机 | [[Intel RealSense]] D455 |
| 建图 | [[RTAB-Map]] 实时局部 SLAM |
| 部署平台 | Jetson edge computing platform |
| 技能库 | 相机处理、RTAB-Map、逆透视投影、路径生成、避障与低层控制 |

真实系统流程为：VLM 触发感知技能 → RTAB-Map 更新 3D occupancy grid → 逆透视/投影生成像素级 prompt → VLM 输出目标像素或路径选择 → 映射到 3D waypoint → 规划技能生成平滑避障轨迹 → 低层控制执行。

---

## 批判性思考

### 优点

1. **工程部署意识强**: 不是单纯追求更大的 VLM 或更复杂的 3D encoder，而是把 3B VLM、传统 SLAM/规划和工具调用结合起来，明确面向边缘端机器人部署
2. **2D-3D 桥接方式直观有效**: 将 3D waypoint 投影成 2D visual prompt，避免让 VLM 在不可见的 3D 坐标空间中推理
3. **技能库抽象具有可扩展性**: 新传感器或新平台可以通过替换感知/规划技能适配，不必重新训练大模型
4. **将主动感知纳入导航推理**: QD-PCoT 让模型在不确定时主动查询深度/几何信息，比直接回归坐标更稳健
5. **消融结果清晰**: VLM-as-Brain、CDFG、QD-PCoT 的增益逐级体现，模块贡献比较容易解释

### 局限性

1. **依赖深度与位姿/建图系统**: 虽然避免了大规模 3D encoder，但仍依赖 RGB-D、SLAM、相机标定和机器人位姿，系统误差会影响投影 prompt 质量
2. **“Agentic”能力主要来自技能调度**: 高层 VLM 仍是微调后的调用器，是否具备强泛化的工具组合和异常恢复能力，需要更多开放环境验证
3. **AgentVLN-Instruct 数据细节不足**: 主文没有充分披露数据规模、构造模板、QA 生成策略和质量控制，复现难度可能较高
4. **对比公平性需要细看**: AgentVLN 使用 RGB + Depth 和外部 SLAM/规划技能；与纯 RGB 或不同传感器设定的 VLM 方法比较时，应区分模型能力与系统能力
5. **真实实验偏展示型**: 主文展示室内外部署效果，但缺少大规模真实机器人成功率、延迟、能耗、地图失败率等量化指标
6. **CoT 可解释性与可靠性**: QD-PCoT 用自然语言查询补充空间信息，但查询是否总能覆盖关键歧义、错误查询如何纠正，仍需要更系统的 failure analysis

### 适用场景

- 有 RGB-D / SLAM / 路径规划栈的移动机器人
- 需要在边缘端实时运行的 VLN 系统
- 环境几何可通过传统建图方法可靠获得，但语义指令复杂、需要 VLM 理解的导航任务
- 希望把基础模型作为高层调度器，而不是完全端到端替代机器人导航栈的系统

### 不适用场景

- 无深度、无可靠里程计或无法建图的极简传感器设置
- 大量动态障碍、人群交互或强非刚性场景
- 需要模型端到端学习低层控制而非调用外部规划/控制栈的研究设定

---

## 与已有笔记的关系

- 与 [[DualVLN]] / [[InternVLA-N1]] 相比：AgentVLN 也是双层/双系统思想，但更强调显式技能库与像素 prompt 投影，而不是学习隐式 latent plan
- 与 [[StreamVLN]] / [[PROSPECT]] 相比：AgentVLN 不把重点放在长上下文流式记忆或潜在预测，而是把长期几何交给 SLAM/规划技能，把 VLM 用作高层调度
- 与 [[EmergeNav]] 相比：两者都偏向“结构化推理脚手架 + VLM”，但 AgentVLN 更工程化，依赖 RGB-D、SLAM 与可执行技能库；EmergeNav 更强调零样本推理结构
- 与 [[EfficientVLN]] 相比：AgentVLN 同样追求轻量部署，但不通过额外深度估计模块让 VLM 隐式学习 3D，而是显式调用感知技能并做 2D-3D 坐标映射

---

## 关键词

[[Vision-Language Navigation]] · [[VLM]] · [[Embodied Intelligence]] · [[POSMDP]] · [[Skill Library]] · [[Cross-Space Representation Mapping]] · [[QD-PCoT]] · [[RTAB-Map]] · [[Unitree Go2]] · [[Intel RealSense]] · [[Qwen2.5-VL]]
