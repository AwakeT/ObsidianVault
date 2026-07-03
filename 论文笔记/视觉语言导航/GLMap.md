---
title: "Multi-Scale Gaussian-Language Map for Zero-shot Embodied Navigation and Reasoning"
method_name: "GLMap"
authors: [Sixian Zhang, Yiyao Wang, Xinhang Song, Keming Zhang, Zijian Xu, Shuqiang Jiang]
year: 2026
venue: arXiv
tags: [semantic-map, gaussian-splatting, embodied-navigation, object-navigation, instance-navigation, situated-question-answering, zero-shot]
image_source: online
arxiv_html: https://arxiv.org/html/2605.01736v1
created: 2026-05-20
---

# 论文笔记：Multi-Scale Gaussian-Language Map for Zero-shot Embodied Navigation and Reasoning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | State Key Laboratory of AI Safety, Institute of Computing Technology, CAS; University of Chinese Academy of Sciences |
| 日期 | May 2026 |
| 项目主页 | N/A |
| 对比基线 | [[VLFM]], [[ApexNAV]], [[UniGoal]], [[GOAT]], [[g3D-LF]], [[GPT4Scene]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.01736) / [HTML](https://arxiv.org/html/2605.01736v1) / [Code](https://github.com/sx-zhang/GLMap) |

---

## 一句话总结

> 提出多尺度 Gaussian-Language Map，把环境表示为 2D 索引网格 + 实例/区域语义单元，每个语义单元同时存储自然语言描述和可快速渲染的 3D Gaussian，从而以训练自由方式增强 ObjectNav、InstNav 和 SQA。

---

## 核心贡献

1. **Gaussian-Language Map 表示**: 同时满足显式几何、多尺度语义和大模型友好接口，每个语义单元直接暴露 text description 与 rendered image
2. **实例级 + 区域级多尺度语义**: instance unit 记录物体类别、属性、颜色和 3D Gaussian；region unit 记录功能区域、场景上下文和包含的 instance IDs
3. **解析式 Gaussian Estimator**: 不使用 3DGS 的梯度优化，而是从 RGB-D 反投影得到的点云中按 voxel / neighborhood 直接估计 Gaussian 参数，支持在线增量构建
4. **零样本兼容 LLM/VLM/MLLM**: GLMap 输出自然语言和渲染图像，不需要 feature-to-token projection 或额外 alignment training
5. **多任务验证**: 在 zero-shot ObjectNav、zero-shot InstNav 和 SQA 上均提升现有方法，显示 map representation 对导航和具身推理都有价值

---

## 问题背景

### 要解决的问题

具身智能体需要理解并记忆环境的几何结构和语义结构，以完成 [[Object Navigation|ObjectNav]]、Instance Navigation 和 Situated Question Answering 等任务。一个有效的地图不仅要知道物体在哪里，还要知道物体属性、区域功能、空间关系，并能被当前大模型直接使用。

### 现有方法的局限

- [[Topological Map|拓扑地图]] 高效但空间关系粗糙，边通常只表达邻接关系，难以定位精细目标
- grid semantic map 有明确坐标，但通常只存类别 label，缺少属性、上下文和 affordance
- dense geometric map / point cloud 有几何细节，但存储开销高，且常需要训练额外投影模块对齐到 LLM/MLLM token space
- CLIP/VLM feature map 语义较丰富，但是隐式向量，不是 LLM/MLLM 原生输入，实例边界也不清楚

### 本文的动机

大模型原生擅长处理 **文本和图像**，而不是中间 feature tensor。GLMap 因此不把语义压进隐式 embedding，而是为每个语义单元保存自然语言描述和可渲染图像，使 LLM、VLM、MLLM 可以零样本读取地图。

---

## 方法详解

### 模型架构

GLMap 是一种 **incremental semantic-geometric map**：

```text
RGB-D + camera pose
  -> MLLM semantic parsing
  -> instance / region text descriptions
  -> GroundingDINO + MobileSAM masks
  -> back-project masks to 3D point clouds
  -> analytic Gaussian Estimator
  -> instance / region matching and merging
  -> 2D indexing grid update
  -> downstream ObjectNav / InstNav / SQA
```

GLMap 包含三个核心组成：

1. **2D indexing grid**  
   每个 grid cell 存储投影落入该区域的语义单元 ID，用于空间查询和 value map 构建。

2. **Instance semantic units**  
   每个实例单元包含开放词汇文本描述和一组 3D Gaussians。

3. **Region semantic units**  
   每个区域单元包含区域文本描述和所含 instance IDs；区域本身不直接保存 Gaussians，但可以通过包含的实例 Gaussians 合成渲染图。

### 核心模块

#### 模块1: GLMap Definition

**Instance unit**:

```text
instance = {
  textual description,
  set of 3D Gaussians
}
```

每个 Gaussian 包含：

- mean / position
- covariance
- color
- opacity

**Region unit**:

```text
region = {
  textual description,
  contained instance IDs
}
```

region 负责表达：

- functional region
- scene context
- object relations
- affordance

这种设计避免每个 region 重复存储几何，同时还能通过其包含的 instance Gaussians 渲染 region image。

#### 模块2: Gaussian Estimator

**设计动机**: 原始 [[4D Gaussian Splatting|3D Gaussian Splatting]] 依赖多视图优化，而 embodied RGB-D 任务已有深度和相机内参，可以直接得到高质量点云，因此无需梯度优化。

**具体流程**:

1. 将实例 mask 对应的 RGB-D 像素反投影为点云
2. 点云离散到 regular voxel grid
3. 每个 voxel 收集其 Chebyshev neighborhood 内的点
4. 用邻域点的均值估计 Gaussian mean
5. 用邻域点的协方差估计 Gaussian covariance，并加小的 diagonal regularization
6. 用点颜色均值估计 Gaussian color
7. opacity 固定为 0.8

由于 smooth area 中 voxel Gaussian 会冗余，作者进一步使用 curvature-aware merging：

- shape similarity: covariance / geometry 距离
- color similarity: RGB 距离
- curvature proxy: covariance eigenvalues 中最小特征值与总特征值比例
- 高曲率区域保留更多 Gaussians，保证边界和细节
- 平坦区域更激进合并，降低存储开销

#### 模块3: Incremental Update

**语义解析**:

- 用 Gemma3-27B 从 egocentric RGB 图像生成 instance-level 与 region-level descriptions
- 用 GroundingDINO 根据文本描述定位对应区域
- 用 MobileSAM 精化 mask

**Instance update**:

- 新观测 instance 先通过文本 embedding 与已有 instance 做语义一致性匹配
- text embedding 使用 nomic-embed-text
- 相似度阈值为 0.8
- 语义匹配后，再判断两组 Gaussians 是否满足可合并的几何一致性
- 匹配成功则合并 Gaussian set 和文本描述
- 文本 buffer 超过 300 时，用 Qwen3-8B 做轻量语义归并
- 匹配失败则创建新的 global instance ID

**Region update**:

- 当前帧 region 中的 local instance IDs 先映射为 GLMap global instance IDs
- region 匹配同时检查：
  - region 文本语义一致
  - instance set 至少有一个共同 global instance
- 匹配成功则合并 region 描述和包含实例
- 否则创建新的 global region ID

#### 模块4: ObjectNav / InstNav with GLMap

导航任务中，GLMap 为每个 semantic unit 计算与目标描述的相似度：

- LLM-based: prompt LLM 判断目标与 instance/region text 的相关性
- VLM-based: 渲染 semantic unit 的图像，与目标文本做相似度

然后把 semantic unit relevance 投影到 2D grid，生成 value map：

```text
semantic unit relevance
  -> grid cell score
  -> Gaussian smoothing
  -> select frontier nearest to max-value cell
  -> local policy navigates to waypoint
```

ObjectNav 主要依赖类别与区域上下文；InstNav 更依赖实例属性、颜色和空间关系，因此 region unit 对 InstNav 尤其重要。

#### 模块5: SQA with GLMap

Situated Question Answering 给定 situation description 和问题，例如：

```text
"Sitting at the edge of the bed and facing the couch."
"Can I go straight to the coffee table in front of me?"
```

GLMap 处理方式：

1. 根据 situation description 估计 agent 所在 region 的概率
2. 估计 agent 正面朝向的 instance 概率
3. 得到 agent position 和 orientation
4. 从 GLMap 中用 3D Gaussian Splatting 渲染 agent 周围四个方向的视图
5. 把 rendered images、方向标注和 question 输入 MLLM 生成答案

---

## 关键公式

### 公式1: Gaussian 参数解析估计

```text
points in voxel neighborhood
  -> mean = average 3D position
  -> covariance = local point covariance + diagonal regularization
  -> color = average RGB
  -> opacity = fixed value
```

**含义**: 直接从 RGB-D 点云拟合 3D Gaussian primitives，不需要多视图可微渲染优化。

### 公式2: Gaussian 合并

```text
mergeable if:
  shape distance + color distance < curvature-aware threshold
```

**含义**: 平坦区域强合并减少冗余，高曲率区域弱合并保留边界细节。

### 公式3: 导航 Value Map

```text
semantic relevance of units
  -> assign scores to indexed grid cells
  -> smooth with 2D Gaussian kernel
  -> choose frontier closest to max-value grid
```

**含义**: 将 instance / region 的语义相关性转化为几何坐标上的探索价值。

---

## 实验

### 实验设置

| 项目 | 内容 |
|------|------|
| ObjectNav 数据集 | [[MP3D]], [[HM3D]] |
| InstNav 数据集 | HM3D text-described instance goals |
| SQA 数据集 | SQA3D / [[ScanNet]] |
| ObjectNav / InstNav 指标 | SR, SPL |
| SQA 指标 | EM-1, EM-R1 |
| MLLM semantic parser | Gemma3-27B |
| Mask 提取 | GroundingDINO + MobileSAM |
| Text embedding | nomic-embed-text |
| 语义相似度阈值 | 0.8 |
| voxel size | 1 cm |
| 文本合并 | Qwen3-8B, buffer length 300 |

### Table 1: 多尺度语义消融 (HM3D ObjectNav)

| Indexing grid | Instance unit | Region unit | SR | SPL |
|---------------|---------------|-------------|----|-----|
| ✓ | - | - | 52.5 | 30.4 |
| ✓ | ✓ | - | 57.4 | 31.3 |
| ✓ | - | ✓ | 56.2 | 30.9 |
| ✓ | ✓ | ✓ | **59.1** | **32.2** |

**说明**: instance 和 region 都能提升导航，二者组合最好，说明对象属性和区域上下文是互补信息。

### Table 2: 与 LLM/VLM/MLLM 方法集成

| 方法 | ObjectNav SR | ObjectNav SPL | SQA EM-1 | SQA EM-R1 |
|------|--------------|---------------|----------|-----------|
| ESC | 39.2 | 22.3 | - | - |
| ESC + GLMap | 48.8 (+9.6) | 25.2 (+2.9) | - | - |
| VLFM | 52.5 | 30.4 | - | - |
| VLFM + GLMap | 59.1 (+6.6) | 32.2 (+1.8) | - | - |
| ApexNAV | 59.6 | 33.0 | - | - |
| ApexNAV + GLMap | 62.7 (+3.1) | 33.7 (+0.7) | - | - |
| GPT4Scene | - | - | 57.2 | 60.4 |
| GPT4Scene + GLMap | - | - | 58.5 (+1.3) | 61.3 (+0.9) |

**说明**: GLMap 可以作为外部 map representation 插入 LLM、VLM、MLLM 系统，且无需额外训练。

### Table 3: 与其他地图结构对比

| 地图结构 | 方法 | 语义单元 | ObjectNav SR | InstNav SR | SQA EM-1 |
|----------|------|----------|--------------|------------|----------|
| Topological map | UniGoal | instance + region text | 54.5 | 20.2 | 34.2 |
| Grid map | GOAT | object label | 50.6 | 17.0 | - |
| Grid map | g3D-LF | region visual feature | 55.6 | 11.5 | 47.7 |
| Dense geometric map | Chat-Scene | region visual feature | - | - | 54.6 |
| GLMap | Ours | instance + region text/rendered image | **62.7** | **22.5** | **58.5** |

**关键发现**: GLMap 在三个任务上都优于对应地图结构。它的优势来自明确实例边界、显式空间索引、多尺度语义和大模型原生输入形式。

### Table 4: Zero-shot ObjectNav SOTA 对比

| 方法 | MP3D SR/SPL | HM3D SR/SPL |
|------|-------------|-------------|
| VLFM | 36.4 / 17.5 | 52.5 / 30.4 |
| SG-Nav | 40.2 / 16.0 | 54.0 / 24.9 |
| UniGoal | 41.0 / 16.4 | 54.5 / 25.1 |
| FBN | 41.1 / 17.3 | 53.2 / 30.7 |
| ApexNAV | 39.2 / 17.8 | 59.6 / 33.0 |
| BeliefMapNav | 37.3 / 17.6 | 61.4 / 30.6 |
| GLMap | **42.5 / 18.3** | **62.7 / 33.7** |

### Table 5: Zero-shot InstNav (HM3D)

| 方法 | SR | SPL |
|------|----|-----|
| ZSON | 10.6 | 4.9 |
| PSL | 16.5 | 7.5 |
| GOAT | 17.0 | 8.8 |
| UniGoal | 20.2 | 11.4 |
| GLMap | **22.5** | **13.7** |

### Table 6: SQA (SQA3D)

| 方法 | EM-1 | EM-R1 |
|------|------|-------|
| SQA3D | 46.6 | - |
| g3D-LF | 47.7 | - |
| Scene-LLM | 54.2 | - |
| Chat-Scene | 54.6 | 57.5 |
| 3DGraphLLM | 55.9 | - |
| GPT4Scene | 57.2 | 60.4 |
| GLMap | **58.5** | **61.3** |

---

## 优点与局限

### 优点

1. **大模型友好**: text + rendered image 是 LLM/VLM/MLLM 原生输入，避免额外 token alignment training
2. **多尺度语义清晰**: instance 负责对象属性和边界，region 负责上下文、功能区域和空间关系
3. **显式几何可查询**: 2D indexing grid 支持空间查询、value map 和前沿选择
4. **比 point cloud 更紧凑**: 3D Gaussian 表示支持 compact storage 和快速渲染
5. **在线增量构建**: Gaussian Estimator 是解析式的，不依赖耗时优化，适合 embodied task
6. **任务覆盖广**: 同一个地图表示同时支持 ObjectNav、InstNav 和 SQA

### 局限性

1. **依赖 RGB-D、camera pose 和内参**: 真实机器人若只用单目 RGB，需要额外深度和定位模块
2. **依赖 MLLM 语义解析质量**: instance/region description 错误会影响后续 grounding、合并和推理
3. **GroundingDINO + MobileSAM 误差会累积**: mask 错误直接导致 Gaussian geometry 错误和 instance 合并错误
4. **动态环境未重点处理**: 论文主要面向静态室内场景，动态物体和长期变化可能破坏 map consistency
5. **在线效率需要实机验证**: 虽然 Gaussian Estimator 是解析式，但 MLLM 描述、开放词汇 grounding、SAM 分割和渲染在端侧仍可能较重
6. **区域语义依赖文本合并策略**: 文本 buffer 超长后用 LLM 合并，可能丢失细粒度描述或引入语言模型偏差

---

## 相关工作

### 地图表示

- [[Topological Map]]: 高效但空间关系粗糙
- [[Semantic Map]]: 常以类别标签或 feature 存储语义
- [[Point Cloud]]: 几何明确但存储和大模型接口不友好
- [[4D Gaussian Splatting|3D Gaussian Splatting]]: 可渲染、紧凑的显式视觉表示
- [[ConceptGraphs]]: 物体中心的 3D 场景图表示

### 导航与推理

- [[VLFM]]: VLM frontier scoring
- [[GOAT]]: 多模态目标导航，使用实例记忆
- [[UniGoal]]: 用 scene graph 支持 ObjectNav/InstNav
- [[GPT4Scene]]: 用 BEV 和 keyframe images 给 MLLM 提供显式视觉参考
- [[g3D-LF]]: 3D language feature field

---

## 代码与资源

| 类型 | 链接 |
|------|------|
| arXiv | https://arxiv.org/abs/2605.01736 |
| HTML | https://arxiv.org/html/2605.01736v1 |
| Code | https://github.com/sx-zhang/GLMap |

---

## 个人思考

GLMap 对家庭服务机器人很有启发：它不是把地图做成单一 occupancy grid 或 CLIP feature field，而是将地图拆成“可定位的语义单元”。这与长期家庭记忆需求很接近，因为用户通常问的是：

- “电视旁边那个黑色遥控器在哪里？”
- “餐桌附近有没有可以放杯子的地方？”
- “我背后能打开什么通风？”

这些问题需要实例属性、区域上下文和空间关系，单纯类别地图不够。GLMap 的 text + rendered image 接口也适合接入本地 VLM/MLLM。

但在真实产品中要控制成本：Gemma3-27B + GroundingDINO + MobileSAM + Gaussian rendering 的在线管线较重。更可行的工程化路线可能是：

1. 高频使用轻量检测/分割更新 instance geometry
2. 低频调用 MLLM 生成或修正 instance/region descriptions
3. 用变化检测决定是否重建 Gaussian unit
4. 将 GLMap 作为长期语义记忆层，而不是每帧完整重建

---

## 速查卡片

> [!summary] GLMap
> - **任务**: zero-shot ObjectNav / InstNav / SQA
> - **核心**: 2D indexing grid + instance/region semantic units
> - **表示**: 每个语义单元保存 natural language description + 3D Gaussian rendered image
> - **构建**: MLLM semantic parsing + GroundingDINO + MobileSAM + analytic Gaussian Estimator
> - **导航**: 目标相似度投影到 value map，选择高价值 frontier
> - **结果**: ObjectNav MP3D 42.5/18.3, HM3D 62.7/33.7；InstNav SR 22.5；SQA EM-1 58.5
> - **局限**: 依赖 RGB-D、pose、语义解析/分割质量，在线端侧计算成本仍需优化
