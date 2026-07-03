---
title: "RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching"
method_name: "RAFT-Stereo"
authors: [Lahav Lipson, Zachary Teed, Jia Deng]
year: 2021
venue: 3DV 2021 (Best Student Paper)
tags: [stereo-matching, optical-flow, recurrent-network, real-time-stereo, cost-volume, depth-estimation]
image_source: mixed
arxiv_html: https://arxiv.org/html/2109.07547
created: 2026-05-12
---

# 论文笔记：RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Princeton University |
| 日期 | September 2021 |
| 项目主页 | — |
| 对比基线 | [[DSMNet]], [[GANet]], [[PSMNet]], [[HITNet]], [[LEAStereo]] |
| 链接 | [arXiv](https://arxiv.org/abs/2109.07547) / [Code](https://github.com/princeton-vl/RAFT-Stereo) |

---

## 一句话总结

> 将光流网络 RAFT 适配到双目立体匹配，引入多级 GRU 跨分辨率传播信息，首次在 Middlebury 和 ETH3D 排行榜登顶。

---

## 核心贡献

1. **3D Correlation Volume**: 将 RAFT 的 4D [[Correlation Volume]] 降维为 3D，仅沿水平方向匹配，通过单次矩阵乘法高效计算
2. **Multi-Level GRU**: 引入多分辨率 (1/8, 1/16, 1/32) [[GRU]] 更新算子，通过跨分辨率连接高效传播全局信息
3. **Slow-Fast GRU + 实时推理**: 低分辨率 GRU 迭代更多次、高分辨率更少次，配合共享 backbone 实现 26 FPS 的实时立体匹配

---

## 问题背景

### 要解决的问题
双目立体匹配需要从左右视图对中估计逐像素视差图。现有方法依赖 3D 卷积处理 cost volume，计算和内存开销巨大，难以处理高分辨率图像。

### 现有方法的局限
- 基于 3D 卷积的方法（[[GCNet]], [[PSMNet]], [[GANet]]）计算量大、内存占用高，无法直接处理百万像素图像
- 3D 卷积方法泛化能力差，在训练域外数据上表现下降明显
- [[DSMNet]] 虽提升泛化但仍依赖 3D 卷积，操作分辨率受限

### 本文的动机
光流网络 [[RAFT]] 使用迭代优化替代 3D 卷积，在光流任务上表现优异。双目立体匹配本质上是水平方向约束的光流估计，可以将 RAFT 的设计适配过来，同时解决 3D 卷积方法的计算瓶颈和泛化问题。

---

## 方法详解

### 模型架构

RAFT-Stereo 采用 **迭代优化** 架构，由三个主要组件构成：

- **输入**: 校正后的左右立体图像对 $(I_L, I_R)$
- **Feature Encoder**: 对左右图像提取特征图 $\mathbf{f}, \mathbf{g} \in \mathbb{R}^{H \times W \times D}$，使用 [[Instance Normalization]]
- **Context Encoder**: 仅对左图提取上下文特征，使用 [[Batch Normalization]]，初始化隐状态
- **Correlation Pyramid**: 3D 关联体积的 4 级金字塔
- **Multi-Level GRU Update Operator**: 在 1/8, 1/16, 1/32 分辨率上交叉连接的 [[GRU]] 网络迭代更新视差
- **输出**: 全分辨率视差图 $\mathbf{d}$，通过 [[Convex Upsampling]] 从 1/8 分辨率上采样

### 核心模块

#### 模块1: 3D Correlation Volume

**设计动机**: 利用 [[Epipolar Geometry|极线约束]] ——校正后的立体图像对中，对应点在同一水平线上——将 RAFT 的 4D 关联体积简化为 3D。

**具体实现**:
- 对左右图像分别提取特征 $\mathbf{f}, \mathbf{g} \in \mathbb{R}^{H \times W \times D}$
- 仅计算同一行像素之间的内积，构建 3D 关联体积
- 通过沿最后一维的 [[Average Pooling]] 构建 4 级金字塔，各层维度为 $H \times W \times W/2^k$
- Correlation Lookup 操作：以当前视差估计为中心，在 1D 网格上从各层索引特征

#### 模块2: Multi-Level GRU Update Operator

**设计动机**: 原始 RAFT 在单一分辨率上迭代更新，[[Receptive Field|感受野]] 增长缓慢，对大面积无纹理区域效果差。

**具体实现**:
- 在 1/8, 1/16, 1/32 三个分辨率上分别维护隐状态
- 各分辨率的 GRU 以相邻分辨率的隐状态作为上下文输入（cross-connected）
- 最高分辨率 (1/8) 的 GRU 执行 correlation lookup 并生成最终视差更新
- 低分辨率 GRU 通过上/下采样与相邻层交换信息

#### 模块3: Slow-Fast GRU（实时变体）

**设计动机**: 1/8 分辨率的 GRU 更新计算量约为 1/16 分辨率的 4 倍，通过差异化迭代次数换取速度。

**具体实现**:
- "Slow-Fast" 策略：1/32, 1/16, 1/8 分辨率分别迭代 30, 20, 10 次
- 共享 backbone 替代分离的 feature encoder 和 context encoder
- 仅使用 1/8 和 1/16 两级 GRU（bi-level）
- 实现 CUDA 双线性采样器替代 PyTorch 默认实现消除瓶颈

---

## 关键公式

### 公式1: [[Correlation Volume|3D 关联体积]]

$$
\mathbf{C}_{ijk} = \sum_h \mathbf{f}_{ijh} \cdot \mathbf{g}_{ikh}, \quad \mathbf{C} \in \mathbb{R}^{H \times W \times W}
$$

**含义**: 对左图每个像素 $(i,j)$ 和右图同一行的每个像素 $(i,k)$，通过特征内积计算匹配代价

**符号说明**:
- $\mathbf{f}_{ijh}$: 左图像素 $(i,j)$ 的第 $h$ 维特征
- $\mathbf{g}_{ikh}$: 右图像素 $(i,k)$ 的第 $h$ 维特征
- $\mathbf{C}_{ijk}$: 左图 $(i,j)$ 与右图 $(i,k)$ 之间的关联分数

### 公式2: [[L1 Loss|带指数加权的序列 L1 损失]]

$$
\mathcal{L} = \sum_{i=1}^{N} \gamma^{N-i} ||\mathbf{d}_{gt} - \mathbf{d}_i||_1, \quad \text{where } \gamma = 0.9
$$

**含义**: 对迭代优化过程中每一步的预测视差施加监督，越靠后的预测权重越大

**符号说明**:
- $N$: 总迭代次数
- $\mathbf{d}_{gt}$: 真值视差图
- $\mathbf{d}_i$: 第 $i$ 次迭代的预测视差
- $\gamma = 0.9$: 指数衰减系数，使后期预测的损失权重更大

---

## 关键图表

### Figure 1: Overview / RAFT-Stereo 整体架构

![[RAFT-Stereo_fig1.png|600]]

**说明**: RAFT-Stereo 整体流程。Feature Encoder（蓝色）从左右图提取特征并构建 Correlation Pyramid；Context Encoder 从左图提取上下文特征和初始隐状态。视差从 0 初始化，GRU（绿色）迭代地从 Correlation Pyramid 中查询特征并更新视差。

### Figure 2: Correlation Lookup / 关联金字塔查询

![[RAFT-Stereo_fig2.png|600]]

**说明**: 以当前视差估计为中心，在金字塔各层通过 1D 网格和线性插值查询关联特征，拼接为单一特征向量。各层网格大小取决于金字塔层级。

### Figure 3: Multilevel GRU / 多级 GRU 结构

![[RAFT-Stereo_fig3.png|600]]

**说明**: 三级卷积 GRU 分别在 1/32、1/16、1/8 分辨率上操作。相邻分辨率通过上采样/下采样交换信息。最高分辨率 GRU（红色）从 Correlation Pyramid 查询特征并输出视差更新 $\Delta$。

### Figure 4: ETH3D 定性结果

![[RAFT-Stereo_fig4.png|600]]

**说明**: ETH3D 立体数据集上的可视化结果。RAFT-Stereo 对无纹理表面和过曝区域具有鲁棒性。

### Figure 5: Middlebury 定性对比

![[RAFT-Stereo_fig5.png|600]]

**说明**: Middlebury 测试集上与 LEAStereo、HITNet 的对比。RAFT-Stereo 能恢复极其精细的细节（自行车辐条、植物叶片、锐利物体边界），其他方法无法做到。

### Figure 6: 速度-精度权衡

![[RAFT-Stereo_fig6.png|600]]

**说明**: 从合成数据到 KITTI-2015 的零样本泛化性能与推理速度对比。RAFT-Stereo (Realtime) 版本达到约 26 FPS，同时保持与 SOTA 方法竞争的 D1 error。

### Table 1: 合成到真实的零样本泛化

| Method | KITTI-15 | Middlebury full | Middlebury half | Middlebury quarter | ETH3D |
|--------|----------|----------------|-----------------|-------------------|-------|
| HD³ | 26.5 | 50.3 | 37.9 | 20.3 | 54.2 |
| gwcnet | 22.7 | 47.1 | 34.2 | 18.1 | 30.1 |
| PSMNet | 16.3 | 39.5 | 25.1 | 14.2 | 23.8 |
| GANet | 11.7 | 32.2 | 20.3 | 11.2 | 14.1 |
| DSMNet | **6.5** | **21.8** | **13.8** | **8.1** | **6.2** |
| **Ours** | **5.74** | **18.33** | **12.59** | **9.36** | **3.28** |

**说明**: 所有方法在 SceneFlow 上训练，直接测试。RAFT-Stereo 在所有三个数据集上均取得零样本泛化 SOTA。

### Table 2: KITTI-2015 排行榜

| Method | all | foreg. | backg. |
|--------|-----|--------|--------|
| AcfNet | 1.89 | 3.80 | 1.51 |
| AMNet | 1.84 | 3.43 | 1.53 |
| OptStereo | 1.82 | 3.43 | 1.50 |
| GANet-deep | 1.81 | 3.46 | 1.48 |
| SUW-Stereo | 1.80 | 3.45 | **1.47** |
| GANet+DSMNet | 1.77 | 3.23 | 1.48 |
| CSPN | **1.74** | **2.88** | 1.51 |
| LEAStereo | **1.65** | 2.91 | **1.40** |
| **Ours** | 1.96 | 2.89 | 1.75 |

**说明**: KITTI-2015 排行榜结果（EPE > 3px 的错误像素百分比）。RAFT-Stereo 在前景像素上排名第二。

### Table 3: ETH3D 排行榜

| Method | bad 0.5 (%) | bad 1.0 (%) | bad 2.0 (%) | AvgErr |
|--------|------------|------------|------------|--------|
| HSM | 10.88 | 4.00 | 1.36 | 0.28 |
| NOSS-ROB | 10.99 | 3.30 | 1.29 | 0.31 |
| iResNet | 10.26 | 3.68 | 1.00 | 0.24 |
| AdaStereo | 10.22 | 3.09 | 0.65 | 0.24 |
| HIT-Net | 7.83 | 2.79 | 0.80 | 0.20 |
| **Ours** | **7.04** | **2.44** | **0.44** | **0.18** |

**说明**: ETH3D 测试集排行榜。RAFT-Stereo 在所有指标上均排名第一。

### Table 4: Middlebury 排行榜

| Methods | AvgErr | MedErr | bad 0.5 (%) | bad 1.0 (%) | bad 2.0 (%) | bad 4.0 (%) |
|---------|--------|--------|------------|------------|------------|------------|
| EdgeStereo | 2.68 | 0.72 | 55.6 | 32.4 | 18.7 | 10.8 |
| HSM-Net | 2.07 | 0.56 | 50.7 | 24.6 | 10.2 | 4.83 |
| LEAStereo | 1.43 | 0.53 | 49.5 | 20.8 | 7.15 | **2.75** |
| MC-CNN | 2.63 | 0.44 | 40.1 | 16.1 | 6.35 | 3.81 |
| LocalExp | 2.24 | 0.43 | 38.7 | 13.9 | 5.43 | 3.69 |
| CRLE | 2.25 | 0.42 | 38.1 | 13.4 | 5.75 | 3.90 |
| HITNet | 1.71 | 0.40 | 34.2 | 13.3 | 6.46 | 3.81 |
| NOSS-ROB | 2.08 | 0.42 | 38.2 | 13.2 | 5.01 | 3.46 |
| **Ours** | **1.27** | **0.26** | **27.7** | **9.37** | **4.74** | 2.75 |

**说明**: Middlebury 测试集排行榜。RAFT-Stereo 排名第一，bad 2px error 为 4.74%，相比次优方法下降 26%。

### Table 5: 合成数据组合泛化实验

| SceneFlow | Falling Things | Tartan Air | Sintel Stereo | ETH3D | KITTI-15 | Middlebury (Full) |
|-----------|---------------|------------|---------------|-------|----------|-------------------|
| ✓ | — | — | — | **4.44** | 6.37 | 23.40 |
| — | ✓ | — | — | 26.93 | 6.13 | 25.39 |
| — | — | ✓ | — | 4.62 | 5.87 | 26.28 |
| ✓ | ✓ | — | — | 25.2 | 5.88 | 20.95 |
| ✓ | ✓ | ✓ | — | 7.65 | 5.76 | **20.65** |
| ✓ | ✓ | ✓ | ✓ | 5.65 | **5.62** | 21.99 |

**说明**: 不同合成数据组合的泛化效果。组合 SceneFlow + Falling Things + Tartan Air + Sintel Stereo 在 KITTI-15 上最优。

### Table 6: 消融实验

| Experiment | Method | FlyingThings3D | Runtime (s) | Parameters |
|-----------|--------|---------------|-------------|------------|
| # GRU Levels | **3 Levels** | **9.40** | 0.132 | 11.23M |
| | 1 Level | 9.64 | **0.091** | 9.46M |
| Backbone | Single Backbone | **9.37** | **0.121** | 10.75M |
| | Sep. Backbones | 9.40 | 0.132 | 11.23M |
| Resolution | **1/4th** | **7.92** | 0.338 | 11.12M |
| | 1/8th | 9.40 | **0.132** | 11.23M |
| Slow-Fast GRU | Regular | **9.40** | 0.132 | 11.23M |
| | Slow-Fast | 9.98 | **0.063** | 11.23M |
| Collapsed Cost Vol. | RAFT | — | 0.224 | 5.26M |
| | RAFT-Stereo | — | **0.132** | 11.23M |

**关键发现**: 3 级 GRU 精度优于 1 级但更慢；共享 backbone 精度不降反而更快；Slow-Fast GRU 运行时间减半仅损失 0.58 的 1px error；1/4 分辨率操作大幅提升精度但 2.5x 慢。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[SceneFlow]] (FlyingThings3D) | ~22K | 合成飞行物体场景 | 预训练 |
| [[KITTI-2015]] | 200 对 | 真实驾驶场景 | 微调 + 评测 |
| [[ETH3D]] | — | 灰度/黑白高分辨率 | 评测 |
| [[Middlebury]] | 23 对 | 高分辨率室内场景 | 微调 + 评测 |
| Falling Things | 61.5K | 随机散落物体合成数据 | 额外训练 |
| Tartan Air | 296K | 仿真环境 SLAM 数据 | 额外训练 |
| Sintel Stereo | 2.1K | 光流合成数据集的立体子集 | 额外训练 |

### 实现细节

- **Backbone**: ResNet-like feature encoder (256 channels)，下采样 4x 或 8x
- **优化器**: [[AdamW]]，one-cycle 学习率调度，最低 $1e^{-4}$
- **Batch Size**: 8（最终模型），6（消融实验）
- **训练轮数**: 200k steps（最终）/ 100k steps（消融）
- **硬件**: 2x NVIDIA RTX 6000 GPU
- **训练分辨率**: 360x720 随机裁剪
- **迭代次数**: 训练 22 次 / 推理 32 次（标准）/ 80 次（ETH3D）

### 实时推理配置

- 共享 backbone + bi-level (1/8, 1/16) Slow-Fast GRU + 7 次迭代
- KITTI 分辨率 (1248x384) 达到 **26 FPS**
- D1 error 5.91（对比 DSMNet 6.5）

### 可视化结果

RAFT-Stereo 的定性优势：
- 恢复极精细细节（自行车辐条、植物叶片边界）
- 对无纹理平面、过曝区域鲁棒
- 不需要分块处理百万像素级图像

---

## 批判性思考

### 优点
1. **架构简洁**: 抛弃 3D 卷积范式，仅用 2D 操作+迭代优化，训练简单（仅 L1 损失）
2. **泛化能力出色**: 零样本合成到真实迁移在所有主流 benchmark 上 SOTA
3. **灵活的速度-精度权衡**: 通过 Slow-Fast GRU、共享 backbone、迭代次数等配置可连续调节
4. **内存高效**: 无需 3D cost volume，可直接处理百万像素级全分辨率图像

### 局限性
1. **非实时 vs 实时版本差距明显**: 标准版 0.132s 距离实时尚远，实时版牺牲精度
2. **KITTI 排行榜表现非最优**: 微调后 KITTI-2015 仅排前列但非第一，说明在 domain-specific 场景下 3D 卷积方法仍有竞争力
3. **对极线校正的依赖**: 假设完美校正的水平极线约束，实际应用中校正误差会影响性能

### 潜在改进方向
1. 引入自适应校正误差补偿（CREStereo 后续解决了这一问题）
2. 结合预训练视觉基础模型提升语义理解（FoundationStereo 的方向）
3. 时序信息利用实现视频立体匹配

### 可复现性评估
- [x] 代码开源
- [x] 预训练模型
- [x] 训练细节完整
- [x] 数据集可获取

---

## 关联笔记

### 基于
- [[RAFT]]: 核心架构来源——光流网络的迭代优化范式
- [[GCNet]]: 首个端到端 3D 卷积立体匹配网络

### 对比
- [[DSMNet]]: 3D 卷积方法中泛化最好，但计算量大
- [[HITNet]]: tile-based 实时方法，需额外损失函数
- [[LEAStereo]]: NAS 搜索的立体匹配网络
- [[GANet]]: guided aggregation 方法

### 方法相关
- [[Correlation Volume]]: 核心匹配表示
- [[GRU]]: 迭代更新的核心组件
- [[Convex Upsampling]]: 从低分辨率恢复全分辨率视差
- [[Instance Normalization]]: feature encoder 中使用

### 硬件/数据相关
- [[SceneFlow]]: 主要预训练数据集
- [[KITTI-2015]]: 自动驾驶场景评测
- [[ETH3D]]: 高精度立体评测
- [[Middlebury]]: 经典高分辨率立体评测

---

## 速查卡片

> [!summary] RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching
> - **核心**: 将 RAFT 光流架构适配双目立体匹配，多级 GRU 高效传播全局信息
> - **方法**: 3D Correlation Volume + Multi-Level Cross-Connected GRU + Slow-Fast 实时策略
> - **结果**: Middlebury 第一 (bad 2px 4.74%)，ETH3D 第一 (bad 1px 2.44%)，实时版 26 FPS
> - **代码**: https://github.com/princeton-vl/RAFT-Stereo

---

*笔记创建时间: 2026-05-12*
