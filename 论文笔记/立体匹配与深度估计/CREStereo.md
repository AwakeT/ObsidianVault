---
title: "Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation"
method_name: "CREStereo"
authors: [Jiankun Li, Peisen Wang, Pengfei Xiong, Tao Cai, Ziwei Yan, Lei Yang, Jiangyu Liu, Haoqiang Fan, Shuaicheng Liu]
year: 2022
venue: CVPR 2022 (Oral)
tags: [stereo-matching, recurrent-network, adaptive-correlation, practical-stereo, depth-estimation, cascaded-refinement]
image_source: online
arxiv_html: https://arxiv.org/html/2203.11483
created: 2026-05-12
---

# 论文笔记：Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Megvii Research, Tencent, UESTC |
| 日期 | March 2022 |
| 项目主页 | — |
| 对比基线 | [[RAFT-Stereo]], [[HITNet]], [[LEAStereo]], [[AANet]], [[GwcNet]] |
| 链接 | [arXiv](https://arxiv.org/abs/2203.11483) / [Code](https://github.com/megvii-research/CREStereo) |

---

## 一句话总结

> 针对消费级相机的非理想校正和复杂真实场景，设计自适应组关联层 + 级联递归网络 + 专用合成数据集，在 Middlebury 和 ETH3D 双双登顶。

---

## 核心贡献

1. **Adaptive Group Correlation Layer (AGCL)**: 2D-1D 交替局部搜索 + 可变形搜索窗口 + 组关联，解决非理想极线校正下的匹配模糊问题
2. **Cascaded Recurrent Network**: 级联多阶段递归更新模块 (RUM)，在 coarse-to-fine 的特征金字塔上迭代精化视差；推理时使用 stacked cascaded 架构处理高分辨率输入
3. **新合成数据集**: 使用 Blender 渲染更丰富的形状/纹理/光照变化（含透明、金属反射、开孔结构），显著提升真实世界泛化能力

---

## 问题背景

### 要解决的问题
消费级设备（智能手机双摄）拍摄的立体图像存在三大实际问题：精细结构恢复难、非理想极线校正、泛化到真实场景。

### 现有方法的局限
- 现有方法假设完美极线校正，对应点在同一水平线上——实际双摄镜头焦距不同、存在残余畸变，校正后仍有垂直偏移
- 精细结构（猫胡须、铁丝网）在低分辨率特征图上丢失
- 合成训练数据形状/纹理多样性不足，模型在真实场景中泛化差

### 本文的动机
三管齐下：(1) 自适应关联层应对校正误差 (2) 级联递归网络保留精细细节 (3) 更丰富的合成数据提升泛化。

---

## 方法详解

### 模型架构

CREStereo 采用 **级联递归精化** 架构：

- **输入**: 立体图像对 $I_1, I_2$
- **Feature Extraction**: 共享权重的特征提取网络 → 3 级特征金字塔（1/16, 1/8, 1/4）
- **Positional Encoding + Self-Attention**: 对特征图添加位置编码后用线性注意力聚合全局上下文
- **Cascaded RUM**: 3 个级联阶段，各阶段在 1/16 → 1/8 → 1/4 分辨率上递归精化视差
- **Stacked Cascades (推理)**: 高分辨率图像输入时构建图像金字塔，多阶段依次处理不同分辨率层级
- **输出**: 全分辨率视差图，通过 [[Convex Upsampling]] 从 1/4 分辨率上采样

### 核心模块

#### 模块1: Adaptive Group Correlation Layer (AGCL)

**设计动机**: 解决两个问题——(1) 全局关联计算量大且引入噪声 (2) 非理想校正导致对应点不在同一水平线上。

**具体实现**:
- **Local Feature Attention**: 在关联计算前，对特征图添加 [[Positional Encoding]] 并用线性 [[Self-Attention]] 聚合全局上下文
- **2D-1D Alternate Local Search**: 不做全局匹配，仅在局部窗口内搜索
  - 1D 搜索：沿极线方向，$f(d) \in [-r, r]$, $g(d) = 0$, $r = 4$
  - 2D 搜索：$k \times k$ 网格（$k = \sqrt{2r+1}$），带 dilation $l$，类似 [[Dilated Convolution]]
  - 1D 和 2D 交替执行，配合迭代重采样实现传播
- **Deformable Search Window**: 学习额外偏移 $\text{d}x, \text{d}y$，使搜索窗口自适应变形（类似 [[Deformable Convolution]]），处理遮挡和无纹理区域
- **Group-wise Correlation**: 将特征分为 $\mathcal{G}$ 组，各组独立计算关联并拼接，输出 $\mathcal{G}D \times H \times W$

#### 模块2: Recurrent Update Module (RUM)

**设计动机**: 在多个分辨率上递归精化，低分辨率提供鲁棒性和大感受野，高分辨率恢复细节。

**具体实现**:
- 基于 [[GRU]] blocks 构建，与 AGCL 配合
- "Sampler" 从当前预测 $f_n$ 采样分组特征的坐标网格
- $\{f_1, ..., f_n\}$ 为中间预测序列，$f_0$ 为初始化（全零）
- 关联体积用学习到的偏移构建 $o \in \mathbb{R}^{2 \times (2r+1) \times h \times w}$
- GRU 更新当前预测并将偏移反馈给下一轮 AGCL
- 各级联阶段的 RUM 共享权重

#### 模块3: Stacked Cascades for Inference

**设计动机**: 训练时用 3 级特征金字塔固定分辨率精化，但推理时高分辨率输入需要更大感受野。

**具体实现**:
- 将高分辨率输入构建图像金字塔（多次下采样）
- 各下采样层级分别送入共享权重的特征提取网络
- 多阶段级联：1 stage、2 stages、3 stages 按图像金字塔层级组合
- 低分辨率阶段处理大尺度结构，高分辨率阶段恢复细节
- 跳跃连接连接各阶段的 RUM 输出
- 无需微调，训练时共享权重自然适用

---

## 关键公式

### 公式1: [[Correlation Volume|Adaptive Local Correlation]]

$$
\text{Corr}(x, y, d) = \frac{1}{C} \sum_{i=1}^{C} \mathbf{F}_1(i, x, y) \mathbf{F}_2(i, x', y')
$$

其中 $x' = x + f(d), \; y' = y + g(d)$

**含义**: 在位置 $(x,y)$ 处，第 $d$ 个关联对的匹配代价为重采样后左右特征的归一化内积

**符号说明**:
- $\mathbf{F}_1, \mathbf{F}_2$: 左右图的注意力增强特征图
- $C$: 特征通道数
- $f(d), g(d)$: 第 $d$ 个关联对的水平和垂直固定偏移（1D 模式 $g(d)=0$；2D 模式为网格偏移）
- $D$: 局部关联对数量（非全局视差范围）

### 公式2: [[Correlation Volume|Deformable Adaptive Correlation]]

$$
\text{Corr}(x, y, d) = \frac{1}{C} \sum_{i=1}^{C} \mathbf{F}_1(i, x, y) \mathbf{F}_2(i, x'', y'')
$$

其中 $x'' = x + f(d) + \text{d}x, \; y'' = y + g(d) + \text{d}y$

**含义**: 在固定偏移基础上增加可学习的额外偏移，使搜索窗口自适应变形

**符号说明**:
- $\text{d}x, \text{d}y$: 网络预测的额外偏移量（类似 deformable convolution）

### 公式3: [[L1 Loss|级联加权 L1 损失]]

$$
\mathcal{L} = \sum_s \sum_{i=1}^{n} \gamma^{n-i} ||\mathbf{d}_{\text{gt}} - \mu_s(\mathbf{f}^s_i)||_1
$$

**含义**: 对各级联阶段 $s$ 和各迭代步 $i$ 的预测施加指数加权 L1 监督（$\gamma = 0.9$）

**符号说明**:
- $s \in \{\frac{1}{16}, \frac{1}{8}, \frac{1}{4}\}$: 特征金字塔各级
- $\mathbf{f}^s_i$: 阶段 $s$ 第 $i$ 次迭代的预测
- $\mu_s$: 上采样到预测分辨率的算子
- $\gamma = 0.9$: 指数衰减权重

---

## 关键图表

### Figure 1: Holopix50K 实际效果

![[CREStereo_fig1.png|600]]

**说明**: Holopix50K 数据集上的预测结果。CREStereo 在精细结构（铁丝网、猫胡须）和纹理丰富区域均展现高质量视差估计。

### Figure 2: Overview / 整体架构

![[CREStereo_fig2.png|600]]

**说明**: 左：训练阶段，3 级特征金字塔 + 3 阶段级联 RUM，每阶段前一阶段的输出作为下一阶段初始化。右：推理阶段的 stacked cascaded 架构，图像金字塔多分辨率输入。

### Figure 3: RUM + AGCL 模块详解

![[CREStereo_fig3.png|600]]

**说明**: 左：Recurrent Update Module (RUM)，GRU 迭代更新视差并将偏移反馈给 AGCL。右：AGCL 详细流程，右图特征经 cross-attention → 分组 → sampler 采样 → 自适应局部关联 → 拼接输出。

### Figure 4: Adaptive Local Correlation / 自适应局部搜索

![[CREStereo_fig4.png|600]]

**说明**: 2D 搜索（上）和 1D 搜索（下）的示意图。两种模式搜索相同数量的邻居但采用不同的空间分布，配合可学习偏移（offset）实现窗口自适应变形。

### Figure 5: 合成数据集示例

![[CREStereo_fig5.png|600]]

**说明**: 新合成数据集包含丰富的形状（ShapeNet 物体、树木、开孔结构）、纹理（重复纹理、无纹理、金属反射）和光照变化，针对真实场景难例设计。

### Figure 6: 训练损失与泛化对比

![[CREStereo_fig6.png|600]]

**说明**: 与 SceneFlow 对比，CREStereo 的合成数据集训练损失更低，ETH3D 和 Middlebury 验证误差也更低，证明数据质量对泛化的重要性。

### Figure 7: Middlebury & ETH3D 定性对比

![[CREStereo_fig7.png|600]]

**说明**: 与 HITNet、RAFT-Stereo 在 Middlebury 和 ETH3D 上的视觉对比。CREStereo 在精细结构边界和纹理区域更准确。

### Figure 8: KITTI 2015 定性对比

![[CREStereo_fig8.png|600]]

**说明**: 与 LEAStereo、AANet 在 KITTI 2015 上的对比。CREStereo 保留更多细节。

### Figure 9: Holopix50K 多方法定性对比

![[CREStereo_fig9.png|600]]

**说明**: 在 Holopix50K 真实世界数据上与 AANet、HSMNet、GwcNet、LEAStereo、RAFT-Stereo 的全面对比。CREStereo 在所有场景中视觉效果最优。

### Figure 10: 扰动鲁棒性

![[CREStereo_fig10.png|600]]

**说明**: ETH3D 上施加不同扰动（模糊、颜色变换、噪声、旋转、垂直偏移、畸变）后的平均误差对比。CREStereo 在所有扰动类型下均最鲁棒。

### Figure 11: 手机拍摄场景对比

![[CREStereo_fig11.png|600]]

**说明**: 智能手机拍摄的重复纹理和无纹理场景。CREStereo mxIoU 97.87%/99.52%，显著优于 RAFT-Stereo 的 66.95%/73.74%。

### Table 1: RUM 消融实验

| Method | Middlebury Bad 2.0 | Middlebury AvgErr | ETH3D Bad 1.0 | ETH3D AvgErr |
|--------|-------------------|-------------------|---------------|-------------|
| 2D all-pairs | 47.38 | 5.62 | 6.17 | 0.38 |
| 1D all-pairs | 44.41 | 4.93 | 6.03 | 0.38 |
| 1D local | 19.87 | 3.03 | 3.13 | 0.28 |
| 2D local | 20.70 | 2.99 | 3.33 | 0.29 |
| 1D+2D local | 19.23 | 3.01 | 3.05 | 0.28 |
| 1D local, 2 levels | 13.84 | 2.24 | 2.35 | 0.23 |
| 2D local, 2 levels | 14.07 | 2.15 | 2.09 | 0.23 |
| 1D+2D local, 2 levels | **12.48** | **1.99** | 2.20 | **0.22** |
| **1D+2D local, 3 levels** | 12.67 | 1.80 | **2.01** | 0.21 |
| w/o def. & group. & atten. | 6.86 | 1.11 | 1.26 | 0.19 |
| w/o deformable search | 6.84 | 1.08 | 1.22 | 0.19 |
| w/o group-wise correlation | 6.82 | 1.08 | 1.20 | 0.18 |
| w/o attention | 6.49 | 1.07 | 1.22 | 0.18 |
| **full method** | **6.46** | **1.05** | **1.03** | **0.17** |

**关键发现**: 局部关联远优于全局关联；2D+1D 交替优于单一模式；级联层级越多越好；AGCL 中各组件（deformable、group-wise、attention）都有贡献。

### Table 2: Stacked Cascades 消融

| Method | Input size | Middlebury Bad 2.0 | Middlebury AvgErr |
|--------|-----------|-------------------|-------------------|
| single cascade | 768x1024 | 6.46 | 1.05 |
| single cascade | 1024x1536 | 6.00 | 1.61 |
| 2 stacked cascades | 1024x1536 | 5.30 | 0.94 |
| 2 stacked cascades | 1536x2048 | **4.53** | **0.93** |
| 3 stacked cascades | 1536x2048 | 4.58 | **0.92** |

**说明**: 单 cascade 分辨率提高反而误差增大；stacked cascades 有效利用多分辨率信息。

### Table 3: Middlebury 排行榜

| Method | Bad 2.0 | Bad 1.0 | AvgErr | RMS | A95 |
|--------|---------|---------|--------|-----|-----|
| **CREStereo** | **3.71** | **8.25** | **1.15** | **7.70** | **1.58** |
| RAFT-Stereo | 4.74 | 9.37 | 1.27 | 8.41 | 2.29 |
| LocalExp | 5.43 | 13.9 | 2.24 | 13.4 | 4.81 |
| HITNet | 6.46 | 13.3 | 1.71 | 9.97 | 4.26 |
| LEAStereo | 7.15 | 20.8 | 1.43 | 8.11 | 2.65 |

**说明**: CREStereo 在 Middlebury 排行榜 120+ 方法中大多数指标排名第一，Bad 2.0 超越次优 RAFT-Stereo 21.73%。

### Table 4: ETH3D 排行榜

| Method | Bad 1.0 | Bad 0.5 | AvgErr | RMSE |
|--------|---------|---------|--------|------|
| **CREStereo** | **0.98** | **3.58** | **0.13** | **0.28** |
| RAFT-Stereo | 2.44 | 7.04 | 0.18 | 0.36 |
| HITNet | 2.79 | 7.83 | 0.20 | 0.46 |
| AdaStereo | 3.09 | 10.22 | 0.24 | 0.44 |

**说明**: ETH3D 排行榜，CREStereo Bad 1.0 超越 RAFT-Stereo 59.84%。

### Table 5: 智能手机场景定量评测

| Method | mxIoU | mxIoUbd |
|--------|-------|---------|
| **Ours** | **97.50%** | **72.61%** |
| RAFT-Stereo | 94.58% | 69.26% |
| HSMNet | 91.70% | 60.17% |
| AANet | 91.02% | 63.70% |
| GwcNet | 90.77% | 64.26% |
| STTR | 90.82% | 62.12% |
| LEAStereo | 92.38% | 58.06% |

**说明**: 400 张智能手机拍摄场景上，CREStereo 的 mxIoU 和 mxIoUbd 均最优。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| CREStereo 合成数据 | 自制 | Blender 渲染，ShapeNet+树+开孔结构 | 主训练 |
| [[SceneFlow]] | 39K | 合成多场景 | 辅助训练 |
| Sintel | 1.2K | 电影级合成 | 辅助训练 |
| Falling Things | — | 散落物体 | 辅助训练 |
| InStereo2K, CARLA, AirSim | — | 多源合成 | 辅助训练 |
| [[Middlebury]] 2014 | 23 对 | 高分辨率室内 | 微调 + 评测 |
| [[ETH3D]] | 27 对 | 灰度高分辨率 | 评测 |
| [[KITTI-2012]]/[[KITTI-2015]] | 200 对 | 驾驶场景 | 微调 + 评测 |
| Holopix50K | — | 真实世界双摄 | 评测 |

### 实现细节

- **框架**: PyTorch
- **优化器**: [[Adam]]，学习率 0.0004
- **Batch Size**: 16
- **训练步数**: 300K iterations
- **Warm-up**: 6K iterations 线性增加到标准学习率
- **学习率调度**: 180K 后线性衰减至 5%
- **训练分辨率**: 384x512
- **硬件**: 8x NVIDIA GTX 2080Ti GPU
- **数据增强**: 不对称色彩增强 + 右图 homography 扰动（<2px）+ 随机遮挡补丁 + 随机 resize/crop
- **Middlebury 微调**: Middlebury 占训练集 2%
- **KITTI 微调**: 额外 50K iterations，学习率 0.0001，KITTI 占 75%

### 可视化结果

- 在 Holopix50K 真实场景中恢复猫胡须、铁丝网等极精细结构
- 对非理想校正的智能手机图像鲁棒
- 扰动实验表明对模糊、噪声、旋转、畸变等均鲁棒

---

## 批判性思考

### 优点
1. **面向实际应用的系统设计**: 同时解决校正误差、精细结构、泛化三大实际问题
2. **AGCL 设计精巧**: 2D-1D 交替搜索 + 可变形窗口 + 组关联的组合有效应对非理想校正
3. **Stacked cascades 优雅**: 训练和推理架构统一，共享权重无需额外微调
4. **排行榜双冠**: Middlebury 和 ETH3D 同时第一，在真实场景也表现优异

### 局限性
1. **推理速度不足**: 论文承认无法在移动设备上实时运行，未报告具体推理时间
2. **合成数据集未公开**: 数据生成细节虽有描述但具体数据集未开源（只开源了模型）
3. **stacked cascades 增加推理开销**: 高分辨率输入需要多阶段处理，延迟线性增长

### 潜在改进方向
1. 轻量化适配移动端部署（后续 CREStereo++ 可能解决）
2. 结合预训练视觉基础模型（如 FoundationStereo 的方向）
3. 支持时序信息实现视频立体匹配

### 可复现性评估
- [x] 代码开源（MegEngine 实现）
- [x] 预训练模型
- [x] 训练细节完整
- [ ] 合成数据集未公开

---

## 关联笔记

### 基于
- [[RAFT-Stereo]]: 迭代精化范式的直接前身
- [[RAFT]]: GRU-based 迭代更新框架
- [[LoFTR]]: 注意力机制用于特征匹配的启发

### 对比
- [[RAFT-Stereo]]: CREStereo 在 Middlebury/ETH3D 上大幅超越
- [[HITNet]]: tile-based 实时方法，精度较低
- [[LEAStereo]]: NAS 方法
- [[AANet]]: 自适应聚合方法

### 方法相关
- [[Correlation Volume]]: AGCL 的改进基础
- [[GRU]]: RUM 的核心组件
- [[Deformable Convolution]]: 可变形搜索窗口的灵感来源
- [[Self-Attention]]: 特征增强中使用
- [[Positional Encoding]]: 增强特征的位置依赖性
- [[Convex Upsampling]]: 最终上采样方法

### 硬件/数据相关
- [[SceneFlow]]: 辅助训练数据
- [[Middlebury]]: 主要评测 benchmark（第一名）
- [[ETH3D]]: 主要评测 benchmark（第一名）
- [[KITTI-2015]]: 评测 benchmark

---

## 速查卡片

> [!summary] CREStereo: Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation
> - **核心**: 自适应组关联层处理非理想校正 + 级联递归网络恢复精细细节
> - **方法**: AGCL (2D-1D alternate + deformable + group-wise) + Cascaded RUM + Stacked Cascades
> - **结果**: Middlebury 第一 (Bad 2.0: 3.71)，ETH3D 第一 (Bad 1.0: 0.98)
> - **代码**: https://github.com/megvii-research/CREStereo

---

*笔记创建时间: 2026-05-12*
