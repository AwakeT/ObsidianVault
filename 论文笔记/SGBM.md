---
title: "Stereo Processing by Semiglobal Matching and Mutual Information"
method_name: "SGM / SGBM"
authors: [Heiko Hirschmüller]
year: 2008
venue: IEEE TPAMI 2008
tags: [stereo-matching, semi-global-matching, classical-method, dynamic-programming, mutual-information, depth-estimation, cost-aggregation]
image_source: none
created: 2026-05-12
---

# 论文笔记：Stereo Processing by Semiglobal Matching and Mutual Information

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | German Aerospace Center (DLR) |
| 日期 | February 2008（原始版本 CVPR 2005） |
| 项目主页 | — |
| 对比基线 | Graph Cut, Belief Propagation, Dynamic Programming, Local Methods |
| 链接 | [IEEE Xplore](https://ieeexplore.ieee.org/document/4359315/) / [OpenCV SGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html) |

---

## 一句话总结

> 将 NP-hard 的 2D MRF 全局优化近似为多方向 1D 扫描线动态规划求和，配合互信息匹配代价，实现高精度实时立体匹配。

---

## 核心贡献

1. **Semi-Global Matching (SGM)**: 将 2D MRF 优化近似为沿 8 或 16 个方向的 1D 扫描线优化，通过聚合各方向代价近似全局最优，复杂度线性于像素数和视差范围
2. **Mutual Information 匹配代价**: 基于 [[Mutual Information|互信息]] 的像素级匹配代价，对辐射变化（光照差异、不同传感器）鲁棒
3. **完整的后处理流水线**: 遮挡检测（左右一致性检查）、亚像素精化（抛物线插值）、散斑滤波、中值滤波等后处理步骤
4. **实用性**: 成为工业界和学术界最广泛使用的经典立体匹配方法，OpenCV 中实现为 `StereoSGBM`

---

## 问题背景

### 要解决的问题
从校正后的立体图像对中估计稠密视差图。全局方法（Graph Cut, BP）精度高但计算量大，局部方法速度快但精度低。

### 现有方法的局限
- **全局方法** (Graph Cut, Belief Propagation): 2D MRF 优化是 NP-hard 问题，计算代价极高
- **局部方法** (Block Matching): 快速但对无纹理区域、遮挡和深度不连续处理差
- **扫描线优化** (Dynamic Programming): 仅沿单一方向优化，产生明显的条纹伪影（streak artifacts）

### 本文的动机
通过沿多个方向执行 1D 优化并聚合结果，在计算效率（线性复杂度）和匹配精度（近似全局最优）之间取得最佳平衡。

---

## 方法详解

### 算法架构

SGM 由四个阶段构成：

- **匹配代价计算**: 逐像素计算左右图在各视差下的匹配代价 $C(p, d)$
- **代价聚合**: 沿 8 或 16 个方向执行扫描线优化，聚合为 $S(p, d)$
- **视差选择**: Winner-Takes-All (WTA) 从聚合代价中选取最优视差
- **后处理**: 左右一致性检查、亚像素精化、散斑滤波

### 核心模块

#### 模块1: Mutual Information 匹配代价

**设计动机**: 传统匹配代价（SAD, SSD）对光照变化敏感。互信息度量两幅图像的统计依赖关系，对单调辐射变换不变。

**具体实现**:
- 基于联合灰度直方图 $H_{I_L, I_R}$ 和边缘灰度直方图 $H_{I_L}$, $H_{I_R}$
- 互信息 $MI = H_{I_L} + H_{I_R} - H_{I_L, I_R}$
- 像素级 MI 代价需要用当前视差估计迭代计算直方图
- 初始用 Census Transform 或 BT (Birchfield-Tomasi) 代价启动
- 通常迭代 2-3 次收敛

#### 模块2: Semi-Global 代价聚合

**设计动机**: 2D MRF 能量最小化是 NP-hard。沿单方向 DP 会有条纹伪影。沿多方向聚合可有效抑制条纹。

**具体实现**:
- 对每个方向 $\mathbf{r}$（共 8 或 16 个方向），沿路径递归计算 $L_\mathbf{r}(p, d)$
- 路径代价包含匹配代价项 + 平滑约束项（P1 惩罚 1 像素视差变化，P2 惩罚更大变化）
- 减去路径最小值防止代价单调递增
- 所有方向的路径代价求和得到聚合代价 $S(p, d) = \sum_\mathbf{r} L_\mathbf{r}(p, d)$
- 视差选择: $d^*(p) = \arg\min_d S(p, d)$

#### 模块3: 后处理

**设计动机**: 遮挡区域无有效匹配，散斑噪声影响视差图质量。

**具体实现**:
- **左右一致性检查**: 分别计算左→右和右→左视差，不一致的像素标记为遮挡
- **亚像素精化**: 对 $S(p, d)$ 在最优视差附近做抛物线插值
- **散斑滤波**: 连通域分析，移除小于阈值的视差区域
- **中值滤波**: 可选的平滑步骤

---

## 关键公式

### 公式1: [[Energy Function|全局能量函数]]

$$
E(D) = \sum_p \left( C(p, d_p) + \sum_{q \in N_p} P_1 \cdot \mathbb{1}[|d_p - d_q| = 1] + \sum_{q \in N_p} P_2 \cdot \mathbb{1}[|d_p - d_q| > 1] \right)
$$

**含义**: 全局能量由数据项（匹配代价）和平滑项（视差变化惩罚）构成，最小化此能量即为 2D MRF 优化

**符号说明**:
- $D$: 视差图
- $C(p, d_p)$: 像素 $p$ 在视差 $d_p$ 处的匹配代价
- $N_p$: 像素 $p$ 的邻域
- $P_1$: 小惩罚，应对倾斜/曲面的 1 像素视差变化
- $P_2$: 大惩罚（$P_2 > P_1$），抑制视差不连续

### 公式2: [[Dynamic Programming|路径代价递推]]

$$
L_\mathbf{r}(p, d) = C(p, d) + \min \begin{cases} L_\mathbf{r}(p-\mathbf{r}, d) \\ L_\mathbf{r}(p-\mathbf{r}, d-1) + P_1 \\ L_\mathbf{r}(p-\mathbf{r}, d+1) + P_1 \\ \min_i L_\mathbf{r}(p-\mathbf{r}, i) + P_2 \end{cases} - \min_k L_\mathbf{r}(p-\mathbf{r}, k)
$$

**含义**: 沿方向 $\mathbf{r}$ 递推时，路径代价 = 匹配代价 + 前一步最小代价（含平滑惩罚）- 正则化项

**符号说明**:
- $L_\mathbf{r}(p, d)$: 沿方向 $\mathbf{r}$ 到达像素 $p$ 、视差 $d$ 的路径代价
- $p - \mathbf{r}$: 沿方向 $\mathbf{r}$ 的前一个像素
- 四个候选：保持视差 / 变化 $\pm 1$（加 $P_1$） / 跳变（加 $P_2$）
- 最后减去 $\min_k$ 项防止代价无限增长

### 公式3: [[Cost Aggregation|多方向代价聚合]]

$$
S(p, d) = \sum_\mathbf{r} L_\mathbf{r}(p, d)
$$

**含义**: 将所有方向（通常 8 或 16 个）的路径代价求和，近似 2D 全局优化

### 公式4: [[Winner-Takes-All|视差选择]]

$$
d^*(p) = \arg\min_d S(p, d)
$$

**含义**: 在聚合代价最小处选取最优视差

### 公式5: [[Mutual Information|互信息匹配代价]]

$$
MI_{I_L, I_R} = H_{I_L} + H_{I_R} - H_{I_L, I_R}
$$

$$
C_{MI}(p, d) = -mi_{I_L, I_R}(I_L(p), I_R(p - d))
$$

**含义**: 像素级互信息代价由联合熵和边缘熵计算，对辐射变换不变

**符号说明**:
- $H_{I_L}$, $H_{I_R}$: 左右图的边缘熵
- $H_{I_L, I_R}$: 联合熵
- $mi_{I_L, I_R}$: 从联合/边缘直方图导出的像素级互信息值

---

## 关键图表

> **注意**: SGM 为经典算法论文（IEEE TPAMI 2008），原文图表未在线公开。以下为文字描述。

### 算法流程示意

```
左图/右图 → 匹配代价 C(p,d)
              ↓
    ┌─────────────────────┐
    │ 8/16 方向扫描线 DP    │
    │ L_r(p,d) for r=1..R │
    └─────────────────────┘
              ↓
    聚合: S(p,d) = ΣL_r(p,d)
              ↓
    WTA: d*(p) = argmin S(p,d)
              ↓
    后处理: LR check → 亚像素 → 散斑滤波
              ↓
          视差图 D
```

### SGM 与其他方法的精度-速度权衡

| 类别 | 代表方法 | 精度 | 速度 | 特点 |
|------|---------|------|------|------|
| 全局 | Graph Cut, BP | 高 | 慢（秒-分钟） | 2D MRF 精确优化 |
| **半全局** | **SGM** | **较高** | **快（1-2s）** | **多方向 1D DP 近似** |
| 局部 | Block Matching | 低 | 极快 | 窗口匹配 |

---

## OpenCV SGBM 实现

### 与原始 SGM 的差异

| 特征 | 原始 SGM (Hirschmüller) | OpenCV StereoSGBM |
|------|------------------------|-------------------|
| 匹配代价 | [[Mutual Information]] | Birchfield-Tomasi (BT) |
| 匹配单位 | 像素级 | 块匹配（blockSize=1 时等价像素级） |
| 方向数 | 8 或 16 | 5（MODE_SGBM）/ 8（MODE_HH） |
| 后处理 | 论文描述 | 包含 StereoBM 的预/后滤波 |

### OpenCV 模式

| 模式 | 方向数 | 描述 |
|------|--------|------|
| `MODE_SGBM` (默认) | 5 | 单遍扫描，快但较不准确 |
| `MODE_HH` | 8 | 双遍全 DP，最接近原始 SGM |
| `MODE_SGBM_3WAY` | 3 | 优化实现 |
| `MODE_HH4` | 4 | 4 方向变体 |

### 关键参数

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `minDisparity` | 最小视差 | 0 |
| `numDisparities` | 视差搜索范围（必须 16 的倍数） | 64-256 |
| `blockSize` | 匹配块大小（奇数） | 3-11 |
| `P1` | 小惩罚（1 像素变化） | $8 \times cn \times blockSize^2$ |
| `P2` | 大惩罚（>1 像素变化） | $32 \times cn \times blockSize^2$ |
| `disp12MaxDiff` | 左右一致性检查阈值 | 1 |
| `uniquenessRatio` | 唯一性比率 (%) | 10-15 |
| `speckleWindowSize` | 散斑滤波窗口 | 100-200 |
| `speckleRange` | 散斑滤波范围 | 1-2 |

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| [[Middlebury]] | 高分辨率室内 | 评测（当时排名前列） |
| 航空影像 | 大规模推扫式相机 | 实际应用验证 |

### 性能

- 在 Middlebury 标准评测中跻身当时最优方法，尤其在亚像素精度方面表现最佳
- 运行时间 1-2 秒（标准测试图像），线性复杂度 $O(W \times H \times D \times R)$
  - $W \times H$: 图像尺寸
  - $D$: 视差范围
  - $R$: 方向数（8 或 16）
- 内存占用: $O(W \times H \times D)$

---

## 批判性思考

### 优点
1. **精度与效率的最佳平衡**: 线性复杂度下近似全局优化，在实际应用中表现出色
2. **广泛适用性**: 从航空摄影到自动驾驶到消费级产品，SGM 是工业界使用最广泛的立体匹配方法
3. **理论简洁**: 多方向 1D DP 聚合的思想简单直观
4. **互信息代价的鲁棒性**: 对传感器差异、光照变化容忍度高
5. **完整的后处理流水线**: 论文提供了可直接工程化的完整系统

### 局限性
1. **无纹理区域表现差**: 匹配代价在无纹理区域模糊，多方向聚合仍难以完全解决
2. **P1/P2 参数敏感**: 惩罚参数需针对场景调优，固定参数难以适应所有场景
3. **边缘模糊**: 在深度不连续处视差容易过度平滑
4. **无语义理解**: 纯几何方法，无法利用高层语义信息处理遮挡和模糊匹配
5. **非端到端学习**: 各模块分离设计，无法联合优化

### 后续改进
1. **SGM-Nets** (CVPR 2017): 用 CNN 学习 P1/P2 惩罚参数
2. **MC-CNN + SGM**: 用 CNN 学习匹配代价，SGM 做聚合
3. **Adaptive P2**: 根据图像梯度自适应调整 P2
4. **自适应路径数**: 根据场景复杂度动态选择聚合方向数

### 可复现性评估
- [x] 算法描述完整
- [x] OpenCV 公开实现
- [ ] 原始代码（DLR 内部实现未开源）
- [x] 评测数据集可获取

---

## 关联笔记

### 基于
- [[Markov Random Field]]: 能量函数的理论基础
- [[Dynamic Programming]]: 1D 扫描线优化的核心算法
- [[Mutual Information]]: 匹配代价函数

### 对比（深度学习方法）
- [[RAFT-Stereo]]: 迭代精化方法，精度远超 SGM 但需 GPU
- [[CREStereo]]: 自适应关联层，处理非理想校正
- [[HITNet]]: tile-based 实时方法，借鉴了 SGM 的空间传播思想
- [[FoundationStereo]]: 零样本基础模型，代表深度学习 vs 经典方法的差距
- [[MobileStereoNet]]: 轻量深度学习方法

### 方法相关
- [[Cost Aggregation]]: SGM 的核心——多方向代价聚合
- [[Winner-Takes-All]]: 视差选择策略
- [[Census Transform]]: 常用替代匹配代价
- [[Birchfield-Tomasi]]: OpenCV SGBM 使用的匹配代价
- [[Left-Right Consistency]]: 遮挡检测后处理

### 硬件/数据相关
- [[Middlebury]]: 经典立体评测 benchmark
- [[OpenCV]]: 最广泛使用的 SGBM 实现

---

## 速查卡片

> [!summary] SGM: Stereo Processing by Semiglobal Matching and Mutual Information
> - **核心**: 多方向 1D 扫描线 DP 近似 2D 全局优化，线性复杂度
> - **方法**: Mutual Information 匹配代价 + 8/16 方向路径代价聚合 + P1/P2 平滑惩罚 + WTA + 后处理
> - **结果**: 精度接近全局方法，速度 1-2s，成为工业界标准
> - **实现**: OpenCV `StereoSGBM`（用 BT 代价替代 MI，默认 5 方向）

---

*笔记创建时间: 2026-05-12*
