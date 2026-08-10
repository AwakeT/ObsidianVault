---
title: "Vision Pretraining for Dense Spatial Perception"
method_name: "LingBot-Vision"
authors: [Zelin Fu, Bin Tan, Changjiang Sun, Shaohui Liu, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, Nan Xue]
year: 2026
venue: arXiv
tags: [vision-pretraining, dense-spatial-perception, self-supervised-learning, boundary-modeling, depth-completion, visual-foundation-model]
zotero_collection: 立体匹配与深度估计
image_source: local
arxiv_html: https://arxiv.org/html/2607.05247v1
created: 2026-07-09
---

# 论文笔记：Vision Pretraining for Dense Spatial Perception

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | LingBot-Vision |
| 机构 | Robbyant / 作者团队未在首页展开列出完整机构 |
| 日期 | July 2026 |
| 项目主页 | https://technology.robbyant.com/lingbot-vision |
| 代码 | https://github.com/robbyant/lingbot-vision |
| 模型权重 | https://huggingface.co/collections/robbyant/lingbot-vision |
| 链接 | [arXiv](https://arxiv.org/abs/2607.05247) / [HTML](https://arxiv.org/html/2607.05247v1) |
| 对比基线 | [[DINOv2]], [[DINOv3]], [[V-JEPA 2.1]], [[SigLIP 2]], [[LingBot-Depth]] |

---

## 一句话总结

> LingBot-Vision 把边界和形状不连续性从“下游输出”提升为[[自监督学习|自监督预训练]]的原生学习信号，通过边界强制掩码、分类化边界场和在线 a-contrario 验证，让 ViT 学到更适合深度、分割、视频目标传播和深度补全的稠密空间表示。

---

## 核心贡献

1. **边界中心的视觉预训练范式**：作者主张边界不是任务头需要预测的结果，而是支撑[[稠密预测]]、深度、分割和机器人感知的基础组织信号，可以从原始图像中自举出来。
2. **Masked Boundary Modeling**：用 teacher 在线发现的边界 token 决定 student 必须 mask 哪些 patch，将最难由上下文补全的边界区域变成训练目标。
3. **Boundary-Forcing + Geometry Routing**：边界 token 同时接收语义 iBOT 监督和几何边界场监督；非边界 mask token 仍走普通语义自蒸馏，从而避免语义抽象和几何敏感性互相冲突。
4. **边界场分类化重参数化**：把连续边界场回归改成每像素、每通道的离散 bin 分类，使边界分支能够继承 DINO/iBOT 的 centering、sharpening 和防坍塌机制。
5. **大规模 LingBot-Vision 与 LingBot-Depth 2.0**：在 160.75M 图像上训练约 1.1B 参数 ViT-g/16；在 NYUv2、KITTI、ADE20K、DAVIS、YouTube-VOS 等 dense 任务上接近或超过更大模型，并将 LingBot-Depth 升级到 2.0。

---

## 问题背景

### 要解决的问题

现代[[视觉基础模型]]常常擅长识别“图像里有什么”，但对物体边界、深度不连续、遮挡轮廓、运动边缘等细粒度空间结构不够敏感。对于[[具身智能]]和机器人来说，这种空间结构不是锦上添花，而是可测量、可导航、可操作世界的基础。

### 现有方法的局限

- [[DINO]] / [[iBOT]] / [[DINOv2]] / [[DINOv3]] 这条[[自蒸馏]]路线能产生一定的 object layout，但边界只是副产物；dense feature 质量容易在长训练中退化，DINOv3 还需要 Gram anchoring 来补救。
- [[Masked Image Modeling]] 直接重建像素，空间保真度强，但作为冻结语义表示通常弱于自蒸馏模型。
- [[CLIP]] / [[SigLIP]] 这类语言对齐模型偏向整图语义，patch-level dense prediction 通常较弱。
- 边界、线段、轮廓过去多被当成监督任务输出，需要人工标注、边缘检测器或任务专用 head，不适合作为大规模预训练目标。

### 本文动机

作者认为边界和形状不连续性是视觉空间结构的“压缩表示”：深度 discontinuity、物体 mask、motion silhouette、occlusion contour 都在边界处显现。**若能在预训练阶段让模型主动学习这些区域，就可能得到同时保留语义分组和几何细节的视觉表示。**

---

## 方法详解

### 模型结构

LingBot-Vision 仍然采用 teacher-student [[Vision Transformer]] 自蒸馏框架，但在标准 DINO/iBOT 上增加一个边界建模分支：

- **输入**：多尺度增强视图；global view 用于边界目标生成，local view 主要用于 image-level distillation。
- **Backbone**：ViT-g/16，约 1.1B 参数，SwiGLU FFN，RoPE，4 个 register tokens。
- **语义分支**：class token 走 [[DINO]] loss；masked patch token 走 [[iBOT]] loss。
- **边界分支**：teacher 从自身边界场预测中解码并验证线段，生成边界 token 集合 $B$；student 必须 mask 这些 token，并对它们学习分类化边界场。
- **输出表示**：冻结 patch feature 直接用于 depth linear probing、segmentation linear probing、video object propagation、boundary-token tracking；还可作为 [[LingBot-Depth 2.0]] 的 encoder 初始化。

### 核心模块

#### 模块1：Boundary-Forcing Masked Modeling

**设计动机**：随机 mask 把所有 patch 当成等价目标，但图像内部平坦区域容易由邻域复制，边界 token 才携带最不可替代的结构信息。

**具体做法**：

- teacher 在线预测边界，并把边界 rasterize 到 pixel map。
- 若一条预测边界穿过某个 $P \times P$ patch，则该 patch 对应 token 进入边界 token 集合 $B$。
- student 的 mask 集合从随机 mask $M$ 扩展为 $M^+=M\cup B$。
- 所有 $M^+$ token 都接受 iBOT 语义重建；边界 token 额外接受边界场分类目标。

这一步的关键不是“mask 更多 token”，而是“mask 最有信息量、最难从周围猜出来的 token”。

#### 模块2：Categorical Boundary Field

**设计动机**：直接回归连续边界场在 teacher-student EMA 循环里容易漂移和坍塌，不能使用 DINO 系列成熟的 centering/sharpening 稳定技巧。

**具体做法**：

- 把每个边界场通道离散成 $K$ 个 bin。
- teacher 的连续值被编码为窄的 soft categorical label。
- student 预测每个位置、每个通道上的 bin distribution。
- 非边界区域对应近似 uniform distribution，这自然对应 a-contrario 理论里的 no-structure null hypothesis。

这种设计让“无边界”不需要单独 background 类；偏离 uniform 的程度本身就可以作为线段是否显著的统计证据。

#### 模块3：Online Boundary Target Generation

**设计动机**：模型从随机初始化开始，早期根本没有可靠边界预测，但自蒸馏又必须从第一步就有 teacher target。

**具体做法**：

1. teacher 预测 dense boundary field；
2. 用一个冻结的 single-block ViT 提供稀疏 corner seeds；
3. 结合 corner points 和 boundary field，通过 vote aggregation 解码候选线段；
4. 用 [[a-contrario]] 检验剔除不显著候选；
5. 将保留下来的线段重新渲染为干净的边界场，只用这个 validated target 监督 student。

作者强调，边界目标不是来自外部 detector，也不是人工标注，而是 teacher 每步在线生成并随 EMA teacher 一起演化。

#### 模块4：可扩展训练系统

- 边界 head 只在少量边界 token 上展开到 sub-token resolution，而不是 dense 跑全图。
- soft label 构造和 cross-entropy 被融合为 custom kernels，避免中间大张量。
- teacher-side decoding、corner-line pairing、a-contrario test 和 re-rendering 都做成 batched CUDA kernel。
- 最终 validated boundary target 的生成只占小部分 step time，整体吞吐接近 semantic-only baseline。

---

## 关键公式

### 公式1：[[Exponential Moving Average|EMA Teacher 更新]]

$$
\bar{\theta} \leftarrow \lambda \bar{\theta} + (1-\lambda)\theta
$$

**含义**：teacher 参数 $\bar{\theta}$ 是 student 参数 $\theta$ 的指数滑动平均，训练中 $\lambda$ 逐渐 anneal 到 1，使 teacher target 平滑演化。

**符号说明**：
- $\theta$：student 网络参数。
- $\bar{\theta}$：teacher 网络参数。
- $\lambda$：EMA momentum。

### 公式2：[[DINO|Image-level Distillation 分布]]

$$
p_t=\mathrm{softmax}\left(\frac{h_{\bar{\theta}}(z_{\mathrm{cls}})-c}{\tau_t}\right),\qquad
p_s=\mathrm{softmax}\left(\frac{h_{\theta}(z_{\mathrm{cls}})}{\tau_s}\right)
$$

**含义**：teacher 的 class-token 输出经过 centering 和 sharpening，student 学习匹配 teacher 的类别原型分布。

**符号说明**：
- $h$：class token projection head。
- $z_{\mathrm{cls}}$：ViT class token。
- $c$：teacher output center。
- $\tau_t,\tau_s$：teacher/student temperature，且 $\tau_t<\tau_s$。

### 公式3：[[DINO Loss|DINO 损失]]

$$
\mathcal{L}_{\mathrm{DINO}}=-p_t(x^{(1)})^\top \log p_s(x^{(2)})
$$

**含义**：跨增强视图匹配 teacher 与 student 的 image-level prototype distribution。

### 公式4：[[iBOT|Patch-level Distillation 损失]]

$$
\mathcal{L}_{\mathrm{iBOT}}
=-\frac{1}{|M|}\sum_{i\in M} q_i^t{}^\top\log q_i^s,\qquad
q_i=\mathrm{softmax}\left(\frac{g(z_i^p)-c_{\mathrm{iBOT}}}{\tau}\right)
$$

**含义**：student 对 masked patch token 预测语义 prototype distribution，并匹配 teacher 对应未遮挡 token 的分布。

**符号说明**：
- $M$：随机 mask token 集合。
- $g$：patch projection head。
- $z_i^p$：第 $i$ 个 patch token。

### 公式5：[[Boundary Token|边界 token 集合]]

$$
B=\left\{i\in\{1,\ldots,N\}: \text{a predicted boundary intersects patch}(i)\right\}
$$

**含义**：如果 teacher 在线预测的边界穿过某个 patch，该 patch token 就被视为 boundary token。

### 公式6：[[Masked Boundary Modeling|边界强制 mask]]

$$
M^+=M\cup B
$$

**含义**：student 的最终 mask 集合由随机 mask 和 teacher-discovered boundary token 合并得到，保证边界 token 必定被隐藏并重建。

### 公式7：[[Boundary Field|边界场属性]]

$$
a(p)=\left(d_p,\theta_p,\phi_p^1,\phi_p^2\right)
$$

**含义**：每个靠近线段的像素 $p$ 存储到最近线段的距离、方向和端点角度；这种过参数化表示允许多个 support pixels 对同一线段投票。

**符号说明**：
- $d_p$：像素到线段的距离。
- $\theta_p$：指向线段的方向。
- $\phi_p^1,\phi_p^2$：从像素看线段两个端点的角度。

### 公式8：[[Categorical Reparameterization|边界场 soft label]]

$$
\bar{y}_k^c(p)\propto
\exp\left(-\frac{\delta_c(k,a_c(p))^2}{\tau_\ell}\right),
\qquad k=1,\ldots,K
$$

**含义**：将 teacher 的连续边界场值编码为离散 bin 上的窄 soft label；方向通道使用 circular distance。

**符号说明**：
- $c\in\{d,\theta,\phi^1,\phi^2\}$：边界场通道。
- $K$：每个通道的 bin 数。
- $\delta_c$：bin center 与 teacher value 的距离。
- $\tau_\ell$：label temperature。

### 公式9：[[Boundary Loss|边界分类损失]]

$$
\mathcal{L}_{\mathrm{bnd}}
=-\frac{1}{|B|}\sum_{p\in B}\sum_c \bar{y}^c(p)^\top\log \hat{y}^c(p)
$$

**含义**：只在边界位置上计算 per-channel categorical cross-entropy，使 student 学习 teacher validated boundary field。

### 公式10：[[Multi-Task Learning|总训练目标]]

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{DINO}}
+\lambda_i\mathcal{L}_{\mathrm{iBOT}}
+\lambda_b\mathcal{L}_{\mathrm{bnd}}
+\lambda_k\mathcal{L}_{\mathrm{KoLeo}}
$$

**含义**：完整目标把 image-level 语义、patch-level 语义、boundary geometry 和 KoLeo regularizer 组合起来；边界 token 同时接受 iBOT 和 boundary loss。

**训练设定**：
- $\lambda_i=\lambda_b=1$，$\lambda_k=0.1$。
- teacher target 全部 stop-gradient。
- global view 生成边界目标；local view 只做 image-level distillation。

---

## 关键图表

### Figure 1：Boundary-centric masked modeling 总览

![[assets/LingBot-Vision/LingBot-Vision_fig1_page2.png]]

**说明**：展示输入图像、frozen teacher patch token 的 PCA 投影、通过 a-contrario 验证得到的边界 token，以及由边界 token query 产生的 cosine-similarity map。重点是：模型表示同时具有语义 grouping 和清晰几何边界。

### Figure 2：边界强制 mask 与 geometry routing

![[assets/LingBot-Vision/LingBot-Vision_fig2_page5.png]]

**说明**：toy scene 中，随机 mask 可能漏掉关键边界；本文用 teacher 发现的边界 token 强制进入 mask，并把 boundary token 路由到几何目标，非边界 token 仍走语义自蒸馏。

### Figure 3：边界可以从 corner points 自举

![[assets/LingBot-Vision/LingBot-Vision_fig3_page7.png]]

**说明**：只要有 sparse corner points，即使 boundary field 随机采样，也能解码出 corner-anchored line fragments；当方向通道由图像 level-line orientation 引导时，会得到稳定线段。这解释了为什么从随机初始化开始也能产生初始边界信号。

### Figure 4：Boundary field 表示

![[assets/LingBot-Vision/LingBot-Vision_fig4_page8.png]]

**说明**：线段被提升为 dense field，每个 support pixel 记录距离和角度；反过来，每个 support pixel 又能投票恢复线段。这种冗余性是在线自举和抗噪的关键。

### Figure 5：在线边界目标生成

![[assets/LingBot-Vision/LingBot-Vision_fig5_page10.png]]

**说明**：teacher 早期边界场可能很 noisy，但结合 corner seeds 和 vote aggregation 后能生成候选线段；a-contrario test 再过滤 spurious candidates，最终 re-render 为 clean target field。

### Table 1：ImageNet-1K proof-of-concept 消融

| 方法 | IN-1K k-NN ↑ | NYUv2 δ1 ↑ | NYUv2 RMSE ↓ |
|------|-------------|------------|--------------|
| DINO+iBOT baseline | 81.6 | 81.4 | 0.474 |
| + categorical boundary target | 81.8 | 84.4 | 0.446 |
| + dual supervision | 82.0 | 84.7 | 0.443 |
| + RoPE backbone | 82.4 | 84.9 | 0.440 |
| boundary forcing + semantic only | 81.4 | 81.2 | 0.481 |

**关键发现**：真正带来 dense geometry 提升的是分类化边界目标；只强制 mask 边界但仍用语义目标重建，甚至略低于 baseline。

### Figure 6：Frozen patch feature PCA

![[assets/LingBot-Vision/LingBot-Vision_fig6_page14.png]]

**说明**：LingBot-Vision 的 PCA feature map 比 DINOv2/DINOv3/SigLIP/V-JEPA 等更能保持物体内部一致性和边界清晰度。

### Table 2：Dense visual tasks

| 方法 | 参数 | NYUv2 RMSE ↓ | KITTI RMSE ↓ | ADE20K mIoU | Cityscapes mIoU | VOC mIoU |
|------|------|--------------|--------------|-------------|-----------------|----------|
| DINOv3 | 7B/16 | 0.309 | 2.346 | 55.9 | 81.1 | 86.6 |
| V-JEPA 2.1 ViT-G | 2B/16 | 0.307 | 2.461 | 47.9 | 73.5 | 85.0 |
| DINOv2 | 1B/14 | 0.372 | 2.624 | 49.5 | 75.6 | 83.1 |
| DINOv3 ViT-H+ | 0.8B/16 | 0.352 | 2.635 | 54.8 | 79.5 | 85.8 |
| V-JEPA 2.1 ViT-g | 1B/16 | 0.350 | 2.601 | 47.8 | 71.8 | 84.7 |
| **LingBot-Vision ViT-g** | **1B/16** | **0.296** | **2.552** | **53.5** | **79.6** | **87.5** |

**关键发现**：1B LingBot-Vision 在 NYUv2 上超过 7B DINOv3；KITTI 在 2B 以下模型里最佳；segmentation 接近 DINOv3 ViT-H+，VOC 最优。

### Table 3：Video object segmentation

| 方法 | 参数 | DAVIS-S J&F ↑ | YT-VOS-S J&F ↑ |
|------|------|---------------|----------------|
| DINOv3 | 7B/16 | 71.1 | 74.1 |
| DINOv3 ViT-H+ | 0.8B/16 | 71.1 | 74.0 |
| DINOv2 | 1B/14 | 63.9 | 65.6 |
| V-JEPA 2.1 ViT-g | 1B/16 | 68.1 | 72.3 |
| **LingBot-Vision ViT-g** | **1B/16** | **70.0** | **73.5** |

**关键发现**：没有 temporal supervision，单靠 frozen patch feature 做 label propagation，就接近 DINOv3 7B。

### Figure 7：Boundary-token tracking

![[assets/LingBot-Vision/LingBot-Vision_fig7_page18.png]]

**说明**：在机器人第一视角、猫视频和家庭移动视角中，用 frozen feature 的 cosine similarity 追踪 boundary-token query，说明边界 token 在时间上也形成稳定实体。

### Table 4：ImageNet-1K giant model accuracy

| 模型 | Linear ↑ | k-NN ↑ |
|------|----------|--------|
| LingBot-Vision | 86.32 | 83.39 |
| DINOv2 | 87.00 | 83.68 |
| DINOv3 | 87.87 | 85.68 |
| Franca | 84.50 | 81.11 |
| SigLIP2 | 87.33 | 84.75 |

**关键发现**：LingBot-Vision 的 image-level recognition 略弱于 DINOv3/SigLIP2，但 dense spatial perception 更强，体现了预训练目标的取舍。

### Table 5：蒸馏模型族

| 尺寸 | 模型 | IN1K Linear ↑ | NYU RMSE ↓ | KITTI RMSE ↓ | ADE mIoU |
|------|------|---------------|------------|--------------|----------|
| L | DINOv2 | 86.43 | 0.411 | 3.243 | 48.63 |
| L | DINOv3 | 87.31 | 0.351 | 2.643 | 55.00 |
| L | **LingBot-Vision** | **86.38** | **0.310** | **2.574** | **52.75** |
| B | DINOv3 | 84.79 | 0.371 | 2.826 | 51.74 |
| B | **LingBot-Vision** | **85.05** | **0.339** | **2.793** | **51.44** |
| S | DINOv3 | 80.37 | 0.405 | 2.851 | 46.51 |
| S | **LingBot-Vision** | **82.22** | **0.383** | 3.784 | **47.01** |

**关键发现**：dense advantage 能通过 distillation 传给 ViT-L/B/S。尤其 ViT-L 0.3B 在 NYUv2 上接近 7B DINOv3。

### Table 6：作为 MDM encoder 初始化的对比

| 初始化 | 架构 | DIODE-In block RMSE ↓ | NYU block RMSE ↓ | VOID sparse RMSE ↓ | ClearGrasp D435 RMSE ↓ |
|--------|------|-----------------------|------------------|--------------------|------------------------|
| DINOv2 | ViT-L/14 | 0.152 | 0.169 | 0.214 | 0.015 |
| DINOv3 | ViT-L/16 | 0.114 | 0.154 | 0.190 | 0.012 |
| **LingBot-Vision** | **ViT-L/16** | **0.094** | **0.145** | 0.199 | **0.010** |
| DINOv2 | ViT-g/14 | 0.118 | 0.178 | 0.203 | 0.018 |
| **LingBot-Vision** | **ViT-g/16** | **0.083** | **0.148** | **0.182** | 0.015 |

**关键发现**：在相同 MDM pipeline 下，LingBot-Vision 初始化在多数深度补全 benchmark 上优于 DINOv2/DINOv3，说明预训练获得的边界特征能迁移到 RGB-D 补全。

### Figure 8：数据规模对深度补全的影响

![[assets/LingBot-Vision/LingBot-Vision_fig8_page20.png]]

**说明**：从 3M 到 150M RGB-D 样本，LingBot-Vision 初始化的优势不会被数据规模“洗掉”，反而随着数据增长扩大；DINOv2 在 20M 后趋于饱和。

### Figure 9：镜面/玻璃场景深度补全

![[assets/LingBot-Vision/LingBot-Vision_fig9_page21.png]]

**说明**：LingBot-Depth 2.0 在镜面、玻璃等 active depth sensor 失效区域补全更平滑，边界更清楚。

### Figure 10：深度补全定性对比

![[assets/LingBot-Vision/LingBot-Vision_fig10_page22.png]]

**说明**：相比 CDMs、OMNI-DC、LingBot-Depth 1.0，2.0 在复杂缺失区域更少出现边界模糊和点云外点。

### Table 7：真实传感器 depth completion

| 方法 | Hammer D435 | ClearGrasp D415 | LingBot D415 | LingBot D455 |
|------|-------------|-----------------|--------------|--------------|
| CDMs | 0.032/0.931 | 0.017/0.893 | 0.516/0.831 | 0.623/0.755 |
| LingBot-Depth 1.0 | 0.036/0.861 | 0.022/0.787 | 0.373/0.934 | 0.434/0.877 |
| **LingBot-Depth 2.0** | 0.032/0.895 | **0.010/0.981** | 0.242/0.949 | 0.394/0.930 |
| **LingBot-Depth 2.0 ViT-g** | **0.031/0.916** | 0.016/0.897 | **0.228/0.957** | **0.375/0.940** |

**关键发现**：LingBot-Depth 2.0 在 8 个真实传感器配置中 6 个取得最佳，尤其 ClearGrasp 透明物体场景优势明显。

### Table 8：Block-mask / sparse depth completion

| 方法 | DIODE-In block | DIODE-Out block | NYU block | VOID sparse | NYU sparse | ETH3D sparse |
|------|----------------|-----------------|-----------|-------------|------------|--------------|
| CDMs | 0.461/0.762 | 4.594/0.594 | 0.280/0.782 | 0.990/0.181 | 0.728/0.281 | 4.173/0.061 |
| OMNI-DC | 0.183/0.910 | 3.679/0.492 | 0.151/0.861 | 0.206/0.937 | 0.118/0.949 | 0.600/0.872 |
| LingBot-Depth 1.0 | 0.132/0.981 | 3.404/0.809 | 0.117/0.953 | 0.199/0.897 | 0.143/0.937 | **0.368/0.929** |
| **LingBot-Depth 2.0** | 0.062/0.995 | 2.440/0.826 | 0.116/0.951 | **0.170/0.940** | **0.113/0.950** | 0.522/0.907 |
| **LingBot-Depth 2.0 ViT-g** | **0.060/0.996** | **2.298/0.841** | **0.114/0.954** | 0.174/0.940 | **0.113/0.951** | 0.476/0.924 |

**关键发现**：2.0 在 8 个 benchmark 中 7 个 RMSE 最好；相较 1.0，DIODE-In block 从 0.132 降到 0.062，几乎减半。

### Figure 11：corner anchors 与 a-contrario validation 消融

![[assets/LingBot-Vision/LingBot-Vision_fig11_page26.png]]

**说明**：corner anchors 和 a-contrario validation 是互补 safeguards。只要有其中之一，线段解码仍然较稳定；二者都去掉时结果明显退化。

---

## 实验

### 数据集

| 数据 | 规模/用途 | 说明 |
|------|-----------|------|
| 预训练图像语料 | 160.75M images | 从约 2B web images 中检索和去重得到，包含 ImageNet-21K/1K、GLDv2、Mapillary-SLS 以及多种 dense scene seed 数据 |
| ImageNet-1K | global probing | linear probe 与 k-NN |
| NYU-Depth v2 / KITTI | depth linear probing | 冻结特征 + 线性读出 |
| ADE20K / Cityscapes / VOC12 | semantic segmentation | mIoU |
| DAVIS-2017 / YouTube-VOS | video object segmentation | frozen feature label propagation |
| Depth completion benchmarks | LingBot-Depth 2.0 | DIODE、iBims-1、NYU、VOID、ETH3D、HAMMER、ClearGrasp、自采 LingBot |

### 实现细节

- **Backbone**：ViT-g/16，约 1.1B 参数；蒸馏学生包括 ViT-L、ViT-B、ViT-S。
- **边界 head**：3-layer per-token MLP，output stride $s=2$，head dim 512，$K=32$ bins/channel。
- **优化器**：AdamW，global batch size 3072。
- **训练阶段**：300k self-distillation pretraining + 100k Gram anchoring + 100k high-resolution adaptation。
- **crop 设置**：pretraining global/local crop 为 256/112 px；high-res adaptation 使用 512 px。
- **EMA**：teacher momentum 从 0.994 anneal 到 1.0。
- **temperature**：teacher temperature 前 30k iterations 从 0.04 ramp 到 0.07。
- **weight decay**：0.04 到 0.2 cosine ramp。
- **稳定技巧**：非边界 teacher target 填 uniform random label，teacher-side centering 始终开启。

### 关键实验结论

1. **边界几何目标是主要增益来源**：Table 1 中 categorical boundary target 几乎贡献全部 NYUv2 改善。
2. **dense 优势强于 global recognition 优势**：LingBot-Vision 的 ImageNet accuracy 略低于 DINOv3，但 NYUv2、VOC 和 VOS 表现更突出。
3. **蒸馏保持空间特性**：ViT-L/B/S 学生模型仍在 depth task 上大幅领先同级模型。
4. **下游深度补全受益明显**：LingBot-Depth 2.0 只替换 encoder 初始化并扩大数据，就在多数真实传感器和缺失模式 benchmark 上超越 1.0 与外部 baseline。

---

## 批判性思考

### 优点

1. **方法动机很清晰**：把边界作为空间感知的核心预训练信号，直接对准 dense prediction 的痛点，而不是事后补 dense feature。
2. **技术闭环完整**：边界发现、mask 选择、几何监督、分类化防坍塌、a-contrario 验证、GPU 化训练系统都围绕同一目标设计。
3. **结果覆盖面广**：不仅有 pretraining benchmark，也展示了 depth completion 系统级收益，说明表示学习不是孤立指标。
4. **不依赖人工边界标注**：边界目标在线自举，减少了把下游 detector 偏置灌入 foundation model 的风险。

### 局限

1. **工程复杂度高**：在线线段解码、a-contrario 验证、fused kernels、CUDA batched pipeline 都不是普通实验室容易复现的组件。
2. **边界先验可能偏向 piecewise-linear 世界**：虽然 curved boundary 可由短线段链近似，但自然纹理、软物体、透明/反射表面的真实结构未必都能被线段假设充分表达。
3. **语义分类仍弱于最强模型**：ImageNet linear/k-NN 落后 DINOv3 和 SigLIP2，说明边界中心目标会把部分容量从 image-level invariance 转向 localized structure。
4. **数据和算力门槛仍然很高**：1.1B 参数、160M 图像、三阶段训练和 150M RGB-D 下游数据并不轻量。
5. **与 DINOv3 的公平性仍需谨慎解读**：作者强调数据规模小于 DINOv3，但训练数据来源、过滤策略和实现细节不同，真正归因到 objective 仍需要第三方复现。

### 潜在改进方向

1. **更轻量的边界目标生成**：用近似 a-contrario 或 learned validation head 替代部分复杂 CUDA pipeline，降低复现门槛。
2. **从 2D 边界扩展到 3D discontinuity**：结合 RGB-D、multi-view 或 monocular depth pseudo-label，让预训练目标直接感知 surface normal、occlusion 和 depth jump。
3. **边界与语义的动态权重**：不同训练阶段可能需要不同 $\lambda_b$ / $\lambda_i$，早期更重几何、后期更重语义或反之，值得系统探索。
4. **用于机器人策略前端**：这类表示天然适合抓取、导航、避障和交互区域定位，可与 VLA 模型的视觉 encoder 初始化做系统对比。

### 可复现性评估

- [x] 代码仓库已给出
- [x] checkpoint 已给出
- [x] 主要公式与训练流程描述完整
- [ ] 大规模预训练数据不可完全复现
- [ ] CUDA target-generation pipeline 复现成本较高
- [ ] 150M RGB-D 下游数据不完全公开

---

## 关联笔记

### 基于

- [[DINO]]：image-level self-distillation 基线。
- [[iBOT]]：masked patch token distillation 基线。
- [[DINOv2]]：数据策展和自蒸馏训练范式参考。
- [[DINOv3]]：Gram anchoring、多阶段训练和强基线。
- [[Line Segment Detection]] / [[a-contrario]]：边界候选验证的统计基础。

### 对比

- [[V-JEPA 2.1]]：video/predictive representation baseline。
- [[SigLIP 2]]：language-aligned visual representation baseline。
- [[AM-RADIO]]：multi-teacher agglomeration baseline。
- [[PEspatial]]：spatial post-training baseline。

### 方法相关

- [[Self-Supervised Learning]]：整体训练范式。
- [[Masked Boundary Modeling]]：本文核心机制。
- [[Boundary Field]]：边界几何表示。
- [[Categorical Reparameterization]]：稳定边界自蒸馏。
- [[Depth Completion]]：LingBot-Depth 2.0 的主要下游任务。

---

## 速查卡片

> [!summary] Vision Pretraining for Dense Spatial Perception
> - **核心**：用边界作为自监督预训练的原生信号，让 ViT 学到空间结构敏感的 dense representation。
> - **方法**：teacher 在线生成 validated boundary field；boundary token 强制 mask；边界 token 同时学 iBOT 语义目标和 categorical boundary target。
> - **结果**：1B LingBot-Vision 在 NYUv2 RMSE 0.296，超过 7B DINOv3 的 0.309；VOS 接近 DINOv3；蒸馏学生保持 depth 优势。
> - **下游**：LingBot-Depth 2.0 只改 encoder 初始化和数据规模，就在多数深度补全 benchmark 上超过 1.0 和外部方法。
> - **代码**：https://github.com/robbyant/lingbot-vision

---

*笔记创建时间: 2026-07-09*
