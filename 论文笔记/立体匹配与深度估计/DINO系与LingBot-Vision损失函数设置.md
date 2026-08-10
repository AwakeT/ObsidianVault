# DINO 系 与 LingBot-Vision 的损失函数设置

本文对比 DINO / iBOT / DINOv2 / DINOv3 与 **LingBot-Vision** 的训练损失。

**说明**：DINO、iBOT、DINOv2 的公式为原论文标准形式；DINOv3 的 Gram anchoring，以及 LingBot-Vision 中 $\mathcal{L}_{bnd}$ / geometry routing 的符号化写法，是与架构图一致的**形式化重构**（论文只给出总损失结构 $\mathcal{L} = \mathcal{L}_{DINO} + \lambda_i \mathcal{L}_{iBOT} + \lambda_b \mathcal{L}_{bnd} + \lambda_k \mathcal{L}_{KoLeo}$），细节以原文为准。

---

## 0. 符号约定

| 符号 | 含义 |
|---|---|
| $\theta_s,\ \theta_t$ | student / teacher 参数（teacher 记 $\bar\theta$） |
| $g_s,\ g_t$ | student / teacher 的 projection head 输出（$K$ 维 logits） |
| $\tau_s,\ \tau_t$ | student / teacher softmax 温度（$\tau_t < \tau_s$，teacher 更"尖"） |
| $c$ | centering 向量（防塌缩） |
| $K$ | prototype（原型）数 / 每通道 bin 数 |
| $P,\ N$ | patch（token）数 |
| $V$ | 多视图集合（2 个 global crop + 若干 local crop） |
| $m_p\in\{0,1\}$ | patch $p$ 是否被 mask |
| $M,\ B,\ M^+$ | 随机 mask 集合 / 边界 token 集合 / $M^+ = M\cup B$ |

共用的两个基础操作（DINO 系全程复用，LingBot 也继承）：

Student 分布（标准 softmax）：

$$P_s(x)^{(i)} = \frac{\exp\big(g_s(x)^{(i)}/\tau_s\big)}{\sum_{k=1}^{K}\exp\big(g_s(x)^{(k)}/\tau_s\big)}$$

Teacher 分布（centering + sharpening，防止表征塌缩）：

$$P_t(x)^{(i)} = \frac{\exp\big((g_t(x)^{(i)} - c^{(i)})/\tau_t\big)}{\sum_{k=1}^{K}\exp\big((g_t(x)^{(k)} - c^{(k)})/\tau_t\big)}$$

Centering 的 EMA 更新（$B$ 为 batch 大小，$m$ 为动量）：

$$c \leftarrow m\,c + (1-m)\,\tfrac{1}{B}\sum_{i=1}^{B} g_t(x_i)$$

Teacher 参数的 EMA 更新（stop-gradient，不回传梯度）：

$$\bar\theta \leftarrow \lambda\,\bar\theta + (1-\lambda)\,\theta_s$$

---

## 1. DINO —— image-level 自蒸馏（`[CLS]` token）

只作用于 `[CLS]` token（全局表征）。teacher 只看 global crop，student 看所有 crop，要求同图不同视图的全局分布一致：

$$\mathcal{L}_{\text{DINO}} = \sum_{x\in\{x_1^g,\,x_2^g\}}\ \sum_{x'\in V,\ x'\neq x}\ -\,P_t(x)^{\top}\log P_s(x')$$

- $x_1^g, x_2^g$：两个 global crop（送入 teacher）；
- 内层对所有 $x'\neq x$ 求和：student 的每个视图都去对齐 teacher 的全局分布；
- 本质是**跨视图的交叉熵**，学"视图不变的全局语义"。

---

## 2. iBOT —— patch-level 掩码蒸馏（online tokenizer）

在 DINO 之上加入掩码图像建模（MIM）。student 看被 mask 的图 $\hat u$，teacher 看完整图 $u$，对**被 mask 的 patch** 逐 token 对齐 teacher 分布：

$$\mathcal{L}_{\text{iBOT}} = -\,\frac{1}{\sum_p m_p}\sum_{p=1}^{P} m_p\, P_t^{\text{patch}}(u_p)^{\top}\,\log P_s^{\text{patch}}(\hat u_p)$$

- $u_p$：teacher 在未遮图上第 $p$ 个 patch 的分布（即"在线 tokenizer"给的软目标）；
- $\hat u_p$：student 在遮挡图上同一位置的输出；
- $m_p=1$ 才计入 → 只在 mask 位置算 loss。

iBOT 总损失 = `[CLS]` 的 DINO 项 + masked patch 的 MIM 项。

---

## 3. KoLeo 正则（DINOv2 引入，防特征塌缩、鼓励均匀分布）

基于 Kozachenko–Leonenko 微分熵估计，鼓励 batch 内特征在单位球上均匀展开。设 $z_i$ 为 L2 归一化后的特征，$d_{n,i} = \min_{j\neq i} \lVert z_i - z_j\rVert$ 为样本 $i$ 到最近邻的距离：

$$\mathcal{L}_{\text{KoLeo}} = -\,\frac{1}{n}\sum_{i=1}^{n}\log\big(d_{n,i}\big)$$

- 最近邻越近 → 惩罚越大 → 逼特征彼此推开、铺满空间。

---

## 4. DINOv2 —— 总损失

$$\mathcal{L}_{\text{DINOv2}} = \mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + \lambda_{\text{koleo}}\,\mathcal{L}_{\text{KoLeo}}$$

外加工程细节：teacher 侧用 **Sinkhorn–Knopp** 归一化替代部分 centering、多次迭代做分布平衡（来自 SwAV），此处不展开。

---

## 5. DINOv3 —— 在 DINOv2 上增加 Gram anchoring

DINOv3 主要解决"长时训练下 **稠密 patch 特征退化**"的问题。做法是约束 student 的 **patch–patch 相似度结构（Gram 矩阵）** 去贴合一个 Gram teacher（早期 EMA 快照）。

设 $X_S\in\mathbb{R}^{P\times d}$ 为 L2 归一化后的 student patch 特征，Gram 矩阵 $G_S = X_S X_S^{\top}$：

$$\mathcal{L}_{\text{Gram}} = \big\lVert\, X_S X_S^{\top} - X_{S_G} X_{S_G}^{\top}\,\big\rVert_F^{2}$$

- $X_{S_G}$：Gram teacher 的 patch 特征；$\lVert\cdot\rVert_F$：Frobenius 范数；
- 只约束**相对相似度结构**（谁和谁像），不约束特征绝对值，从而"锚住"稠密特征不变糊。

近似总损失：

$$\mathcal{L}_{\text{DINOv3}} = \mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + \lambda_{\text{koleo}}\,\mathcal{L}_{\text{KoLeo}} + \lambda_{\text{gram}}\,\mathcal{L}_{\text{Gram}}$$

关键：DINOv3 维持稠密特征靠的是 **语义一致性正则（Gram），不含显式几何**。

---

## 6. LingBot-Vision —— 总损失

在 DINOv2 三项（DINO + iBOT + KoLeo）之上，**增加一路显式的边界几何损失** $\mathcal{L}_{\text{bnd}}$：

$$\mathcal{L} = \mathcal{L}_{\text{DINO}} + \lambda_i\,\mathcal{L}_{\text{iBOT}} + \lambda_b\,\mathcal{L}_{\text{bnd}} + \lambda_k\,\mathcal{L}_{\text{KoLeo}}$$

teacher 仍由 EMA 更新：$\bar\theta \leftarrow \lambda\bar\theta + (1-\lambda)\theta_s$。

### 6.1 iBOT 项作用在 $M^+$ 全体

掩码集合改为 boundary-forcing 的 $M^+ = M\cup B$，语义蒸馏覆盖所有被遮 token：

$$\mathcal{L}_{\text{iBOT}} = -\,\frac{1}{|M^+|}\sum_{p\in M^+} P_t^{\text{patch}}(u_p)^{\top}\,\log P_s^{\text{patch}}(\hat u_p)$$

### 6.2 边界几何项 —— 只作用在边界 token $B$

teacher 在线预测并净化出 **categorical boundary field**：把连续边界场按通道 $c\in\mathcal{C}=\{\text{distance},\text{orientation},\text{endpoints}\}$ 各离散为 $K$ 个 bin，得到软目标 $q_p^{(c)}$（非结构区域近似 uniform）。student 的 boundary head 输出 $\hat q_p^{(c)}$。损失为逐通道交叉熵，仅在 $p\in B$ 上求和：

$$\mathcal{L}_{\text{bnd}} = -\,\frac{1}{|B|}\sum_{p\in B}\ \sum_{c\in\mathcal{C}}\ \sum_{k=1}^{K} q_{p,k}^{(c)}\,\log \hat q_{p,k}^{(c)}$$

- 因为已"categorical 化"，$q_p^{(c)}$ 同样可套用第 0 节的 centering / sharpening 做稳定化；
- $|B|$ 通常远小于 $P$，故边界 head 只在少量 token 上展开，计算开销可控。

### 6.3 Geometry routing —— 每个 token 吃哪些 loss

$$\text{token } p \Rightarrow \begin{cases} \text{iBOT 语义}, & p \in M^+\setminus B \\ \text{iBOT 语义} + \mathcal{L}_{\text{bnd}}\ \text{几何}, & p \in B \end{cases}$$

即：**非边界 token 只受语义监督；边界 token 同时受语义 + 几何监督**，两类信号在 token 级分开路由，避免语义抽象与几何敏感性互相冲突。

---

## 7. 对比总览

| 损失项 | DINO | iBOT | DINOv2 | DINOv3 | LingBot-Vision |
|---|:---:|:---:|:---:|:---:|:---:|
| $\mathcal{L}_{\text{DINO}}$（CLS 语义） | ✅ | ✅ | ✅ | ✅ | ✅ |
| $\mathcal{L}_{\text{iBOT}}$（patch 语义） | — | ✅ | ✅ | ✅ | ✅（作用于 $M^+$） |
| $\mathcal{L}_{\text{KoLeo}}$（防塌缩） | — | — | ✅ | ✅ | ✅ |
| $\mathcal{L}_{\text{Gram}}$（稠密锚定） | — | — | — | ✅ | — |
| $\mathcal{L}_{\text{bnd}}$（**边界几何**） | — | — | — | — | ✅（只在 $B$） |
| 掩码策略 | 多视图 | 随机 | 随机 | 随机 | **boundary-forcing** |
| 几何信号 | 无 | 无 | 无 | 隐式（Gram） | **显式（边界监督）** |

一句话差异：

- DINO/iBOT/DINOv2：**只有语义损失**，边界是副产物。
- DINOv3：加 **Gram anchoring** 保稠密特征，但仍是**语义一致性正则，无显式几何**。
- LingBot-Vision：新增 **$\mathcal{L}_{\text{bnd}}$ 显式边界几何** + **boundary-forcing 掩码** + **geometry routing**，把边界几何做成预训练原生信号。

---

## 8. 各超参数含义

| 超参 | 出处 | 作用 |
|---|---|---|
| $\lambda_i$ | LingBot / iBOT | patch 语义蒸馏权重 |
| $\lambda_b$ | LingBot | **边界几何损失权重**（可分阶段动态调节，平衡几何/语义） |
| $\lambda_k,\ \lambda_{\text{koleo}}$ | DINOv2 / LingBot | KoLeo 正则权重 |
| $\lambda_{\text{gram}}$ | DINOv3 | Gram anchoring 权重 |
| $\lambda$ | 全系 | teacher EMA 动量 |
| $\tau_s,\ \tau_t$ | 全系 | student / teacher softmax 温度 |
| $K$ | 全系 | prototype 数 / 每通道 bin 数 |

---

相关笔记：`LingBot-Vision.drawio`（架构图）、`两种掩码自蒸馏流程对比.drawio`（流程对比）。
