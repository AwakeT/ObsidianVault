---
title: "User-Centric Object Navigation: A Benchmark with Integrated User Habits for Personalized Embodied Object Search"
method_name: "UcON"
authors: [Hongcheng Wang, Jinyu Zhu, Hao Dong]
year: 2026
venue: arXiv
tags: [object-navigation, embodied-ai, user-habits, benchmark, retrieval-augmented-generation, personalized-navigation]
zotero_collection: ""
image_source: mixed
arxiv_html: https://arxiv.org/html/2602.06459v1
created: 2026-07-06
---

# 论文笔记：User-Centric Object Navigation: A Benchmark with Integrated User Habits for Personalized Embodied Object Search

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University, PrimeBot |
| 日期 | February 2026 |
| 项目主页 | N/A |
| 对比基线 | [[PixelNav]], [[LGX]], [[L3MVN]], [[VTN]], [[ZSON]] |
| 链接 | [arXiv](https://arxiv.org/abs/2602.06459) / [Code](https://github.com/whcpumpkin/User-Centric-Object-Navigation) |

---

## 一句话总结

> 提出首个显式融入「用户习惯」的物体导航基准 UcON（489 类物体、约 22,600 条习惯），并用一个 [[RAG]] 式习惯检索模块（HRM）从习惯知识库中检索目标相关习惯，缓解 LLM 长上下文推理退化。

---

## 核心贡献

1. **UcON 基准**: 首个显式形式化并大规模评测「习惯条件化物体导航」的基准，含 489 类物体（现有基准最广）、约 22,600 条物体相关习惯，比 Habitat/MP3D/RoboTHOR/ProcTHOR/OVMM 都更贴近真实家庭服务场景
2. **习惯检索模块 HRM**: 受 [[Retrieval-Augmented Generation|RAG]] 启发，用 [[BGE-M3]] 计算查询模板与所有习惯的相似度取 top-$p$，选择性检索目标相关习惯注入 prompt，提升小尺寸 LLM 的习惯推理
3. **系统性实验发现**: 现有 SOTA 在习惯驱动放置下成功率大幅退化；引入用户习惯一致提升成功率；HRM 检索甚至常优于「GT 习惯」设置（反直觉发现）

---

## 问题背景

### 要解决的问题
现有物体导航（[[Object Navigation]]）基准把物体按通用场景先验（scene prior，如床在卧室、沙发靠电视）放置，忽略了个体用户的**放置习惯**。这限制了导航 agent 在个性化家庭环境中的适应性。

### 现有方法的局限
- 主流 ON 基准和方法都假设物体分布遵循常识规则，但真实家庭并非总如此——例如朋友有在沙发上读书的习惯，读完把书留在沙发而非书房
- 依赖场景先验的方法在习惯驱动放置下失效
- 端到端方法（[[VTN]]/[[ZSON]]）架构上无法处理长篇自然语言习惯
- 基础模型方法（[[LGX]]/[[PixelNav]]/[[L3MVN]]）推理基于常识，与非常规的习惯放置根本失配
- LLM 推理性能随 prompt 变长（数百条习惯）显著退化（lost-in-the-middle）

### 本文的动机
物体放置不仅受场景先验影响，更受用户习惯影响。家庭服务 agent 需识别并利用住户习惯来高效找物。作者主张**用户习惯是可学习信号**，能显著提升导航效率，并有意聚焦于**习惯利用**（habit utilization）而非习惯获取（habit acquisition）。

---

## 方法详解

### 模型架构

<!-- 使用 [[概念]] 内联链接所有技术术语 -->

UcON 是「基准 + 方法流程」，整体分为**习惯塑形场景生成**与**导航执行流程**：
- **输入**: 目标物体类别 $o \in O$ + 用户习惯知识库（[[User Habit Knowledge Base|UHKB]]）$H=\{h_i\}_{i=1}^N$（$N>300$ 条自然语言习惯）
- **动作空间**: MoveAhead / RotateRight / RotateLeft / LookUp / LookDown / Open / Done（每 episode ≤300 步）
- **核心模块**: [[Habit Retrieval Module|HRM]] 检索目标相关习惯，[[Object Navigation|物体检测器]] 识别当前观测物体，两者格式化后喂给基础大模型生成 plan
- **输出**: plan → 低层规划器 → 低层动作
- **成功判定**: agent 在目标物体可见且距离 <1m 时调用 Done

### 核心模块

#### 模块1: 任务生成（Sec III-B）

**设计动机**: 用 GPT-4 大规模、低成本生成物理合理且语义一致的习惯-放置对。

**具体实现**（三步流水线）:
- 步骤1: 给 GPT-4 所有物体类别，生成自然语言习惯 + 关联位置，位置表达为 4 种相对空间关系：`place_nextto` / `place_onTop` / `place_inside` / `place_under`
- 步骤2: 用仿真内置 API 验证放置的物理合理性（如小花瓶上放不下大箱子则 `place_onTop` 无效）
- 步骤3: 再用 GPT-4 验证习惯文本与验证过的放置之间的语义一致性

#### 模块2: 习惯塑形场景（Habit-shaped Scene，Sec III-D）

**设计动机**: 把用户习惯真实地"物化"到仿真场景中。

**具体实现**:
- 用 [[OmniGibson]] 仿真器（22 个初始场景），采样 $k$ 个物体并按关联习惯重新放置
- 顺序放置以尊重位置依赖
- 若一个习惯指定多个可能放置，随机选一个实例化，但**所有关联习惯都加入 UHKB**（制造多义/冲突）
- 初始场景没有的物体从 OmniGibson 数据集导入

#### 模块3: 习惯检索模块 HRM（Sec IV）

**设计动机**: LLM 推理性能随 prompt 变长退化，需选择性提供相关习惯。

**具体实现**:
- 用 [[BGE-M3]] 模型计算查询模板「Please retrieve the habit about {TargetObject}.」与所有存储习惯的相似度
- 取 top-$p$ 最相关结果注入 prompt
- 机制简单，但核心贡献是证明「用检索到的相关习惯增强 prompt」能提升基于 LLM 的导航

---

## 关键公式

### 公式1: [[SPL|路径长度加权成功率]]

$$
\text{SPL} = \frac{1}{N} \sum_{i=1}^{N} SR_i \frac{l_i}{\max(p_i, l_i)}
$$

**含义**: 衡量 agent 找物的效率，综合成功与否和路径长度。越接近最短路径且成功，SPL 越高。

**符号说明**:
- $SR_i$: 第 $i$ 个 episode 成功的二值指示
- $l_i$: 第 $i$ 个 episode 的最短路径长度
- $p_i$: agent 实际走的路径长度
- $N$: episode 总数

---

## 关键图表

<!-- 图片外链优先：arXiv HTML -->

### Figure 1: Habit-shaped Scene & Reasoning / 习惯塑形场景与推理示例

![[UcON_fig1.png|600]]

**说明**: 下半部分展示 5 个按用户习惯放置的物体示例（枕头在沙发、香蕉在冰箱、书在窗边扶手椅、报纸在餐桌、碗在水槽旁）。上半部分对比 LLM「带习惯推理」vs「不带习惯推理」——查询"找报纸"时，带习惯（"边吃早餐边读报"）推出报纸在餐桌（✓），不带习惯只能靠常识推living room（✗）。

### Figure 2: Task & Method Process / 任务与方法流程总览

![Figure 2](https://arxiv.org/html/2602.06459v1/x2.png)

**说明**: 起始采样 $k$ 个物体及其习惯组成 [[User Habit Knowledge Base|UHKB]]（每物体可多习惯）和所有可能位置（GPT-4 生成）；每物体随机采样一个位置形成习惯塑形场景。[[Habit Retrieval Module|HRM]] 检索相关习惯，物体检测器识别当前观测物体，两者经 prompter 格式化后喂给基础大模型生成 plan → 低层规划器 → 低层动作。

### Figure 3: Case Study (Large Env) / 大场景案例研究

![Figure 3](https://arxiv.org/html/2602.06459v1/x3.png)

**说明**: 大场景找书实验。书初始不可见，agent 被告知"睡前在床上读书"习惯。（A）Retrieval 级：据习惯推断书在卧室，朝出现床的方向走，最终在床脚找到书。（B）None 级：无习惯只能靠常识推 living room，绕客厅几圈后失败。

### Figure 4: Case Study (Small Env) / 小场景案例研究

![[UcON_fig4.png|600]]

**说明**: 小场景找报纸（被笔记本刻意遮挡）。agent 被告知"在 console table 上边吃早餐边读报"习惯。（A）Retrieval：先定位早餐所在房间，再据视野中的 console table 判断报纸在附近，找到目标。（B）None：只能靠常识在客厅盲搜，最终失败。

### Table I: 基准对比

| Benchmark | User Habits | Object Category |
|-----------|:-----------:|:---------------:|
| Habitat ObjectNav | ✗ | 6 |
| MP3D ObjectNav | ✗ | 21 |
| RoboTHOR | ✗ | 43 |
| ProcTHOR | ✗ | 108 |
| OVMM | ✗ | 150 |
| **UcON (Ours)** | ✓ | **489** |

**说明**: UcON 是唯一显式含用户习惯、且物体类别数最多（489）的基准。

### Table II: 动作空间

| Action | 描述 |
|--------|------|
| MoveAhead | 前进 0.25m |
| RotateRight | 右转 90° |
| RotateLeft | 左转 90° |
| LookUp | 相机上仰 30° |
| LookDown | 相机下俯 30° |
| Open | 打开视野内且 <1m 的闭合容器 |
| Done | 结束 episode |

### Table III: 各方法 × 习惯级别性能对比（SR/SPL）

| Method | Language Model | Object Detector | None | Full | GT* | Retrieval |
|--------|----------------|-----------------|------|------|-----|-----------|
| LGX | LLaMA-3.1-8B-Ins | GT | 20.2/14.8 | 25.2/18.4 | 20.4/14.6 | **28.2/21.3** |
| LGX | LLaMA-3.1-8B-Ins | YOLO-Worldv2-X | 16.2/10.4 | 19.0/12.0 | 18.1/12.2 | 19.5/13.2 |
| LGX | GPT-4 | GT | 25.1/18.8 | 27.8/22.1 | 28.2/21.8 | **30.6/23.1** |
| PixelNav | LLaMA-3.2-11B-Vision | GT | 15.6/13.0 | 17.3/14.7 | 18.3/16.8 | 17.4/14.6 |
| PixelNav | LLaMA-3.1-8B-Ins | GT | 16.9/14.9 | 17.1/14.3 | 17.2/15.1 | **20.5/17.1** |
| L3MVN | RoBERTa-Large | GT | 10.6/9.8 | / | 10.0/9.0 | 10.3/9.3 |
| VTN | / | / | 2.1/1.4 | / | / | / |
| ZSON | / | / | 1.0/0.7 | / | / | / |

**说明**: None=不给习惯，Full=完整习惯库，GT*=直接给目标物体关联习惯（理论上不可得），Retrieval=HRM 检索。端到端方法（VTN 2.1/ZSON 1.0）几近失效。引入习惯普遍提升 SR，HRM 的 Retrieval 常优于 GT。L3MVN（encoder-only RoBERTa）无法从习惯获益、也扛不住 Full 长上下文。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| UcON（[[OmniGibson]]） | 489 类物体, ~22,600 习惯, 22 场景 | 习惯塑形场景, 每 episode 采 20 物体+20 习惯 | 仿真评测 |
| 真实世界 | 3 环境（小30m² / 大90m²） | 手工设计 UHKB | 真机验证 |

### 实现细节

- **仿真**: [[OmniGibson]]，每 episode 随机选 20 物体、导入其习惯、选 20 条习惯改物体位置
- **消费级硬件约束**: 目标在 RTX 4090/3090 上本地运行（隐私考量，避免外部服务），用 7B/13B LLM/VLM
- **真机**: Unitree GO2 Edu + Astra Pro Plus RGB-D 相机（架在 4×0.5m 杆上），旋转角设 60° 匹配 FoV；GO2→ThinkPad E14→WiFi 卸载到 RTX 3090 工作站；YOLO-Worldv2-X 检测 + BFS 深度低层规划；PixelNav 作唯一基线
- **人类测试**: 26 名参与者，98.5% 习惯判为可行、96.7% 放置判为与习惯一致

### 可视化结果
- Q3 反直觉发现：Retrieval 常优于 GT，因为检索到的**间接相关习惯**（如涉及"breakfast table""countertop"的习惯）提供关键上下文地标——若目标 A 与地标 B 空间相关，检索 B 的习惯能让 agent 先导航向 B 再推断 A（见 Box V-E 的 tea bag 案例）
- Q4：即便给 GT 物体检测器 + 检索习惯，成功率仍不高，暴露当前 LLM 缺乏：(1) 多步规划、(2) 持久探索记忆、(3) 基于证据的信念更新/剪枝

---

## 批判性思考

### 优点
1. 问题立意新颖真实：把"个性化用户习惯"引入物体导航，填补现有基准全靠 scene prior 的空白
2. 规模与真实性领先：489 类物体、22,600 习惯，且有 26 人的人类可行性验证
3. Retrieval>GT 的反直觉发现有洞察力，揭示了"间接相关地标习惯"的价值
4. 明确聚焦习惯利用而非获取，把可复现的下游推理问题独立出来评测

### 局限性
1. 习惯为 GPT-4 合成，缺乏真实长尾多样性与时间漂移（作者自述）
2. 只做习惯利用，习惯获取/持续更新留给未来
3. HRM 仅为概念验证，未显式建模时间/地点等丰富上下文或部分可观测下的不确定性
4. 绝对成功率偏低（最高 30.6%），暴露任务本身对当前 LLM 极具挑战

### 潜在改进方向
1. 上下文条件化推理 + 多步规划 + 基于证据的信念更新/主动搜索
2. 从长期交互中持续获取和更新习惯
3. 更精细的 scene-aware 检索方法

### 可复现性评估
- [x] 代码开源（github.com/whcpumpkin/User-Centric-Object-Navigation）
- [ ] 预训练模型
- [x] 训练细节完整（超参、硬件、流程）
- [x] 数据集可获取（基于 OmniGibson + GPT-4 生成流水线）

---

## 关联笔记

### 基于
- [[Object Navigation]]: 任务基础
- [[Retrieval-Augmented Generation]]: HRM 的思想来源
- [[OmniGibson]]: 仿真平台

### 对比
- [[PixelNav]]: 像素引导零样本导航，主要基线（真机唯一对比）
- [[LGX]]: LLM 常识推理方向选择
- [[L3MVN]]: RoBERTa 预测物体出现概率
- [[VTN]]: 闭词表端到端 Transformer 导航
- [[ZSON]]: 开词表 CLIP 目标嵌入端到端导航

### 方法相关
- [[Habit Retrieval Module]]: 核心检索模块
- [[User Habit Knowledge Base]]: 习惯知识库
- [[BGE-M3]]: 习惯相似度检索模型
- [[YOLO-World]]: 物体检测器

### 硬件/数据相关
- [[Success Rate]] / [[SPL]]: 评测指标

---

## 速查卡片

> [!summary] UcON
> - **核心**: 首个融入「用户习惯」的物体导航基准（489 类物体 / ~22,600 习惯），打破全靠场景常识的假设
> - **方法**: GPT-4 生成习惯塑形场景 + BGE-M3 习惯检索模块 HRM，从习惯库检索目标相关习惯注入 prompt
> - **结果**: SOTA 在习惯放置下大幅退化；习惯提升 SR，HRM Retrieval 常优于 GT；最高 SR 仅 30.6% 暴露任务难度
> - **代码**: github.com/whcpumpkin/User-Centric-Object-Navigation

---

*笔记创建时间: 2026-07-06*
