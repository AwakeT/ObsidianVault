---
type: concept
aliases: [场景图, 3D Scene Graph, 3D 场景图]
---

# Scene Graph

## 定义
将场景表示为图结构的方法：节点为检测到的物体/房间，边为它们之间的几何或语义关系；在具身导航中常作为记忆/推理基底，有时结合开放词汇检索。

## 核心要点
1. **检测器中心**: 节点来自物体/房间检测，依赖检测质量
2. **优点**: 紧凑、可解释、便于符号推理与开放词汇检索
3. **局限**（导航语境）: (i) 将观测压缩为稀疏符号，丢弃纹理/空间布局等细粒度线索；(ii) 对检测噪声敏感，噪声沿下游推理累积；(iii) 阻止 VLM 直接对多视图图像推理
4. 3D 场景图进一步加入几何/空间层级（房间-物体-关系）

## 代表工作
- [[ConceptGraphs]]: 开放词汇 3D 场景图
- [[MSGNav]]: 多模态 3D 场景图零样本导航
- [[EvoMemNav]]: 指出场景图记忆的局限，转而用图像锚定的 [[Visual-Semantic Memory Graph]]

## 相关概念
- [[Visual-Semantic Memory Graph]]
- [[Topological Map]]
- [[Object Navigation]]
