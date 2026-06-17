---
type: concept
aliases: [Query Dropping, Query 选择, Query 丢弃]
---

# Query Selection

## 定义
两阶段 [[DETR]] 中，从编码器输出选择高置信度 token 初始化解码器 object query 的机制；推理时也可据此丢弃低置信度 query 以降延迟。

## 核心要点
1. 按 encoder 输出处每个 token 的最大类别 logit 的 sigmoid 置信度排序
2. 推理时保留 Top-K query，无需重训即可减少检测数与延迟
3. [[RF-DETR]] 将 query 数作为 NAS 旋钮；丢 100 个最低置信度 query 几乎不降精度

## 数学形式
$$
\text{keep}(q_i) \iff \text{rank}(\max_c \sigma(z_{i,c})) \le K
$$

## 代表工作
- [[RF-DETR]]: query 丢弃作为可调旋钮
- [[RT-DETR]]: 不确定性最小 query 选择

## 相关概念
- [[DETR]]
- [[Sigmoid]]
