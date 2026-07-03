---
type: concept
aliases: [Temporal IoU, tIoU, 时间交并比]
---

# Temporal Intersection-over-Union

## 定义

Temporal Intersection-over-Union 是衡量两个时间段重叠程度的指标，常用于动作定位、视频事件检测和离线规划评估。

## 数学形式

$$
\operatorname{tIoU}(A,B)=
\frac{\operatorname{duration}(A\cap B)}
{\operatorname{duration}(A\cup B)}
$$

## 核心要点

1. 比逐帧准确率更关注一个动作段在时间上的覆盖质量。
2. 需要预测动作标签与 ground truth 标签一致时，重叠才有意义。
3. 适合评估机器人子任务规划中的持续动作片段。

## 代表工作

- [[Vesta]]: 用 temporal IoU 评估 zero-shot action planning benchmark。

## 相关概念

- [[Action Planning]]
- [[VideoMAE]]
