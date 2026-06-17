---
type: concept
aliases: [Video Depth, 视频深度估计]
---

# Video Depth Estimation

## 定义
Video Depth Estimation 指为视频中每一帧估计像素级深度，并尽量保持跨帧尺度和时间一致性。

## 核心要点
1. 难点包括相机运动、动态物体、遮挡和尺度漂移。
2. 常用指标包括 AbsRel、scale-only alignment 和 scale-and-shift alignment。
3. D4RT 通过查询 $t_{src}=t_{tgt}=t_{cam}$ 的像素点，直接得到深度。

## 代表工作
- [[D4RT]]: 将视频深度作为统一 4D query interface 的一个特例。

## 相关概念
- [[Depth Anything]]
- [[Point Cloud]]
- [[4D Reconstruction]]
