---
type: concept
aliases: [RexOmni]
---

# Rex-Omni

## 定义
一个基于 VLM 的统一检测与 grounding 专家模型（3B），采用量化坐标 token（point-based / quantized）表示进行 grounding 与检测，是 LocateAnything 的主要对比基线。

## 核心要点
1. **point-based prediction**: 用点预测缓解结构幻觉与高延迟。
2. **quantized 坐标**: 仍逐 token 自回归生成坐标，存在串行解码瓶颈（约 5.0 BPS）。
3. **数据引擎角色**: 也被 LocateAnything 用作数据引擎中由查询直接生成框的工具。

## 代表工作
- Rex-Omni (Jiang et al., 2025): 统一 VLM 检测/grounding。
- [[LocateAnything]]: 在吞吐（12.7 vs 5.0 BPS）与高 IoU 精度上超越 Rex-Omni。

## 相关概念
- [[Visual Grounding]]
- [[Object Detection]]
- [[Parallel Box Decoding]]
