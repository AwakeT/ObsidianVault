---
type: concept
aliases: [CUT3R, Continuous Updating Transformer for 3D Reconstruction]
---

# CUT3R

## 定义
DUSt3R 谱系的**持久状态（persistent state）循环 3D 感知模型**：用一组 state token 在线编码场景全局信息，随每帧读取/更新，输出统一坐标系下的度量尺度 pointmap，无需全局对齐优化，支持静态与动态场景，并能查询未观测视角。

## 核心要点
1. 持久 state token（768×768）作为隐式 [[Spatial Memory|空间记忆]]，通过 state-update / state-readout 双 decoder 与图像交互。
2. 输出 self / world 双 pointmap + 6-DoF 位姿，一次前向；隐式对齐替代 [[Global Optimization]]。
3. 用 [[Raymap]] 查询虚拟视角，"幻想"未观测结构。
4. 约 17 FPS（约 25× DUSt3R-GA），在线 SOTA/competitive。

## 代表工作
- [[CUT3R]]（本篇，CVPR 2025 Oral）：Continuous 3D Perception Model with Persistent State。
- [[TTT3R]]：在 CUT3R 状态更新上做 test-time training 改进，修复长序列遗忘。

## 相关概念
- [[3D Reconstruction]]
- [[DUSt3R]]
- [[Global Scene Representation]]
- [[Raymap]]
- [[Point Map]]
