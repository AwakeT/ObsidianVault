---
type: concept
aliases: [视觉里程计, VO, Visual Odometry]
---

# Visual Odometry

## 定义
仅从图像序列增量式估计相机自身运动（逐帧相对位姿并累积成轨迹）的技术，是视觉 SLAM 的前端。"无标定 VO（uncalibrated VO）"指连相机内参也一并估计。

## 核心要点
1. 输出相机轨迹，常用 **ATE（绝对轨迹误差）** 与 **RPE（相对位姿误差）** 评估。
2. 分稀疏（特征点，如 ORB-SLAM/DPVO）、稠密（如 DROID-VO）、稠密无约束（pointmap 回归类）。
3. 累积误差会导致漂移，回环检测与 BA 可缓解。
4. pointmap 谱系方法把 VO 转化为逐帧全局 pointmap / 位姿回归，无需相机标定。

## 代表工作
- [[MUSt3R]]：无标定 VO 达 SOTA（TUM ATE 平均 5.5cm）。
- [[CUT3R]] / [[TTT3R]]：在线位姿估计，长序列 VO。
- [[Spann3R]]：早期 pointmap 在线重建的 VO 尝试。

## 相关概念
- [[Camera Pose Estimation]]
- [[Bundle Adjustment]]
- [[Structure from Motion]]
