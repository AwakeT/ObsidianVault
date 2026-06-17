---
type: concept
aliases: [三维重建, 3D Scene Reconstruction]
---

# 3D Reconstruction

## 定义
3D Reconstruction 指从图像、视频或传感器观测中恢复场景或物体的三维几何结构。

## 核心要点
1. 传统方法依赖 SfM、MVS、SLAM 和 bundle adjustment。
2. 现代前馈模型可直接预测深度、点云和相机参数。
3. 动态视频进一步扩展为 4D Reconstruction。

## 代表工作
- [[DUSt3R]]: 端到端双视图/多视图 3D 重建代表。
- [[VGGT]]: 大规模前馈 3D 重建模型。
- [[D4RT]]: 将 3D 重建扩展为动态 4D 查询式重建。

## 相关概念
- [[Structure from Motion]]
- [[Point Cloud]]
- [[4D Reconstruction]]
