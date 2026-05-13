---
type: concept
aliases: [CroCo, Cross-view Completion]
---

# CroCo

## 定义
基于跨视图补全的自监督预训练方法，训练 ViT 从一个视图预测另一个视图的 masked patches，学习强大的 3D 感知特征。

## 核心要点
1. 自监督预训练，不需要 3D 标注
2. 为 DUSt3R 和后续方法提供预训练 encoder
3. CroCo v2 进一步改进特征质量

## 代表工作
- [[DUSt3R]]: 使用 CroCo 预训练的 encoder
- [[MonST3R]]: 保留 CroCo encoder 的几何先验

## 相关概念
- [[DUSt3R]]
- [[Vision Transformer]]
