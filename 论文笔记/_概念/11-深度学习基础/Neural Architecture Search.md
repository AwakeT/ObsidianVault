---
type: concept
aliases: [NAS, 神经架构搜索, 权重共享 NAS, Weight-Sharing NAS]
---

# Neural Architecture Search

## 定义
自动搜索具有不同精度-延迟权衡的模型架构族的方法。权重共享 NAS（weight-sharing NAS）让所有候选子网络共享一套权重并联合训练，从而解耦训练与搜索。

## 核心要点
1. 早期 NAS（NASNet、AmoebaNet）只追精度、计算昂贵；硬件感知 NAS 每换硬件需重搜
2. [[Once-for-All|OFA]] 提出权重共享，单次训练得到数千子网络
3. [[RF-DETR]] 首次将端到端权重共享 NAS 用于目标检测/分割，训练时随机采样配置做梯度更新（"架构增强"正则）

## 数学形式
$$
\theta^{*} = \arg\min_{\theta}\ \mathbb{E}_{c \sim \mathcal{U}(\mathcal{C})}[\mathcal{L}(f(x;\theta,c))]
$$

## 代表工作
- [[RF-DETR]]: 端到端权重共享 NAS 用于检测
- [[Once-for-All]]: 权重共享 NAS 奠基工作

## 相关概念
- [[Once-for-All]]
- [[FlexiViT]]
- [[Dropout]]
