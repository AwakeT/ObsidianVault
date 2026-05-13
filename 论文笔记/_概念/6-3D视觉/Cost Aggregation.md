---
type: concept
aliases: [代价聚合]
---

# Cost Aggregation

## 定义
在立体匹配中对初始匹配代价进行空间聚合以平滑噪声、传播信息的步骤。

## 数学形式
$$S(p, d) = \sum_{\mathbf{r}} L_{\mathbf{r}}(p, d)$$

SGM 中通过多方向路径代价求和聚合。

## 核心要点
1. 局部方法：窗口内求和/加权平均（SAD, NCC）
2. 半全局方法：多方向扫描线 DP（SGM）
3. 全局方法：3D 卷积（PSMNet, GwcNet）、Transformer（FoundationStereo DT）
4. 深度学习方法中对应 cost volume filtering / hourglass network

## 代表工作
- [[SGBM]]: 多方向 1D DP 聚合
- [[FoundationStereo]]: AHCF（APC + DT）

## 相关概念
- [[Correlation Volume]]
- [[Stereo Matching]]
- [[Winner-Takes-All]]
