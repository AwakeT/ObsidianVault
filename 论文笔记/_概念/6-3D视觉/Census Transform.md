---
type: concept
aliases: [Census 变换]
---

# Census Transform

## 定义
将像素邻域内每个像素与中心像素比较大小，编码为二进制串的非参数局部变换，对光照变化鲁棒。

## 核心要点
1. 输出为固定长度的二进制描述符，匹配代价用 Hamming 距离计算
2. 对单调灰度变换完全不变
3. 常作为 SGM 的匹配代价替代互信息（计算更快）

## 代表工作
- [[SGBM]]: 常用替代匹配代价

## 相关概念
- [[Birchfield-Tomasi]]
- [[Mutual Information]]
