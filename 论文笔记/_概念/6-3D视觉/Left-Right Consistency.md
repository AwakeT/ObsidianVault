---
type: concept
aliases: [左右一致性检查, LR Check, 遮挡检测]
---

# Left-Right Consistency

## 定义
分别计算左→右和右→左视差图，检查互逆一致性以检测遮挡区域的后处理方法。

## 数学形式
$$|d_L(x, y) - d_R(x - d_L(x,y), y)| < \tau$$

不满足则标记为遮挡。

## 核心要点
1. 遮挡区域在一个方向有匹配但反向无匹配，一致性检查可有效检测
2. 阈值 $\tau$ 通常取 1 像素
3. 是 SGM 等经典方法的标准后处理步骤

## 代表工作
- [[SGBM]]: 标准后处理步骤

## 相关概念
- [[Stereo Matching]]
- [[Cost Aggregation]]
