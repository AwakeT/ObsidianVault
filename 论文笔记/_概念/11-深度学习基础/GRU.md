---
type: concept
aliases: [Gated Recurrent Unit, 门控循环单元, ConvGRU]
---

# GRU

## 定义
一种门控循环神经网络单元，通过更新门和重置门控制信息流，比 LSTM 更简洁。在立体匹配中常用 ConvGRU 变体进行迭代视差精化。

## 数学形式
$$z = \sigma(W_z [h_{t-1}, x_t])$$
$$r = \sigma(W_r [h_{t-1}, x_t])$$
$$\tilde{h} = \tanh(W [r \odot h_{t-1}, x_t])$$
$$h_t = (1-z) \odot h_{t-1} + z \odot \tilde{h}$$

## 核心要点
1. 更新门 $z$ 控制保留多少旧状态，重置门 $r$ 控制忽略多少旧状态
2. ConvGRU 将全连接替换为卷积，适用于空间特征图
3. 在 RAFT/RAFT-Stereo 中每次迭代从 correlation volume 查询特征并更新视差

## 代表工作
- [[RAFT-Stereo]]: Multi-Level GRU 跨分辨率传播
- [[CREStereo]]: Recurrent Update Module (RUM)
- [[FoundationStereo]]: 3 级 ConvGRU 迭代精化

## 相关概念
- [[Correlation Volume]]
- [[Transformer]]
