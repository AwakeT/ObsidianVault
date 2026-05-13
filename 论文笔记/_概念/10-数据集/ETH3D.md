---
type: concept
aliases: [ETH3D Stereo]
---

# ETH3D

## 定义
由 ETH Zurich 提供的灰度高分辨率立体匹配评测基准，涵盖室内外场景。

## 核心要点
1. 灰度图像，高分辨率（最高 6048x4032）
2. 评测指标: Bad X.0, AvgErr, RMSE
3. 47 对立体图像（训练 + 测试）
4. 对无纹理和过曝区域有挑战性

## 代表工作
- [[FoundationStereo]]: BP-0.5 = 1.26（微调后第一名）
- [[CREStereo]]: Bad 1.0 = 0.98
- [[RAFT-Stereo]]: Bad 1.0 = 2.44

## 相关概念
- [[Middlebury]]
- [[Stereo Matching]]
