---
type: concept
aliases: [KITTI 2015]
---

# KITTI-2015

## 定义
自动驾驶场景的立体匹配评测基准（2015 版），相比 2012 版增加了动态场景和更精确的评测。

## 核心要点
1. 200 对真实驾驶场景立体图像
2. 评测指标: D1-bg, D1-fg, D1-all (误差 > 3px 且 > 5% 的百分比)
3. 分别评测背景和前景像素

## 代表工作
- [[MobileStereoNet]]: 3D-MSNet D1-all = 2.10
- [[HITNet]]: D1-all = 1.98, 0.02s
- [[FoundationStereo]]: D1 = 2.8（零样本）

## 相关概念
- [[KITTI-2012]]
- [[Stereo Matching]]
