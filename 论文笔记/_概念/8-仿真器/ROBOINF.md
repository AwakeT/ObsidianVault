---
type: concept
aliases: [ROBOINF, RoboInf]
---

# ROBOINF

## 定义
XLANG Lab 的可扩展机器人操作数据生成引擎/管线：构建操作场景、生成场景条件任务、合成可执行成功检查、经仿真反馈产出机器人运动程序，并在域随机化下 rollout 成功轨迹。

## 核心要点
1. Qwen-VLA 用其内部早期版本生成 vision-language-action 合成监督。
2. 本次 release 用 random-placement 设置：20 桌面场景 × 10 物体初始位姿 = 200 基础配置；450 操作任务；每任务 300 条带增强的成功轨迹。
3. 域随机化覆盖光照、相机位姿、背景、桌面纹理、机器人初态、物体位姿、控制器动力学（约 3K 背景 + 1K 桌面纹理）。
4. 轨迹按运动规划阶段分割为子任务段，提供多时间粒度监督；共 359,848 条全成功轨迹（含子任务段）。

## 代表工作
- XLANG Lab 2026 (blog): 原始提出
- [[Qwen-VLA]]: 合成数据来源

## 相关概念
- [[Vision-Language-Action]]
