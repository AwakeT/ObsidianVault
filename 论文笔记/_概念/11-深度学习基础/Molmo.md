---
type: concept
aliases: []
---

# Molmo

## 定义
Allen AI 提出的开源 VLM 系列，具备强 pointing（指点）能力，能根据自然语言查询预测图像中目标的点坐标，常用于 grounding 数据合成。

## 核心要点
1. **pointing 能力**: 由查询预测候选点，可结合 gt 框过滤或转框（接 SAM）。
2. **数据引擎用途**: LocateAnything 用 Molmo 预测点，保留落在 gt 框内的点作为可靠监督。

## 代表工作
- Molmo (Deitke et al., 2025): 开源 pointing VLM。
- [[LocateAnything]]: 在多目标 grounding 数据引擎中用 Molmo 生成点监督。

## 相关概念
- [[Visual Grounding]]
- [[SAM]]
- [[Rex-Omni]]
