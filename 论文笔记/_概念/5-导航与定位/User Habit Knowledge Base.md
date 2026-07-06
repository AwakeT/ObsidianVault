---
type: concept
aliases: [UHKB, 用户习惯知识库]
---

# User Habit Knowledge Base

## 定义
个性化物体导航中存储大量（N>300）自然语言用户习惯的知识库，每条习惯描述用户与物体交互的放置倾向（如"读完书留在沙发"），可含多用户的矛盾习惯。

## 核心要点
1. 每 episode 概率采样一个 UHKB，含目标及其他物体的习惯
2. 只有极少数习惯与当前目标相关，制造长上下文与多义/冲突
3. agent 需据探索证据消解空间冲突、优先化习惯

## 代表工作
- [[UcON]]: 提出该知识库并用 HRM 从中检索

## 相关概念
- [[Habit Retrieval Module]]
- [[Object Navigation]]
