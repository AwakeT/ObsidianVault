# VLN 远程推理项目 · 部署运行文档

> 目标：让第一次接触本项目的同事，照着本文档从零把整套系统跑起来。
>
> 最后更新：2026-07-03 ｜ 状态标注：✅已验证　🔧推进中　⏳待完善

---

## 1. 项目简介

本项目是一套 **VLN（视觉语言导航）远程推理系统**：板端设备把观测图像发到云端大模型，云端推理后返回导航动作。

一句话数据流：**板端读图 → WebSocket 发到云端 → Qwen3-VL 模型推理 → 返回动作（如"左转 55 度"）→ 板端执行**。

典型性能：端到端约 **0.9 秒/步**（其中模型推理约 0.68 秒）。

---

## 2. 整体架构

### 2.1 拓扑图

```
┌──────────────┐   WebSocket    ┌──────────────┐   SSH + nc 隧道   ┌────────────────────┐
│  板端客户端   │ ─────────────→ │   SSH 桥接    │ ───────────────→ │    云端 VLN 服务     │
│ client_*.py  │  ws://.../ws/  │ ws_lan_bridge │  120.48.90.140   │   remote_server.py  │
│  读图/发step  │   navigate     │   (ssh+nc)    │     :2222         │   Qwen3-VL 推理      │
└──────────────┘ ←───────────── └──────────────┘ ←─────────────── └────────────────────┘
                    action                                            容器内 127.0.0.1:18000
```

> ⚠️ **为什么要 SSH 桥接**：云容器不能直接开放公网入站端口，服务只在容器内 `127.0.0.1:18000`；且云平台的 HTTP 转发**不支持 WebSocket**。所以走 SSH 隧道（`ssh + nc`）把容器内 18000 端口转发出来。

### 2.2 组件清单

| 组件 | 跑在哪 | 职责 | 主要代码 | 依赖 | 状态 |
|------|--------|------|----------|------|------|
| 云端服务 | 云容器 `120.48.90.140` | 加载模型、WebSocket 推理 | `remote_server.py` | 模型权重、GPU、transformers | ✅ |
| SSH 桥接 | 板端（或电脑） | SSH 隧道转发 18000 端口 | `ws_lan_bridge.py` | SSH 密钥、`nc` | ✅电脑 / 🔧板端 |
| 板端客户端 | 板端 Linux | 读图、发 step、收 action | `client_remote_linux.py` | websockets、Pillow、图片 | ✅笔记本 / ⏳板端 |

### 2.3 端到端数据流（跟一个请求走一遍）

```
1. 客户端读取 obs_1.jpg + map_1.jpg（观测图 + 地图图）
2. 编码为 base64，打包成 step 消息
3. 通过 WebSocket 发到 ws://127.0.0.1:18000/ws/navigate
4. 经 SSH 桥接隧道 → 云端容器 127.0.0.1:18000
5. 服务端用 Qwen3-VL 推理，生成 action
6. action 顺原路返回客户端
7. 客户端记录/执行，处理下一对图像
```

---

## 3. 前置条件

### 3.1 云端（服务器）
- GPU：【待补充：显存要求，如 ≥24GB】
- Python：【待补充：版本，如 3.10+】
- 模型权重：【待补充：模型从哪获取、放到什么路径】
- 代码仓库：【待补充：git 地址 / 获取方式】
- 依赖安装：【待补充：`pip install -r requirements.txt` 或具体依赖】
  - 已知关键依赖：`fastapi`、`uvicorn`、`transformers`（含 `vln_qwen3_vl` 模型）

### 3.2 板端
- 系统：Linux（如地平线 RDK / 旭日系列，ARM64）
- 依赖：`python3`、`ssh`、`nc`（远程容器需有 `nc`）
- Python 包：`websockets`、`Pillow`
- SSH 私钥：能连云服务器的 `id_ed25519`

### 3.3 网络
- 板端能访问云服务器 SSH：`120.48.90.140:2222`
- SSH 认证方式：authkey（`ssh-authkey-xxx@120.48.90.140`）

---

## 4. 部署步骤（核心）

> **总原则：自底向上，每步亮"绿灯"再进下一步。** 服务 → 桥接 → 客户端，顺序不能颠倒。

### 4.1 第一步：启动云端服务 ✅

在云服务器（容器内）执行：

```bash
VLN_API_TOKEN=DISABLE \
REMOTE_HOST=0.0.0.0 \
REMOTE_PORT=18000 \
VLN_MODEL_PATH=/你的/模型路径 \
VLN_DEVICE=cuda:0 \
python remote_server.py
```

🟢 **绿灯信号**：日志出现
```
VLN Qwen3-VL 模型加载成功！
Application startup complete.
Uvicorn running on http://0.0.0.0:18000
```

本机自测：
```bash
curl http://localhost:18000/
# 预期返回：{"detail":"Not Found"}   ← 根路径无路由，404 属正常，证明服务活着
```

> **环境变量说明**见 [附录 A](#附录-a配置项清单)。`VLN_API_TOKEN=DISABLE` 表示关闭鉴权（仅用于内网/调试），正式对外须改为强 token。

---

### 4.2 第二步：打通 SSH 桥接 ✅电脑 / 🔧板端

桥接脚本通过 `ssh + nc` 把云端容器的 18000 端口隧道到本地。

**板端版配置**（`ssh_bridge_board.py`，改动见注释）：
```python
LISTEN_HOST = "127.0.0.1"   # 板端本地监听
LISTEN_PORT = 18000
KEY = "/home/sunrise/.ssh/id_ed25519"          # 板端上的密钥路径
SSH_PORT = "2222"
SSH_TARGET = "ssh-authkey-xxx@120.48.90.140"   # ⚠️ 会话重启后需更新
REMOTE_HOST = "127.0.0.1"
REMOTE_PORT = 18000
```

启动：
```bash
# 密钥权限必须 600，否则 ssh 拒绝
chmod 600 /home/sunrise/.ssh/id_ed25519
python3 ssh_bridge_board.py
```

🟢 **绿灯信号**：
```
listening on ('127.0.0.1', 18000), forwarding via ssh ... -> 127.0.0.1:18000
```

验证隧道打通：
```bash
curl http://127.0.0.1:18000/
# 预期返回：{"detail":"Not Found"}   ← 请求穿过桥接到达了云端服务
```

> 桥接脚本零第三方依赖（纯 Python 标准库），板端有 `python3` 和 `ssh` 即可运行。

---

### 4.3 第三步：运行板端客户端 ⏳

```bash
# 安装依赖（ARM 平台若 Pillow 编译失败，改用系统包 apt install python3-pil）
python3 -m pip install --user websockets Pillow

# 准备本地图片：/home/sunrise/local_images/ 下放 obs_1.jpg / map_1.jpg ...

# 启动客户端
python3 client_remote_linux.py \
  --uri "ws://127.0.0.1:18000/ws/navigate?token=DISABLE" \
  --image-dir "/home/sunrise/local_images"
```

🟢 **绿灯信号**：客户端打印 `CONNECTED` → 发送 step → 收到 `action`。

---

## 5. 消息协议（WebSocket JSON）

连接地址：`ws://<桥接地址>:18000/ws/navigate?token=<token>`

### 客户端 → 服务端

| 类型 | 结构 | 说明 |
|------|------|------|
| `talk` | `{"type":"talk","message":"<对话JSON字符串>"}` | 纯对话测试 |
| `go_to_person` | `{"type":"go_to_person","user_prompt":"找到目标人物"}` | 启动导航会话 |
| `step` | `{"type":"step","last_action_result":null,"observation":{...}}` | 发送一步观测 |

`observation` 结构：
```json
{
  "pose": {"x":0,"y":0,"z":0,"qx":0,"qy":0,"qz":0,"qw":1},
  "rgb": {"front_camera": "<base64 JPEG>"},
  "front_camera_id": "front_camera",
  "face": {"camera_id":"", "bbox":[]},
  "map": {"image": "<base64 JPEG>", "real_position": null}
}
```

### 服务端 → 客户端

| 类型 | 说明 |
|------|------|
| `action` | 导航动作（如"左转 55 度"、"直行 0.6 米"） |
| `goto_point` | 目标点 |
| `result` | 结果 |
| `error` | 错误信息 |

---

## 6. 验证与测试

### 6.1 命令行冒烟测试
`test_recognition.py`：用一对本地图片跑通 `go_to_person → step → action`。
```bash
python3 test_recognition.py \
  --uri "ws://127.0.0.1:18000/ws/navigate?token=DISABLE" \
  --obs /path/obs_1.jpg --map /path/map_1.jpg
```

### 6.2 可视化测试台
`vln服务测试页.html`：浏览器打开，可连接 WebSocket、发 talk/go_to_person/step、看收发日志。用本地 HTTP 服务器托管：
```bash
python -m http.server 8000
# 浏览器打开 http://localhost:8000/vln服务测试页.html
```

---

## 7. 常见问题排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `curl localhost:18000` 返回 404 | 正常！根路径无路由 | 无需处理，这是服务正常的标志 |
| WebSocket 秒断 `code=1006` | 用了不支持 WS 的 HTTP 转发 | 必须走 SSH 桥接，不要用平台 HTTP 转发 |
| WebSocket 被拒 `4401` | token 不匹配 | 服务端 `VLN_API_TOKEN` 与客户端 `?token=` 要一致 |
| SSH `permissions too open` | 私钥权限过大 | `chmod 600 id_ed25519` |
| SSH `channel open failed` | authkey 不支持 `-L` 转发 | 用本项目的 `ssh+nc` 桥接脚本，别用 `ssh -L` |
| 桥接连不上 / SSH 报错 | `SSH_TARGET` 过期 | 会话重启后更新 `SSH_TARGET` 再重启桥接 |
| 板端 Pillow 装不上 | ARM 无预编译 wheel | `apt install python3-pil` |
| 客户端连不上 | 桥接没起或没通 | 先 `curl 127.0.0.1:18000` 确认桥接绿灯 |

---

## 附录 A：配置项清单

### 云端服务环境变量
| 变量 | 说明 | 默认 |
|------|------|------|
| `VLN_MODEL_PATH` | 模型权重目录 | 【待补充】 |
| `VLN_PROCESSOR_PATH` | processor 目录 | 等于 `VLN_MODEL_PATH` |
| `VLN_API_TOKEN` | 鉴权 token，`DISABLE` 关闭 | 必填 |
| `VLN_DEVICE` | 推理设备 | `cuda:0` |
| `REMOTE_HOST` | 监听地址 | `0.0.0.0` |
| `REMOTE_PORT` | 监听端口 | `8000`（本项目用 18000） |

### 桥接脚本配置
| 变量 | 说明 |
|------|------|
| `KEY` | SSH 私钥路径 |
| `SSH_PORT` | SSH 端口（`2222`） |
| `SSH_TARGET` | `ssh-authkey-xxx@120.48.90.140`，⚠️会变 |
| `REMOTE_HOST/PORT` | 容器内服务地址（`127.0.0.1:18000`） |

### 会话重启后需要更新的东西
- 桥接脚本里的 `SSH_TARGET`（authkey 会话重启后变化）

---

## 附录 B：性能基线

| 指标 | 平均值 |
|------|--------|
| 客户端打包发送 | ~45 ms/步 |
| 端到端等待 | ~867 ms/步 |
| 服务端推理 | ~680 ms/步 |
| 客户端到动作总耗时 | ~912 ms/步 |

（基于 10 对本地图像的离线测量）

---

## 附录 C：待补充清单（交付前请填全）

- [ ] 云端 GPU/显存要求、Python 版本
- [ ] 模型权重获取方式与存放路径
- [ ] 代码仓库地址与拉取方式（git）
- [ ] 云端依赖安装（requirements.txt）
- [ ] `src/`、`util/` 各模块职责说明
- [ ] `util/config_loader` 读取的配置文件位置与内容
- [ ] 板端最终形态：离线图片 or 实时摄像头
