# VLN 桥接与板端客户端运行手册

最后验证时间：2026-06-23

## 1. 当前拓扑

Linux 板端通过 Windows 局域网桥接连接：

```text
Linux 板端客户端
  -> ws://192.168.31.9:18000/ws/navigate?token=DISABLE
  -> Windows 上的 ws_lan_bridge.py
  -> SSH shell 通道
  -> 远程容器 127.0.0.1:18000
  -> VLN 服务
```

使用端口 `18000`。

当前 Windows 桥接配置位于 `C:\Users\11440\Desktop\ws_lan_bridge.py`：

```python
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 18000
KEY = r"C:\Users\11440\.ssh\id_ed25519"
SSH_PORT = "2222"
SSH_TARGET = "ssh-authkey-0d2fffecec3ec57da74cc1b4@120.48.90.140"
REMOTE_HOST = "127.0.0.1"
REMOTE_PORT = 18000
```

如果远程 authkey 会话重启，先更新 `SSH_TARGET`，再重启桥接。

## 2. 启动或重启 Windows 桥接

在 Windows 机器的 PowerShell 中运行：

```powershell
$old = Get-CimInstance Win32_Process | Where-Object {
  $_.Name -eq 'python.exe' -and $_.CommandLine -like '*ws_lan_bridge.py*'
}
$old | ForEach-Object {
  Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

$python = 'C:\Users\11440\AppData\Local\Programs\Python\Python312\python.exe'
$script = 'C:\Users\11440\Desktop\ws_lan_bridge.py'
$p = Start-Process -FilePath $python -ArgumentList @('-u', $script) -WindowStyle Hidden -PassThru

Start-Sleep -Seconds 2
$p
netstat -ano | findstr :18000
```

预期监听结果：

```text
0.0.0.0:18000   LISTENING   <bridge_pid>
```
## 3. 验证桥接

在 Windows 上：

```powershell
curl.exe -i http://192.168.31.9:18000/
```

预期结果：

```text
HTTP/1.1 404 Not Found
server: uvicorn
{"detail":"Not Found"}
```

对 `/` 返回 `404` 属于正常现象；它证明局域网桥接已经到达远程 uvicorn 服务。

可选的 WebSocket 冒烟测试：

```python
import asyncio
import json
import websockets

async def main():
    url = "ws://192.168.31.9:18000/ws/navigate?token=DISABLE"
    async with websockets.connect(url, open_timeout=10) as ws:
        print("WS_CONNECTED")
        msg = {
            "type": "talk",
            "message": json.dumps([
                {"role": "user", "content": [{"type": "text", "text": "请回复OK"}]}
            ], ensure_ascii=False),
        }
        await ws.send(json.dumps(msg, ensure_ascii=False))
        print(await ws.recv())

asyncio.run(main())
```

## 4. 准备 Linux 板端客户端

使用不导入项目本地 `util` 模块的独立版客户端。

板端所需的 Python 包：

```bash
python3 -m pip install --user websockets Pillow
```

不要安装或依赖 `pip install util`；PyPI 上名为 `util` 的包与本客户端无关。

本地图像目录结构：

```text
/home/sunrise/local_images/
  obs_1.jpg
  obs_2.jpg
  ...
  map_1.jpg
  map_2.jpg
  ...
```

默认命名模式：

```text
obs_{i}.jpg
map_{i}.jpg
```

## 5. 在 Linux 板端启动客户端

运行：

```bash
python3 client_remote_linux.py \
  --uri "ws://192.168.31.9:18000/ws/navigate?token=DISABLE" \
  --image-dir "/home/sunrise/local_images"
```

如果在移除 `util` 依赖后仍使用原文件名：

```bash
python3 client_remote.py \
  --uri "ws://192.168.31.9:18000/ws/navigate?token=DISABLE" \
  --image-dir "/home/sunrise/local_images"
```

客户端以离线模式运行：

```text
1. 发送启动指令：{"type": "go_to_person", ...}
2. 读取 obs_i.jpg 和 map_i.jpg
3. 将两张图像打包为 base64 JPEG
4. 发送 step
5. 接收 action / goto_point
6. 伪造 last_action_result={"status": "SUCCESS"}
7. 继续处理下一对图像
```

## 6. 计时定义

最近一次计时运行使用了来自 `D:\local_images` 的 10 对本地图像。

计时字段：

```text
client_send_ms：
  从开始本地读取/打包/send_step 到 websocket.send(step) 返回为止。
  包含本地图像读取、JPEG/base64 编码、JSON 序列化以及 websocket 发送。

wait_to_action_ms：
  从 websocket.send(step) 返回到接收到服务端 action 为止。
  包含网络、服务端解析、服务端推理、响应生成以及网络返回。

server_infer_ms：
  服务端上报的 inference_time_ms 字段。

client_to_action_total_ms：
  client_send_ms + wait_to_action_ms。
  即从客户端开始准备 step 到收到 action 的总耗时。
```

## 7. 最新计时结果

汇总：

```text
client_send_avg_ms:          45.53
wait_to_action_avg_ms:      866.71
server_infer_avg_ms:        680.54
client_to_action_avg_ms:    912.24
```

逐步测量：

| step | client_send_ms | wait_to_action_ms | server_infer_ms | client_to_action_total_ms | action |
|---:|---:|---:|---:|---:|---|
| 1 | 70.84 | 856.53 | 598.07 | 927.37 | 左转55.0度 |
| 2 | 41.66 | 849.13 | 704.16 | 890.79 | 直行0.6米 |
| 3 | 41.90 | 843.32 | 700.97 | 885.22 | 直行0.6米 |
| 4 | 40.93 | 878.09 | 720.28 | 919.02 | 左转55.0度 |
| 5 | 40.38 | 845.75 | 707.28 | 886.13 | 右转55.0度 |
| 6 | 45.35 | 846.50 | 679.01 | 891.85 | 直行0.6米 |
| 7 | 43.91 | 980.72 | 661.83 | 1024.63 | 左转55.0度 |
| 8 | 43.22 | 860.55 | 697.27 | 903.77 | 右转55.0度 |
| 9 | 45.03 | 857.39 | 675.24 | 902.42 | 右转55.0度 |
| 10 | 42.10 | 849.14 | 661.31 | 891.23 | 右转55.0度 |

结果解读：

```text
首帧之后，客户端图像打包与发送约为 40-50 ms。
发送之后的端到端等待约为 0.85-0.98 s。
服务端上报的推理时间约为 0.60-0.72 s。
非推理开销平均约为 0.19 s。
```