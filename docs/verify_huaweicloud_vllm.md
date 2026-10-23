# 华为云 vLLM：SSH 隧道联调本地 agent_client

云端 `serve.sh` 默认只绑在 **远程本机** `127.0.0.1:8000`，公网进不去。SSH 端口是唯一通路。本地 Agent 经 **SSH 本地转发** 访问：本机 `8000` → 云端 `127.0.0.1:8000`。

本机 vLLM 与云端隧道共用同一端口，Agent 配置统一为：

```text
http://127.0.0.1:8000/v1
```

不要同时开本机 vLLM 和云端隧道（都会占 `:8000`）。连云端前先停掉本机占用 8000 的进程。

`configs/config.cloud.yaml` 的 `base_url` 已是该地址；`vlm.model` 须与云端 `SERVED_MODEL_NAME` 一致。

```
本机 vlm-agent / field-agent
  → http://127.0.0.1:8000/v1
  → SSH -L 8000:127.0.0.1:8000（唯一通路；SSH -p 以当前 ModelArts 作业为准）
  → 云端 127.0.0.1:8000  (serve.sh 的 VLLM_HOST / VLLM_PORT)
```

Agent 始终打本机 `:8000`，图片走隧道进云端，不需要给云端 8000 开公网。

## 前置

本机 SSH（`-p` 随作业变化，以下示例用 `32238`）：

```bash
ssh -i ~/.ssh/4d_assert.pem -p 32238 ma-user@dev-modelarts-cnnorth9.huaweicloud.com
```

云端启动 vLLM（已验证：`Qwen3-VL-30B-A3B-Instruct`，2 卡 TP）， 其他模型启动参考 [vLLM 华为云启动说明](docs/start_huaweicloud_vllm.md)：

```bash
# SFS (/mnt/dev) 上 os.replace 会失败；运行时缓存放到本地盘
export VLLM_CACHE_ROOT=/tmp/vllm-cache
mkdir -p "$VLLM_CACHE_ROOT"

MODEL_ID=/mnt/dev/vllm/model_cache/models/Qwen3-VL-30B-A3B-Instruct \
SERVED_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct \
MODEL_CACHE_DIR=/mnt/dev/vllm/model_cache \
HF_HUB_OFFLINE=1 \
TENSOR_PARALLEL_SIZE=2 \
bash ${MA_JOB_DIR}/code/vllm/serve.sh
```

说明：

| 变量 | 作用 |
|------|------|
| `MODEL_ID` | 加载权重的本地目录（SFS 上的模型文件） |
| `SERVED_MODEL_NAME` | API 对外名字；须与本地 yaml 的 `vlm.model` 一致 |
| `MODEL_CACHE_DIR` | vLLM `--download-dir`（离线权重根） |
| `VLLM_CACHE_ROOT` 等 | 写在 `$HOME`，避免 SFS 上 `PermissionError: os.replace`（modelinfos） |
| `TENSOR_PARALLEL_SIZE=2` | 与可见 GPU 数一致 |

`serve.sh` 默认 `VLLM_HOST=127.0.0.1`、`VLLM_PORT=8000`。30B MoE 启动较慢，等到日志里服务就绪后再 `curl`。

---

## 1. 确认云端 vLLM 已起来

开一个 **普通 SSH**（不要带 `-N`）：

```bash
ssh -i ~/.ssh/4d_assert.pem -o IdentitiesOnly=yes -p 32238 \
  ma-user@dev-modelarts-cnnorth9.huaweicloud.com
```

云端执行：

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

应能看到 `Qwen/Qwen3-VL-30B-A3B-Instruct`。若这里失败，隧道再通也没用——先把 `serve.sh` 跑到能 `GET /v1/models`。

---

## 2. 本机开隧道（单独留一个终端，别关）

先确认本机 `:8000` 空闲（不要同时跑本机 vLLM）：

```bash
ss -ltn | grep ':8000 ' && echo "stop local vLLM first" || echo "8000 free"
```

另开本机终端：

```bash
ssh -i ~/.ssh/4d_assert.pem \
  -o IdentitiesOnly=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -N -L 8000:127.0.0.1:8000 \
  -p 32238 ma-user@dev-modelarts-cnnorth9.huaweicloud.com
```

| 参数 | 作用 |
|------|------|
| `-N` | 只做转发，不开远程 shell |
| `-L 8000:127.0.0.1:8000` | 本机 `8000` → **云端** `127.0.0.1:8000` |
| `-p 32238` | SSH 唯一通路（换成当前作业端口） |
| `ExitOnForwardFailure=yes` | 本地 8000 被占就直接失败，避免「连上了但没转到」 |

成功时这个窗口会卡住、没有提示，这是正常的。后面步骤用别的终端。

想一条 SSH 既登录又转发（去掉 `-N`）：

```bash
ssh -i ~/.ssh/4d_assert.pem -o IdentitiesOnly=yes \
  -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
  -L 8000:127.0.0.1:8000 \
  -p 32238 ma-user@dev-modelarts-cnnorth9.huaweicloud.com
```

登录后云端与本机都是 `curl :8000`。更稳妥仍是：**一个窗口专职隧道，一个窗口跑 agent**。

---

## 3. 本机验证隧道

隧道那个窗口还开着时，在本机：

```bash
# 避免系统代理把 127.0.0.1 拐走
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

curl -fsS http://127.0.0.1:8000/v1/models
```

返回和第 1 步云端那次一样，隧道就算通了。

---

## 4. 跑本地 agent_client

仓库根目录：

```bash
cd /home/gn/Documents/Miao_space/Lang_vlm

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

uv run vlm-agent --config configs/config.cloud.yaml
```

`config.cloud.yaml` 与本机配置同一端口，只换模型名：

```yaml
vlm:
  base_url: "http://127.0.0.1:8000/v1"
  model: "Qwen/Qwen3-VL-30B-A3B-Instruct"
```

`SERVED_MODEL_NAME` 与配置里的 `model` 必须字符串一致。

带图 / 覆盖地址：

```bash
uv run vlm-agent --config configs/config.cloud.yaml \
  --image packages/agent_client/tests/fixtures/1.jpg \
  --text "描述场景"
```

等价写法（不改 yaml）：

```bash
uv run vlm-agent --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct
```

场端 Agent 同样打 `:8000`：

```bash
uv run field-agent --model Qwen/Qwen3-VL-30B-A3B-Instruct
```

---

## 常见坑

1. **本机 `:8000` 已被占用**  
   本机 vLLM / 旧隧道会挡住 `-L 8000:...`。先 `pkill -f vllm.entrypoints` 或关掉旧 SSH，再开隧道。

2. **隧道窗口关了 / SSH 断了**  
   ModelArts 空闲会踢。保持 `-o ServerAliveInterval=30`，断了就重开第 2 步，再 `curl :8000/v1/models`。SSH `-p` 以当前作业为准。

3. **本机 HTTP(S)_PROXY**  
   `httpx` 可能把 `127.0.0.1:8000` 交给代理。必须设 `NO_PROXY=127.0.0.1,localhost`。`agent_client` 只清了 `socks://`，不会自动加 `NO_PROXY`。

4. **`ensure_vlm_ready` 超时 5s**  
   隧道刚建好或 vLLM 还在加载时会报 `VLM not ready`。先 `curl` 通再跑 agent。30B 冷启动更慢。VL 推理本身 `config.cloud.yaml` 里 `timeout: 180`，图较大时正常会慢。

5. **不要把 `-L` 写成云端公网 IP**  
   `serve.sh` 默认 `VLLM_HOST=127.0.0.1`，必须是 `-L 8000:127.0.0.1:8000`。

6. **SFS 上 `Error saving model info cache` / `PermissionError: os.replace`**  
   缓存写到 `/mnt/dev/.../vllm/modelinfos` 时，NFS/SFS 常不允许原子 `rename`。一般非致命（后面仍会 `Resolved architecture`），但应设 `VLLM_CACHE_ROOT` / `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` 到 `$HOME/.cache/...`（见前置启动命令）。

7. **本地 `vlm.model` ≠ 云端 `SERVED_MODEL_NAME`**  
   `GET /v1/models` 仍可能成功，真正推理会 404：`The model \`...\` does not exist.`。对齐 served name，不要去对齐 `MODEL_ID` 路径。
