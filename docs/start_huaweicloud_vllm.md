# DockerSpaces vLLM

镜像构建、模型下载、OBS/SWR 上传、SFS 挂载、路径 ABI 和排错步骤见：

- [vLLM 华为云部署说明](docs/01-vllm-cloud-deployment.md)

## 云端启动服务

以下命令均使用已经挂载到云端的离线模型缓存。

### Qwen2.5-VL-7B（临时历史缓存）

现有缓存：

```text
/mnt/dev/vlm/model_cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/
```

启动命令：

```bash
CACHE_DIR=/mnt/dev/vlm/model_cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct
REVISION="$(tr -d '\r\n' < "$CACHE_DIR/refs/main")"

MODEL_ID="$CACHE_DIR/snapshots/$REVISION" \
SERVED_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
MODEL_CACHE_DIR=/mnt/dev/vlm/model_cache/huggingface/hub \
HF_HUB_OFFLINE=1 \
TENSOR_PARALLEL_SIZE=1 \
bash "${MA_JOB_DIR}/code/vllm/serve.sh"
```

这是兼容 `/mnt/dev/vlm/...` 历史缓存的临时启动方式。后续会将该模型缓存
迁移到与 Gemma、Qwen3 相同的 `/mnt/dev/vllm/model_cache` 路径 ABI，届时
使用统一的启动方式，不再手动读取 `refs/main` 和拼接 snapshot 路径。

### Hugging Face cache：Gemma 4 26B-A4B

```bash
export HF_HUB_CACHE=/mnt/dev/vllm/model_cache
export VLLM_CACHE_ROOT=/tmp/vllm-cache
mkdir -p "$VLLM_CACHE_ROOT"

MODEL_ID=google/gemma-4-26B-A4B-it \
MODEL_CACHE_DIR=/mnt/dev/vllm/model_cache \
HF_HUB_OFFLINE=1 \
TENSOR_PARALLEL_SIZE=2 \
bash ${MA_JOB_DIR}/code/vllm/serve.sh
```

### 魔搭普通目录：Qwen3-VL-30B-A3B

模型目录：

```text
/mnt/dev/vllm/model_cache/models/Qwen3-VL-30B-A3B-Instruct
```

启动命令：

```bash
export VLLM_CACHE_ROOT=/tmp/vllm-cache
mkdir -p "$VLLM_CACHE_ROOT"

MODEL_ID=/mnt/dev/vllm/model_cache/models/Qwen3-VL-30B-A3B-Instruct \
SERVED_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct \
MODEL_CACHE_DIR=/mnt/dev/vllm/model_cache \
HF_HUB_OFFLINE=1 \
TENSOR_PARALLEL_SIZE=2 \
bash ${MA_JOB_DIR}/code/vllm/serve.sh
```

Gemma 的 Hugging Face cache 使用仓库 ID；魔搭下载的普通模型目录使用
容器内绝对路径。Qwen2.5 当前使用上面的临时 snapshot 绝对路径方案。
