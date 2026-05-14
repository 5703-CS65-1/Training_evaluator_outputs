#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
# 启动 vLLM OpenAI-compatible server（本地 long-running 进程）
# 完成后服务会监听在 http://localhost:$PORT/v1，可直接被评测脚本 / OpenAI SDK 调用。
# ══════════════════════════════════════════════════════════════════════════
#
# 前置条件：
#   1) 已经 bash install_vllm.sh 装好 vLLM
#   2) 已经合并 LoRA：python merge_lora.py --base ... --lora ... --out $MODEL_PATH
#
# 启动后：
#   • 健康检查：    curl http://localhost:8000/v1/models
#   • 后台日志：    tail -f $LOG_FILE
#   • 停止服务：    kill $(cat $PID_FILE)   或   pkill -f vllm.entrypoints.openai.api_server
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── env / 路径 ─────────────────────────────────────────────────────────────
VLLM_ENV="${VLLM_ENV:-/root/autodl-fs/conda_envs/vllm}"
PYTHON_BIN="$VLLM_ENV/bin/python"

MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/models/qwen3vl-8b-aesthetic-merged}"
SERVED_NAME="${SERVED_NAME:-qwen3vl-aesthetic}"

# ── server 配置 ───────────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# 推理参数
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"   # A800-80G 单卡：0.85 ≈ 68GB
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"  # Qwen3-VL 默认 128k，截到 16k 省显存
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"  # 单卡=1，多卡改成 GPU 数
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":1}}"  # 每条只配 1 张图（评测场景, vLLM ≥0.11 要 JSON）

# trust_remote_code 仅当 base 模型自定义代码时需要；Qwen3-VL 已在 transformers 主线
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

# ── 后台运行控制 ──────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs/vllm}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/vllm_server.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/vllm_server.pid}"
RUN_FOREGROUND="${RUN_FOREGROUND:-false}"   # true=前台调试；false=后台 nohup

mkdir -p "$LOG_DIR"

# ══════════════════════════════════════════════════════════════════════════
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] $PYTHON_BIN 不存在，先执行 bash install_vllm.sh" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] 找不到合并后的模型 $MODEL_PATH" >&2
    echo "        先跑 merge_lora.py 生成它，或者改 MODEL_PATH 环境变量" >&2
    exit 1
fi

# 已有同端口实例就不重复启动
if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[INFO] vLLM server 已在 :$PORT 运行；跳过启动。"
    curl -s "http://localhost:$PORT/v1/models" | head -c 400; echo
    exit 0
fi

# 清理之前异常退出留下的 EngineCore 子进程（这种进程会独占 GPU 显存）
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" || true)
if [[ -n "$GPU_PIDS" ]]; then
    echo "[WARN] Found stale GPU processes, killing: $GPU_PIDS"
    for p in $GPU_PIDS; do
        cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
        if [[ "$cmd" == *vllm* || "$cmd" == *EngineCore* || "$cmd" == *python* ]]; then
            echo "  kill -9 $p   ($cmd)"
            kill -9 "$p" 2>/dev/null || true
        fi
    done
    sleep 2
fi

CMD=(
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --model                "$MODEL_PATH"
    --served-model-name    "$SERVED_NAME"
    --host                 "$HOST"
    --port                 "$PORT"
    --dtype                "$DTYPE"
    --max-model-len        "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --limit-mm-per-prompt  "$LIMIT_MM_PER_PROMPT"
)
[[ "$TRUST_REMOTE_CODE" == "true" ]] && CMD+=(--trust-remote-code)

echo "════════════════════════════════════════════════════════"
echo " Starting vLLM OpenAI server"
echo "   model        : $MODEL_PATH"
echo "   served name  : $SERVED_NAME"
echo "   listen       : http://$HOST:$PORT/v1"
echo "   max_model_len: $MAX_MODEL_LEN   gpu_util: $GPU_MEM_UTIL   dtype: $DTYPE"
echo "   log file     : $LOG_FILE"
echo "════════════════════════════════════════════════════════"

if [[ "$RUN_FOREGROUND" == "true" ]]; then
    exec "${CMD[@]}"
else
    nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
    pid=$!
    echo "$pid" >"$PID_FILE"
    echo "[OK] vLLM server pid=$pid log=$LOG_FILE"
    echo "等待加载（首次约 60-180s，watch /v1/models 直到出现 served-model-name）："
    echo "  curl -s http://localhost:$PORT/v1/models"
fi
