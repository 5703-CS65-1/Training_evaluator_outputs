#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
# vLLM 离线 batch 推理评测（不起 server，单进程跑完 Stage A/B/C/D）
#
#   Stage A: vllm.LLM.chat() 离线 batch 推理（GPU）
#   Stage B: 按 10 维 heading 切分 candidate → claims
#   Stage C: judge API 双向打分（precision + recall）
#   Stage D: 聚合 sample / corpus metrics
#
# 这一个脚本跑完会一次性拿到所有产物，无需再起其他脚本。
#
# 用法：直接 bash 启动；要改参数就改本文件下方的赋值
#   bash run_vllm_offline.sh
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Python / 模型 ─────────────────────────────────────────────────────────
PYTHON_BIN="/root/autodl-fs/conda_envs/vllm/bin/python"
MODEL_PATH="/root/autodl-tmp/models/Qwen3-VL-8B-Thinking"

# ── 数据 / 输出 ───────────────────────────────────────────────────────────
DATA_JSONL="data/eval_text/1.jsonl"
IMAGE_DIR="data/eval_images/package_1_images"
OUTPUT_DIR="outputs/base-all1"
LIMIT=""           # 留空字符串("") 跑全量

# ── vLLM engine ───────────────────────────────────────────────────────────
DTYPE="bfloat16"
MAX_MODEL_LEN="16384"
GPU_MEM_UTIL="0.85"
TENSOR_PARALLEL="1"

# ── 采样（Qwen3 thinking 官方推荐）──────────────────────────────────────
TEMPERATURE="0.6"
TOP_P="0.95"
TOP_K="20"
REP_PENALTY="1.05"
PRESENCE_PENALTY="0.0"
MAX_NEW_TOKENS="4096"
SEED="42"

# ── Judge（Stage C）─ 支持 qwen / deepseek 双后端 ───────────────────────
JUDGE_BACKEND="deepseek"   # qwen | deepseek
JUDGE_CONCURRENCY="30"
JUDGE_MAX_TOKENS="16384"
MAX_JSON_RETRIES="3"

# Qwen 预设（DashScope 国际站）
QWEN_JUDGE_MODEL="qwen3.6-plus"
QWEN_JUDGE_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_JUDGE_API_KEY="${DASHSCOPE_API_KEY:-sk-b9a5b0dfe90a4856b702bfa8ea6755cb}"
QWEN_JUDGE_WITH_IMAGE="false"

# DeepSeek 预设（OpenAI-compatible API；model 名透传，"deepseek-v4" 也可直接填）
DEEPSEEK_JUDGE_MODEL="deepseek-reasoner"
DEEPSEEK_JUDGE_BASE_URL="https://api.deepseek.com/v1"
DEEPSEEK_JUDGE_API_KEY="sk-6b7f0407358746d88f7059ea4a485222"
DEEPSEEK_JUDGE_WITH_IMAGE="false"   # deepseek-chat / -reasoner 纯文本，必须 false

case "$JUDGE_BACKEND" in
    qwen)
        JUDGE_MODEL="$QWEN_JUDGE_MODEL"
        JUDGE_BASE_URL="$QWEN_JUDGE_BASE_URL"
        JUDGE_API_KEY="$QWEN_JUDGE_API_KEY"
        JUDGE_WITH_IMAGE="$QWEN_JUDGE_WITH_IMAGE"
        ;;
    deepseek)
        JUDGE_MODEL="$DEEPSEEK_JUDGE_MODEL"
        JUDGE_BASE_URL="$DEEPSEEK_JUDGE_BASE_URL"
        JUDGE_API_KEY="$DEEPSEEK_JUDGE_API_KEY"
        JUDGE_WITH_IMAGE="$DEEPSEEK_JUDGE_WITH_IMAGE"
        ;;
    *)
        echo "[ERROR] Unknown JUDGE_BACKEND='$JUDGE_BACKEND' (qwen | deepseek)" >&2
        exit 1
        ;;
esac

# ── 流程控制 ─────────────────────────────────────────────────────────────
STAGE_A_ONLY="false"   # true → 只跑本地推理（Stage A），不调 judge API
RESUME="true"         # true → 跳过 OUTPUT_DIR 里已经跑过的 sample（A 和 B/C/D 都生效）

# ══════════════════════════════════════════════════════════════════════════
# 预检
# ══════════════════════════════════════════════════════════════════════════
[[ -x "$PYTHON_BIN" ]] || { echo "[ERROR] python 不存在: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "[ERROR] 模型目录不存在: $MODEL_PATH" >&2; exit 1; }
if [[ "$STAGE_A_ONLY" != "true" && -z "$JUDGE_API_KEY" ]]; then
    echo "[ERROR] JUDGE_API_KEY 为空（backend=$JUDGE_BACKEND，且未设 STAGE_A_ONLY=true）" >&2
    echo "  请 export DASHSCOPE_API_KEY=... 或 DEEPSEEK_API_KEY=...，或直接改本脚本预设" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"

CMD=(
    "$PYTHON_BIN" run_vllm_offline.py
    --model-path             "$MODEL_PATH"
    --data                   "$DATA_JSONL"
    --image-dir              "$IMAGE_DIR"
    --output                 "$OUTPUT_DIR"
    --dtype                  "$DTYPE"
    --max-model-len          "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --tensor-parallel-size   "$TENSOR_PARALLEL"
    --temperature            "$TEMPERATURE"
    --top-p                  "$TOP_P"
    --top-k                  "$TOP_K"
    --repetition-penalty     "$REP_PENALTY"
    --presence-penalty       "$PRESENCE_PENALTY"
    --max-new-tokens         "$MAX_NEW_TOKENS"
    --seed                   "$SEED"
    --judge-model            "$JUDGE_MODEL"
    --judge-api-key          "$JUDGE_API_KEY"
    --judge-base-url         "$JUDGE_BASE_URL"
    --judge-max-tokens       "$JUDGE_MAX_TOKENS"
    --judge-concurrency      "$JUDGE_CONCURRENCY"
    --max-json-retries       "$MAX_JSON_RETRIES"
)
[[ -n "$LIMIT" ]]                    && CMD+=(--limit "$LIMIT")
[[ "$JUDGE_WITH_IMAGE" != "true" ]]  && CMD+=(--no-judge-image)
[[ "$STAGE_A_ONLY"     == "true" ]]  && CMD+=(--stage-a-only)
[[ "$RESUME"           == "true" ]]  && CMD+=(--resume)

echo "════════════════════════════════════════════════════════"
echo " ArtcomBench eval — vLLM OFFLINE batch (Stage A/B/C/D)"
echo "  model        : $MODEL_PATH"
echo "  data         : $DATA_JSONL    limit: ${LIMIT:-all}"
echo "  output       : $OUTPUT_DIR"
echo "  sampling     : T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K rep=$REP_PENALTY"
echo "  judge        : [$JUDGE_BACKEND] $JUDGE_MODEL @ $JUDGE_BASE_URL  (conc=$JUDGE_CONCURRENCY, with_image=$JUDGE_WITH_IMAGE)"
echo "  flow         : stage_a_only=$STAGE_A_ONLY   resume=$RESUME"
echo "════════════════════════════════════════════════════════"

exec "${CMD[@]}"
