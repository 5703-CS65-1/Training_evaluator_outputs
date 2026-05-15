#!/usr/bin/env bash
# ArtiMuse offline eval pipeline (Stage A: transformers; B/C/D: judge API)
#
# Stage A: load ArtiMuse (InternVL-3 fork), per-image run bench 10-dim chat
#          + ArtiMuse score head, assemble candidate_text in bench format
# Stage B: split claims by 10 dim headings
# Stage C: judge API bidirectional scoring (supports Qwen / DeepSeek presets
#          via JUDGE_BACKEND)
# Stage D: aggregate sample / corpus metrics
#
# Usage: bash run_artimuse_offline.sh

set -euo pipefail

# Python / model
PYTHON_BIN="/root/envs/artimuse/bin/python"
MODEL_PATH="/root/ArtiMuse/checkpoints/ArtiMuse"
ARTIMUSE_REPO="/root/ArtiMuse"

# Data / output
DATA_JSONL="data/eval_text/1.jsonl"
IMAGE_DIR="data/eval_images/package_1_images"
OUTPUT_DIR="outputs/artimuse-all1"
LIMIT=""

# ArtiMuse inference
DEVICE="cuda:0"
DTYPE="bfloat16"
MAX_NEW_TOKENS="4096"
CHAT_MODE="batch"           # batch (fast) | single (better, slower)
USE_FLASH_ATTN="true"
IMAGE_BATCH_SIZE="4"        # images per batch_chat call (×10 dims); A800 80GB: 4 safe, try 6

# ----- Judge backend: qwen | deepseek -----
JUDGE_BACKEND="deepseek"
JUDGE_CONCURRENCY="30"
JUDGE_MAX_TOKENS="16384"
MAX_JSON_RETRIES="2"

# Qwen preset (DashScope intl)
QWEN_JUDGE_MODEL="qwen3.6-plus"
QWEN_JUDGE_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_JUDGE_API_KEY=""
QWEN_JUDGE_WITH_IMAGE="false"

# DeepSeek preset (OpenAI-compatible). model name is pass-through; set to
# "deepseek-v4" / "deepseek-chat" / "deepseek-reasoner" as desired.
DEEPSEEK_JUDGE_MODEL="deepseek-reasoner"
DEEPSEEK_JUDGE_BASE_URL="https://api.deepseek.com/v1"
DEEPSEEK_JUDGE_API_KEY=""
DEEPSEEK_JUDGE_WITH_IMAGE="false"   # deepseek chat/reasoner are text-only

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

# Flow control
STAGE_A_ONLY="false"
RESUME="true"

# Preflight
[[ -x "$PYTHON_BIN" ]]        || { echo "[ERROR] python missing: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]]        || { echo "[ERROR] model dir missing: $MODEL_PATH" >&2; exit 1; }
[[ -d "$ARTIMUSE_REPO/src" ]] || { echo "[ERROR] ArtiMuse repo invalid: $ARTIMUSE_REPO (no src/)" >&2; exit 1; }
if [[ "$STAGE_A_ONLY" != "true" && -z "$JUDGE_API_KEY" ]]; then
    echo "[ERROR] JUDGE_API_KEY empty (backend=$JUDGE_BACKEND)" >&2
    echo "  please export DEEPSEEK_API_KEY=... or DASHSCOPE_API_KEY=..., or edit this script" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"

CMD=(
    "$PYTHON_BIN" run_artimuse_offline.py
    --model-path         "$MODEL_PATH"
    --artimuse-repo      "$ARTIMUSE_REPO"
    --data               "$DATA_JSONL"
    --image-dir          "$IMAGE_DIR"
    --output             "$OUTPUT_DIR"
    --device             "$DEVICE"
    --dtype              "$DTYPE"
    --max-new-tokens     "$MAX_NEW_TOKENS"
    --chat-mode          "$CHAT_MODE"
    --judge-model        "$JUDGE_MODEL"
    --judge-api-key      "$JUDGE_API_KEY"
    --judge-base-url     "$JUDGE_BASE_URL"
    --judge-max-tokens   "$JUDGE_MAX_TOKENS"
    --judge-concurrency  "$JUDGE_CONCURRENCY"
    --max-json-retries   "$MAX_JSON_RETRIES"
)
[[ -n "$LIMIT" ]]                    && CMD+=(--limit "$LIMIT")
[[ -n "$IMAGE_BATCH_SIZE" ]]         && CMD+=(--image-batch-size "$IMAGE_BATCH_SIZE")
[[ "$JUDGE_WITH_IMAGE" != "true" ]]  && CMD+=(--no-judge-image)
[[ "$STAGE_A_ONLY"     == "true" ]]  && CMD+=(--stage-a-only)
[[ "$RESUME"           == "true" ]]  && CMD+=(--resume)
[[ "$USE_FLASH_ATTN"   != "true" ]]  && CMD+=(--no-flash-attn)

echo "========================================================"
echo " ArtcomBench eval - ArtiMuse OFFLINE (Stage A/B/C/D)"
echo "  model    : $MODEL_PATH"
echo "  repo     : $ARTIMUSE_REPO"
echo "  device   : $DEVICE  dtype=$DTYPE  chat_mode=$CHAT_MODE"
echo "  data     : $DATA_JSONL    limit: ${LIMIT:-all}"
echo "  output   : $OUTPUT_DIR"
echo "  judge    : [$JUDGE_BACKEND] $JUDGE_MODEL @ $JUDGE_BASE_URL"
echo "             conc=$JUDGE_CONCURRENCY  with_image=$JUDGE_WITH_IMAGE"
echo "  flow     : stage_a_only=$STAGE_A_ONLY  resume=$RESUME"
echo "========================================================"

exec "${CMD[@]}"
