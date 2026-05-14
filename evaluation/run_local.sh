#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
# ArtcomBench 评测启动脚本 — 本地 HF 推理通道
# 直接加载 base + LoRA，无需先起 vLLM / LLaMA-Factory API。
# 修改下面的参数后运行: bash run_local.sh
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Python 环境（含 transformers ≥ 5.x、peft、qwen-vl-utils 的环境）──────────
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-fs/conda_envs/llama_factory/bin/python}"

# ── Candidate 模型（Stage A：本地 base + 可选 LoRA）────────────────────────
LOCAL_MODEL_PATH="/root/autodl-fs/models/Qwen3-VL-8B-Thinking"
LOCAL_LORA_PATH="/root/autodl-fs/output/qwen3vl-8b-aesthetic-lora"  # 留空字符串 "" 则不加载 LoRA
LOCAL_DEVICE="auto"
LOCAL_DTYPE="bfloat16"
LOCAL_MAX_NEW_TOKENS=4096
LOCAL_NO_MERGE_LORA=false   # true 则保留 PEFT 模块（不合并），便于调试

# ── Judge 模型（Stage C：API，必填 KEY）────────────────────────────────────
JUDGE_MODEL="qwen3.6-plus"
JUDGE_API_KEY="${DASHSCOPE_API_KEY:-}"
JUDGE_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
JUDGE_MAX_TOKENS=32768

# ── 数据路径 ──────────────────────────────────────────────────────────────
DATA_JSONL="data/eval_text/1.jsonl"
IMAGE_DIR="data/eval_text/package_1_images"
OUTPUT_DIR="outputs/qwen3vl-8b-lora-local-eval"

# ── 运行控制 ──────────────────────────────────────────────────────────────
JUDGE_CONCURRENCY=4         # judge 走 API，可并发
LIMIT="3"                   # 先少量验证流程，确认无误后改为更大或留空跑全量
ENABLE_THINKING=false       # judge 模型深度思考开关
MAX_JSON_RETRIES=2
JUDGE_WITH_IMAGE=false      # judge 是否带图（true 更准但更贵）

# ══════════════════════════════════════════════════════════════════════════
# 以下无需修改
# ══════════════════════════════════════════════════════════════════════════

if [[ -z "${JUDGE_API_KEY}" ]]; then
    echo "[ERROR] JUDGE_API_KEY 为空。请通过环境变量 DASHSCOPE_API_KEY 或脚本中直接填入。" >&2
    exit 1
fi

CMD=(
    "$PYTHON_BIN" eval_pipeline.py
    --backend             "local"
    --local-model-path    "$LOCAL_MODEL_PATH"
    --local-device        "$LOCAL_DEVICE"
    --local-dtype         "$LOCAL_DTYPE"
    --local-max-new-tokens "$LOCAL_MAX_NEW_TOKENS"
    --data                "$DATA_JSONL"
    --image-dir           "$IMAGE_DIR"
    --output              "$OUTPUT_DIR"
    --judge-model         "$JUDGE_MODEL"
    --judge-api-key       "$JUDGE_API_KEY"
    --judge-base-url      "$JUDGE_BASE_URL"
    --judge-max-tokens    "$JUDGE_MAX_TOKENS"
    --judge-concurrency   "$JUDGE_CONCURRENCY"
    --max-json-retries    "$MAX_JSON_RETRIES"
)

[[ -n "$LOCAL_LORA_PATH" ]]    && CMD+=(--local-lora-path "$LOCAL_LORA_PATH")
[[ "$LOCAL_NO_MERGE_LORA" == "true" ]] && CMD+=(--local-no-merge-lora)
[[ -n "$LIMIT" ]]              && CMD+=(--limit "$LIMIT")
[[ "$ENABLE_THINKING" == "true" ]] && CMD+=(--enable-thinking)
[[ "$JUDGE_WITH_IMAGE" != "true" ]] && CMD+=(--no-judge-image)

echo "════════════════════════════════════════════════════════"
echo " ArtcomBench Local-HF Eval — Qwen3-VL-8B (+LoRA)"
echo " local_model    : $LOCAL_MODEL_PATH"
echo " local_lora     : ${LOCAL_LORA_PATH:-<none>}"
echo " judge_model    : $JUDGE_MODEL"
echo " data           : $DATA_JSONL"
echo " output         : $OUTPUT_DIR"
echo " limit          : ${LIMIT:-all}"
echo " judge_concurrency : $JUDGE_CONCURRENCY"
echo "════════════════════════════════════════════════════════"

"${CMD[@]}"
