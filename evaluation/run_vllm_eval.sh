#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
# 走 vLLM 本地 OpenAI server 跑评测（candidate 端）
# 直接复用现有 --backend openai 通道，无需改 pipeline 代码。
# ══════════════════════════════════════════════════════════════════════════
#
# 前置条件：
#   1) bash install_vllm.sh
#   2) python merge_lora.py ...  → 生成合并模型
#   3) bash run_vllm_server.sh   → 起 server，等到 /v1/models 返回 200
#
# 评测端 Python：用现有的 llama_factory env 即可（已有 openai 包）。
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# 评测端 python（不走 vLLM env，避免 transformers 版本扰动；只要有 openai 包就行）
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-fs/conda_envs/llama_factory/bin/python}"

# ── Candidate：本地 vLLM server ────────────────────────────────────────────
CANDIDATE_MODEL="${CANDIDATE_MODEL:-qwen3vl-aesthetic}"   # 与 run_vllm_server.sh 的 SERVED_NAME 一致
CANDIDATE_BASE_URL="${CANDIDATE_BASE_URL:-http://localhost:8000/v1}"
CANDIDATE_API_KEY="${CANDIDATE_API_KEY:-EMPTY}"

# ── Judge：DashScope / OpenAI ──────────────────────────────────────────────
JUDGE_MODEL="${JUDGE_MODEL:-qwen3.6-plus}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${DASHSCOPE_API_KEY:-}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://dashscope-intl.aliyuncs.com/compatible-mode/v1}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-32768}"

# ── 数据路径 ──────────────────────────────────────────────────────────────
DATA_JSONL="${DATA_JSONL:-data/eval_text/1.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-data/eval_images/package_1_images}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3vl-8b-vllm-eval}"

# ── 运行控制 ──────────────────────────────────────────────────────────────
# vLLM 会自动 continuous-batch 多个并发请求；candidate 这边可以拉到 8 甚至更高。
CANDIDATE_CONCURRENCY="${CANDIDATE_CONCURRENCY:-8}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
LIMIT="${LIMIT:-10}"                      # 留空字符串跑全量
ENABLE_THINKING="${ENABLE_THINKING:-false}"
JUDGE_WITH_IMAGE="${JUDGE_WITH_IMAGE:-false}"
MAX_JSON_RETRIES="${MAX_JSON_RETRIES:-2}"

# ══════════════════════════════════════════════════════════════════════════
if [[ -z "$JUDGE_API_KEY" ]]; then
    echo "[ERROR] JUDGE_API_KEY 为空。export DASHSCOPE_API_KEY=... 或在脚本中填入。" >&2
    exit 1
fi

# server 健康检查
if ! curl -fsS "$CANDIDATE_BASE_URL/models" >/dev/null 2>&1; then
    echo "[ERROR] vLLM server 未就绪：$CANDIDATE_BASE_URL/models 连不通" >&2
    echo "        先跑 bash run_vllm_server.sh 并等待加载完成。" >&2
    exit 1
fi
echo "[OK] vLLM server reachable. Available models:"
curl -s "$CANDIDATE_BASE_URL/models" | head -c 400; echo

CMD=(
    "$PYTHON_BIN" eval_pipeline.py
    --backend             openai
    --candidate-model     "$CANDIDATE_MODEL"
    --candidate-base-url  "$CANDIDATE_BASE_URL"
    --candidate-api-key   "$CANDIDATE_API_KEY"
    --judge-model         "$JUDGE_MODEL"
    --judge-api-key       "$JUDGE_API_KEY"
    --judge-base-url      "$JUDGE_BASE_URL"
    --judge-max-tokens    "$JUDGE_MAX_TOKENS"
    --candidate-concurrency "$CANDIDATE_CONCURRENCY"
    --judge-concurrency   "$JUDGE_CONCURRENCY"
    --max-json-retries    "$MAX_JSON_RETRIES"
    --data                "$DATA_JSONL"
    --image-dir           "$IMAGE_DIR"
    --output              "$OUTPUT_DIR"
)

[[ -n "$LIMIT" ]]                      && CMD+=(--limit "$LIMIT")
[[ "$ENABLE_THINKING"   == "true" ]]   && CMD+=(--enable-thinking)
[[ "$JUDGE_WITH_IMAGE"  != "true" ]]   && CMD+=(--no-judge-image)

echo "════════════════════════════════════════════════════════"
echo " ArtcomBench eval — vLLM (local OpenAI server)"
echo "  candidate  : $CANDIDATE_MODEL @ $CANDIDATE_BASE_URL"
echo "  judge      : $JUDGE_MODEL @ $JUDGE_BASE_URL"
echo "  data       : $DATA_JSONL"
echo "  output     : $OUTPUT_DIR"
echo "  limit      : ${LIMIT:-all}"
echo "  cand_conc  : $CANDIDATE_CONCURRENCY    judge_conc: $JUDGE_CONCURRENCY"
echo "════════════════════════════════════════════════════════"

"${CMD[@]}"
