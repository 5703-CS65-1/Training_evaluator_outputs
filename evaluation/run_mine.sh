#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
# ArtcomBench 评测启动脚本 — 针对 Qwen3-VL-8B LoRA 微调模型
# 修改下面的参数后直接运行: bash run_mine.sh
# ══════════════════════════════════════════════════════════════════════════

# ── Candidate 模型配置（Stage A：你的微调模型，必须已通过 API 服务启动）────
CANDIDATE_MODEL="default"                        # LLaMA-Factory API 默认模型名
CANDIDATE_API_KEY="EMPTY"
CANDIDATE_BASE_URL="http://localhost:8000/v1"    # 指向本地 LLaMA-Factory / vLLM 服务

# ── Judge 模型配置（Stage C：用强模型打分，需要真实 API Key）────────────────
JUDGE_MODEL="qwen3.6-plus"
JUDGE_API_KEY=""                                 # ← 填入你的 DashScope API Key
JUDGE_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
JUDGE_MAX_TOKENS=32768

# ── 数据路径 ──────────────────────────────────────────────────────────────
DATA_JSONL="data/eval_text/1.jsonl"
IMAGE_DIR="data/eval_text/package_1_images"
OUTPUT_DIR="outputs/qwen3vl-8b-lora-eval"

# ── 运行控制 ──────────────────────────────────────────────────────────────
CONCURRENCY=4          # 本地模型服务并发不宜过高
LIMIT="10"             # 先跑 10 条验证流程，确认没问题后改为 50 或留空全量
ENABLE_THINKING=false  # judge 模型深度思考（qwen3.6-plus 支持，但 token 消耗大）
CANDIDATE_THINKING=true  # 你的模型是 Thinking 版，建议开启
MAX_JSON_RETRIES=2
JUDGE_WITH_IMAGE=false   # Judge 不需要图片（节省成本）

# ══════════════════════════════════════════════════════════════════════════
# 以下无需修改
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

CMD=(
    python eval_pipeline.py
    --data          "$DATA_JSONL"
    --image-dir     "$IMAGE_DIR"
    --output        "$OUTPUT_DIR"
    --candidate-model    "$CANDIDATE_MODEL"
    --candidate-api-key  "$CANDIDATE_API_KEY"
    --candidate-base-url "$CANDIDATE_BASE_URL"
    --judge-model        "$JUDGE_MODEL"
    --judge-api-key      "$JUDGE_API_KEY"
    --judge-base-url     "$JUDGE_BASE_URL"
    --judge-max-tokens   "$JUDGE_MAX_TOKENS"
    --concurrency        "$CONCURRENCY"
    --max-json-retries   "$MAX_JSON_RETRIES"
)

[[ -n "$LIMIT" ]]               && CMD+=(--limit "$LIMIT")
[[ "$ENABLE_THINKING"    == "true" ]] && CMD+=(--enable-thinking)
[[ "$CANDIDATE_THINKING" == "true" ]] && CMD+=(--candidate-enable-thinking)
[[ "$JUDGE_WITH_IMAGE"   != "true" ]] && CMD+=(--no-judge-image)

echo "════════════════════════════════════════════════════════"
echo " ArtcomBench Faithfulness Eval — Qwen3-VL-8B LoRA"
echo " candidate_model : $CANDIDATE_MODEL"
echo " candidate_url   : $CANDIDATE_BASE_URL"
echo " judge_model     : $JUDGE_MODEL"
echo " concurrency     : $CONCURRENCY"
echo " limit           : ${LIMIT:-all}"
echo " candidate_think : $CANDIDATE_THINKING"
echo " judge_with_img  : $JUDGE_WITH_IMAGE"
echo " output          : $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════"

"${CMD[@]}"
