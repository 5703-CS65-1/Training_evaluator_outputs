#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
# 一次性安装：在 /root/autodl-fs/conda_envs/vllm 下建独立 env，装最新 vLLM
# ══════════════════════════════════════════════════════════════════════════
# 为什么单开 env：
#   - vLLM 对 transformers 版本有严格约束（通常 4.55+ 但不到 5.x 的最新点）
#   - 与现有 llama_factory env (transformers 5.2.0) 直接共用大概率冲突
#   - vLLM CUDA wheels 与项目里的 torch 2.8 + cu128 已对齐，可直接 pip 装
#
# 用法：
#   bash install_vllm.sh
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/root/autodl-fs/conda_envs/vllm}"
PY_VER="${PY_VER:-3.11}"
VLLM_VERSION="${VLLM_VERSION:-}"   # 留空 → pip 装最新；指定如 0.10.1 则锁版本

CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"

# AutoDL 系统盘 / 通常只有 30GB，pip cache 默认会写到 ~/.cache/pip，
# 装 vLLM + torch + cuda wheels 很容易爆盘。统一把 cache 重定向到 NFS。
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-fs/cache/pip}"
export TMPDIR="${TMPDIR:-/root/autodl-fs/cache/tmp}"
export HF_HOME="${HF_HOME:-/root/autodl-fs/cache/huggingface}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"
echo "[INFO] PIP_CACHE_DIR = $PIP_CACHE_DIR"
echo "[INFO] TMPDIR        = $TMPDIR"
echo "[INFO] HF_HOME       = $HF_HOME"

echo "════════════════════════════════════════════════════════"
echo " vLLM env install"
echo "   prefix : $ENV_PREFIX"
echo "   python : $PY_VER"
echo "   vllm   : ${VLLM_VERSION:-<latest>}"
echo "════════════════════════════════════════════════════════"

if [[ -d "$ENV_PREFIX" ]]; then
    echo "[INFO] env $ENV_PREFIX already exists; skipping conda create."
else
    "$CONDA_BIN" create -y --prefix "$ENV_PREFIX" "python=$PY_VER" pip
fi

PIP="$ENV_PREFIX/bin/pip"   # PIP_CACHE_DIR 已通过 env var 注入，pip 会自动使用

# 升级 pip 工具链
"$PIP" install -U pip setuptools wheel

# vLLM 主包（自动拉对应版本的 torch + transformers）
if [[ -n "$VLLM_VERSION" ]]; then
    "$PIP" install "vllm==$VLLM_VERSION"
else
    "$PIP" install vllm
fi

# 评测端如果想直接复用这个 env 跑 eval_pipeline.py，把这两个补上即可
"$PIP" install openai pydantic pillow

# qwen-vl-utils 提供 process_vision_info；vLLM 在 Qwen3-VL 路径上依赖它
"$PIP" install "qwen-vl-utils[decord]>=0.0.14" || true

echo "════════════════════════════════════════════════════════"
echo "[OK] vLLM env ready: $ENV_PREFIX"
"$PIP" show vllm | grep -E "^(Name|Version|Location):"
"$PIP" show transformers | grep -E "^(Name|Version):"
echo "════════════════════════════════════════════════════════"
echo
echo "下一步:"
echo "  1) 合并 LoRA：  /root/autodl-fs/conda_envs/llama_factory/bin/python merge_lora.py \\"
echo "                    --base /root/autodl-fs/models/Qwen3-VL-8B-Thinking \\"
echo "                    --lora /root/autodl-fs/output/qwen3vl-8b-aesthetic-lora \\"
echo "                    --out  /root/autodl-fs/models/qwen3vl-8b-aesthetic-merged"
echo "  2) 起 server： bash run_vllm_server.sh"
echo "  3) 跑评测：    bash run_vllm_eval.sh"
