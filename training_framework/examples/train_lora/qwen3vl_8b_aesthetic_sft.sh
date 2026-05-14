#!/usr/bin/env bash
# 启动脚本：Qwen3-VL-8B-Thinking 美学评论 faithful 能力 LoRA SFT
# 单卡 H20 (96GB)
set -euo pipefail

LF_DIR="/root/autodl-fs/code/LLaMA-Factory"
ENV_PY="/root/autodl-fs/conda_envs/llama_factory/bin/python"
CONFIG="${LF_DIR}/examples/train_lora/qwen3vl_8b_aesthetic_sft.yaml"

# 从 YAML 读取，YAML 是唯一需要修改的地方
_yaml_val() { grep "^${1}:" "${CONFIG}" | awk '{print $2}'; }
MODEL_PATH=$(_yaml_val model_name_or_path)
OUT_DIR=$(_yaml_val output_dir)
TB_DIR=$(_yaml_val logging_dir)

cd "${LF_DIR}"

# 预检：模型 / 数据 / GPU
[ -d "${MODEL_PATH}" ] || { echo "[ERR] model dir missing: ${MODEL_PATH}"; exit 1; }
[ -f "/root/autodl-fs/data/sft_train.jsonl" ] || { echo "[ERR] sft jsonl missing"; exit 1; }
[ -d "/root/autodl-fs/data/train_images" ] || { echo "[ERR] train_images dir missing"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || { echo "[ERR] no GPU"; exit 1; }

mkdir -p "${TB_DIR}" "${OUT_DIR}"

# CUDA / 显存优化
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=0

RUN_LOG="${OUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] start training"
echo "  config:        ${CONFIG}"
echo "  output_dir:    ${OUT_DIR}"
echo "  tensorboard:   ${TB_DIR}"
echo "  log:           ${RUN_LOG}"
echo

"${ENV_PY}" -m llamafactory.cli train "${CONFIG}" 2>&1 | tee "${RUN_LOG}"
