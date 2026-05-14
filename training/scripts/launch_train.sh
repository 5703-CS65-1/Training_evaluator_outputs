#!/usr/bin/env bash
# Qwen3-VL-8B-Thinking LoRA 训练启动脚本
# 用法:
#   bash launch_train.sh                # 正式训练
#   bash launch_train.sh smoketest      # 5-step 烟测
#   bash launch_train.sh nohup          # 后台正式训练

set -euo pipefail

MODE="${1:-train}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-${REPO_ROOT}/training_framework}"
CONFIG_DIR="${REPO_ROOT}/training/configs"
LOG_DIR="${REPO_ROOT}/training_results"
mkdir -p "${LOG_DIR}"

# 注册自定义数据集 (LLaMA-Factory 需要在 data/dataset_info.json 中能查到 apdd_aesthetic_thinking)
SNIPPET="${REPO_ROOT}/training/data/dataset_info_snippet.json"
DSET_INFO="${LLAMAFACTORY_DIR}/data/dataset_info.json"
if [ -f "${SNIPPET}" ] && [ -f "${DSET_INFO}" ]; then
  python - <<PY
import json, sys
with open("${DSET_INFO}") as f: info = json.load(f)
with open("${SNIPPET}") as f: snip = json.load(f)
info.update(snip)
with open("${DSET_INFO}", "w") as f: json.dump(info, f, indent=2, ensure_ascii=False)
print("[launch_train] dataset_info merged: apdd_aesthetic_thinking")
PY
fi

case "${MODE}" in
  smoketest)
    YAML="${CONFIG_DIR}/qwen3vl_8b_lora_smoketest.yaml"
    cd "${LLAMAFACTORY_DIR}" && llamafactory-cli train "${YAML}"
    ;;
  nohup)
    YAML="${CONFIG_DIR}/qwen3vl_8b_lora_train.yaml"
    cd "${LLAMAFACTORY_DIR}"
    nohup llamafactory-cli train "${YAML}" \
      > "${LOG_DIR}/train.log" 2>&1 &
    echo $! > "${LOG_DIR}/train.pid"
    echo "[launch_train] PID=$(cat ${LOG_DIR}/train.pid), log=${LOG_DIR}/train.log"
    ;;
  train|*)
    YAML="${CONFIG_DIR}/qwen3vl_8b_lora_train.yaml"
    cd "${LLAMAFACTORY_DIR}" && llamafactory-cli train "${YAML}"
    ;;
esac
