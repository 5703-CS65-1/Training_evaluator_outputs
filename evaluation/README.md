# ArtcomBench Evaluation Pipeline

针对「美学评论生成」任务的 4 阶段评测框架。输入 = (image, gold_commentary, gold_score)，输出 = 候选评论质量指标 + 整体打分回归指标。

## Pipeline 总览

```
Stage A  per-image candidate text generation
         ├── vLLM 路径 (run_vllm_offline.sh / vllm_offline.py)  适用于 Qwen3-VL / Qwen2.5-VL 等
         └── ArtiMuse 路径 (run_artimuse_offline.sh / artimuse_offline.py)
                                       │
                                       ▼
Stage B  按 10 维 heading 把 candidate text 切成 claim 列表
         (run_*.py 内自动完成；逻辑见 eval_pipeline.py:split_by_headings)
                                       │
                                       ▼
Stage C  judge API 双向打分（precision + recall）
         - precision: 每条 candidate claim 在 gold 文本中是否被支持
         - recall:    每条 gold claim 是否被 candidate 覆盖
         - 同时让 judge 给整段 candidate 输出 1-10 分（与 gold_score 比较）
         JUDGE_BACKEND 可选 deepseek / qwen / openai 兼容 endpoint
                                       │
                                       ▼
Stage D  聚合 sample → corpus 指标 (metrics.py)
         - Faithful Precision (soft)
         - Evidence Grounding
         - LF (Perception / Cognition / Emotion)
         - Claim Recall
         - Score MAE / Spearman
```

10 维 heading 顺序（Stage B 切分用）：
`Layout and Composition · Space and Perspective · Light and Shadow · Color · Details and Texture · Theme and Logic · Mood · The Overall · Creativity · Sense of Order`

## 关键文件

| 文件 | 角色 |
|---|---|
| `run_vllm_offline.sh` / `run_vllm_offline.py` / `vllm_offline.py` | Qwen3-VL 等模型走 vLLM 离线 batch chat |
| `run_artimuse_offline.sh` / `run_artimuse_offline.py` / `artimuse_offline.py` | ArtiMuse baseline 走 transformers per-image 推理 |
| `eval_pipeline.py` | Stage B/C/D 公共流程 |
| `metrics.py` | 指标定义与聚合 |
| `prompts.py` + `prompts/*.md` | Stage A/B/C 提示词 |
| `schemas.py` | judge API 的 Pydantic 结构化输出 |
| `config.py` | 默认路径、judge backend 预设 |
| `merge_lora.py` | 把 LoRA adapter 合并回 base 模型，供 vLLM 加载 |
| `local_inference.py` | 单图本地 debug 推理 |
| `install_vllm.sh` | vLLM + CUDA 安装脚本 |
| `fix_claims_dedup.py` | 对 candidate_claims 做去重修复（可选） |

## 用法

```bash
# 1) 准备测试集：把 evaluation/data/eval_text/1.jsonl 中的 image_path 改为本地路径
# 2) 准备 judge：导出 JUDGE_API_KEY / JUDGE_BASE_URL（默认走 DeepSeek，亦可切 Qwen）
export JUDGE_API_KEY=...
export JUDGE_BASE_URL=https://api.deepseek.com/v1
export JUDGE_BACKEND=deepseek      # or qwen / openai

# Qwen3-VL 系列（base 或 LoRA 合并后的权重）
#   修改 run_vllm_offline.sh 顶部的 MODEL_PATH / RUN_NAME 后：
bash run_vllm_offline.sh

# ArtiMuse baseline
bash run_artimuse_offline.sh
```

跑完产物会写到 `outputs/<RUN_NAME>/`，结构与本仓库 `../eval_results/` 完全一致。

## 与 `eval_results/` 的对应关系

| run_name | 模型 |
|---|---|
| `artimuse-all1` | ArtiMuse baseline (InternVL-3 fork) |
| `base-all1` | Qwen3-VL-8B-Thinking 原始权重 |
| `qwen3vl-8b-aesthetic-all1` | Qwen3-VL-8B-Thinking + LoRA SFT（本项目成果） |

主指标对比见仓库根目录 `README.md` 第 4 节。
