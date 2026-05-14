# Training_evaluator_outputs

基于 **Qwen3-VL-8B-Thinking** 的美学评论 LoRA 微调，及配套自研评测基准 **ArtcomBench** 的完整 pipeline 与关键实验结果。

---

## 目录速览

```
.
├── README.md                # 本文件
├── requirements/            # 训练 / 评测两套环境依赖
│   ├── environment.yml             # 训练 conda env 完整导出
│   ├── training_requirements.txt   # 训练 pip freeze
│   ├── eval_requirements.txt       # 评测最小依赖
│   └── README.md
├── evaluation/              # ArtcomBench 评测 pipeline 源码
│   ├── run_vllm_offline.sh         # Qwen3-VL 系列离线推理 + Stage A-D
│   ├── run_artimuse_offline.sh     # ArtiMuse baseline 离线推理 + Stage A-D
│   ├── prompts/                    # Stage A/B/C 提示词
│   └── data/eval_text/1.jsonl      # 测试集文本（图片不入仓）
├── eval_results/            # 三组关键评测产物（每组 ~15 MB）
│   ├── artimuse-all1/              # ArtiMuse baseline
│   ├── base-all1/                  # Qwen3-VL-8B-Thinking 原始权重
│   └── qwen3vl-8b-aesthetic-all1/  # LoRA 微调后权重 (本项目成果)
├── training/                # LoRA 训练配置 + 数据 + 启动脚本
│   ├── configs/qwen3vl_8b_lora_train.yaml
│   ├── configs/qwen3vl_8b_lora_smoketest.yaml
│   ├── scripts/launch_train.sh
│   ├── scripts/inference_test.py
│   └── data/sft_train.jsonl        # 2 482 条 SFT 样本（仅文本+图片相对路径）
├── training_framework/      # LLaMA-Factory 0.9.5.dev0 源码（可直接 pip -e .）
└── training_results/        # 正式 LoRA 训练日志 + 曲线 + 汇总
    ├── trainer_log.jsonl / trainer_state.json
    ├── training_loss.png / training_eval_loss.png
    ├── all_results.json / train_results.json / eval_results.json
    ├── train.log / tensorboard/
    └── model_card.md
```

> **未入仓内容**：评测图片 (`data/eval_images/`)、SFT 训练图片、模型权重（Qwen3-VL-8B-Thinking 基座 17 GB，LoRA adapter 167 MB）、训练过程中间 checkpoint-{N} 目录（每个约 500 MB）。图片为版权数据；权重可按 `training/README.md` 中的路径自行下载或复现。

---

## 1. 环境准备

```bash
# 训练
conda env create -f requirements/environment.yml -n llama_factory
conda activate llama_factory
cd training_framework && pip install -e ".[torch,metrics]" --no-deps

# 评测
pip install -r requirements/eval_requirements.txt
# vLLM 路径（Qwen3-VL）
bash evaluation/install_vllm.sh
# ArtiMuse 路径
pip install "transformers>=4.46" "torch>=2.4"
```

关键版本：Python 3.11.15 · PyTorch 2.8.0+cu128 · transformers 5.2.0 · peft 0.18.1 · llamafactory 0.9.5.dev0。

---

## 2. 训练流程（Qwen3-VL-8B-Thinking LoRA SFT）

详见 [`training/README.md`](training/README.md)，要点：

1. **数据格式**：sharegpt 风格 JSONL，每条含 `messages` + `images`。本仓库提供 `training/data/sft_train.jsonl`（2 482 条，约 20 MB，**不含图片本体**）。注入到 LLaMA-Factory 的 `data/dataset_info.json` 用 `training/data/dataset_info_snippet.json`（key=`apdd_aesthetic_thinking`）。
2. **配置**：`training/configs/qwen3vl_8b_lora_train.yaml`
   - `finetuning_type: lora`，`lora_rank: 16`，`lora_alpha: 32`，`lora_target: all`
   - `template: qwen3_vl_nothink`，`cutoff_len: 4096`
   - `per_device_train_batch_size: 1`，`gradient_accumulation_steps: 8`，`lr: 1e-4`，`epochs: 3`，bf16 + gradient checkpointing
3. **启动**：
   ```bash
   bash training/scripts/launch_train.sh            # 前台正式训练
   bash training/scripts/launch_train.sh smoketest  # 5 step 自检（约 8 分钟）
   bash training/scripts/launch_train.sh nohup      # 后台正式训练
   ```
   启动脚本会自动把 `dataset_info_snippet.json` 合并到 `training_framework/data/dataset_info.json`。

### 正式训练曲线与产物

`training_results/` 给出本仓库 `Qwen3-VL-8B + LoRA SFT` 的完整训练产物：

| 文件 | 内容 |
|---|---|
| `training_loss.png` | 训练 loss 曲线（444 step） |
| `training_eval_loss.png` | 验证集 loss 曲线 |
| `trainer_log.jsonl` | 每个 logging step 的 loss / lr / epoch |
| `trainer_state.json` | 训练过程完整状态（含 eval 节点） |
| `all_results.json` · `train_results.json` · `eval_results.json` | 汇总指标 |
| `train.log` | LLaMA-Factory stdout 全程 log (442 KB) |
| `tensorboard/` | TensorBoard event 文件 |
| `model_card.md` | Trainer 自动生成的 model card |

关键数字（来自 `all_results.json`）：

| 指标 | 值 |
|---|---|
| epoch | 3.0 |
| total steps | 444 |
| 最终 train_loss | 1.472 |
| 最终 eval_loss | **1.352** |
| 训练时长 | 2 h 29 min（≈ 8 952 s） |
| total FLOs | 8.53e17 |

eval loss 在 step 80 / 160 / 240 / 320 / 400 上的走势：1.591 → 1.432 → 1.380 → 1.359 → 1.352（持续下降，最后趋稳）。

> **不入仓**：LoRA adapter (`adapter_model.safetensors` 167 MB) 与 5 个中间 checkpoint（合计 ~2.5 GB）。下方第 4 节给出微调后权重在评测上的表现。

---

## 3. 评测流程（ArtcomBench Stage A → D）

详见 [`evaluation/README.md`](evaluation/README.md)。整体为 4 阶段：

```
Stage A  per-image candidate text generation (vLLM / ArtiMuse 本地推理)
Stage B  按 10 维 heading 切分 candidate → claims
Stage C  judge API 双向打分（precision: candidate↔gold，recall: gold↔candidate）
Stage D  聚合 sample → corpus 指标
```

两条入口脚本：

```bash
# Qwen3-VL 系列（base / 微调后）
bash evaluation/run_vllm_offline.sh

# ArtiMuse baseline（InternVL-3 fork）
bash evaluation/run_artimuse_offline.sh
```

每个 run 的 `outputs/<run_name>/` 会产出：

| 文件 | 含义 |
|---|---|
| `candidate_texts.jsonl` | Stage A 生成的 10 维评论原文 |
| `candidate_claims.jsonl` | Stage B 切分出的逐条 claim |
| `judged_results.jsonl` | Stage C 双向 judge 原始判定 |
| `sample_metrics.jsonl` | Stage D 每张图的指标 |
| `corpus_metrics.json` | Stage D 全语料聚合指标（10 维 LF、claim recall、faithfulness、score MAE / Spearman） |
| `score_metrics.json` | 单独抽出的整体打分回归指标 |

测试集文本输入：`evaluation/data/eval_text/1.jsonl`（499 张图，含 gold 评论 / gold 分数 / image 路径）。

---

## 4. 主要实验成果

下表来自 `eval_results/*/corpus_metrics.json`（n≈499 样本，全测试集）：

| 模型 | Faithful Precision (soft) | Evidence Grounding | LF Perception | LF Cognition | LF Emotion | Claim Recall | Score MAE ↓ | Score Spearman ↑ |
|---|---|---|---|---|---|---|---|---|
| ArtiMuse (baseline)                          | 0.508 | 0.481 | 0.553 | 0.460 | 0.442 | 0.425 | **1.002** | **0.617** |
| Qwen3-VL-8B-Thinking (base)                  | 0.444 | 0.312 | 0.499 | 0.384 | 0.499 | 0.597 | 2.469 | 0.484 |
| **Qwen3-VL-8B + LoRA SFT (this work)**       | **0.793** | **0.709** | **0.880** | **0.729** | **0.764** | **0.869** | 1.021 | 0.546 |

要点：
- 在「文本评论质量」相关指标（Faithful Precision、Evidence Grounding、3 项 LF、Claim Recall）上，LoRA 微调相对基座 Qwen3-VL 普遍 **+20~40 个百分点**，在 Claim Recall（0.869 vs 0.425）和 LF Perception（0.880 vs 0.553）等维度领先 ArtiMuse baseline。
- 在「整体分数回归」指标上：LoRA 微调把 base 模型的 MAE 从 2.47 降到 1.02（与 ArtiMuse 持平）；Spearman 略低于 ArtiMuse，但比 base 模型高 6 个点。
- ArtiMuse 凭专用 score head 在整体打分回归上仍最强；本工作的 LoRA SFT 在文本评论的细粒度质量上取得明显领先。

每张图的细粒度结果在 `eval_results/<run>/sample_metrics.jsonl`，逐 claim 判定在 `judged_results.jsonl`。

---

## 5. 复现 / 限制

- 训练图片与 17 GB 基座模型权重**未入仓**，请按 `training/README.md` 路径自行准备。
- 评测图片**未入仓**，`evaluation/data/eval_text/1.jsonl` 中的 `image_path` 为占位绝对路径，复现时请改成本地路径。
- 本仓库 `training_results/` 已包含完整正式训练的 log/曲线/JSON 指标；LoRA adapter 权重与中间 checkpoint 体积过大，不入仓。
- LLaMA-Factory 源码版本固定为 `0.9.5.dev0`，已置于 `training_framework/`，无需另行 `git clone`。
