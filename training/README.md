# Training: Qwen3-VL-8B-Thinking LoRA SFT

基于 LLaMA-Factory 0.9.5.dev0 的 LoRA 微调流程。框架源码已附在仓库根目录的 `training_framework/`。

## 路径速查（开发环境）

| 内容 | 路径 |
|---|---|
| Conda 环境 | `/root/autodl-fs/conda_envs/llama_factory`（用 `requirements/environment.yml` 复现） |
| LLaMA-Factory 代码 | 仓库 `training_framework/` |
| 训练配置/脚本 | 本目录 |
| 基座模型权重 | `/root/autodl-fs/models/Qwen3-VL-8B-Thinking`（17 GB，**不入仓**） |
| Checkpoint 输出 | `/root/autodl-tmp/checkpoints/`（高速盘；不入仓） |

## 数据

- `data/sft_train.jsonl`：2 482 条多模态 SFT 样本，sharegpt 格式：
  ```json
  {
    "messages": [
      {"role": "user", "content": "<image>You are an expert art critic ..."},
      {"role": "assistant", "content": "Layout and Composition: ...\n\nSpace and Perspective: ..."}
    ],
    "images": ["/abs/path/to/img.jpg"]
  }
  ```
  仓库中 `images` 字段保留了**绝对路径**但不附图，复现时请把路径前缀替换为本地图片目录。
- `data/dataset_info_snippet.json`：对应 LLaMA-Factory `data/dataset_info.json` 中的 `apdd_aesthetic_thinking` 条目。`scripts/launch_train.sh` 会自动合并。

## 配置

| 文件 | 用途 |
|---|---|
| `configs/qwen3vl_8b_lora_train.yaml` | 正式训练（3 epoch · lora_rank 16 · bf16 · grad ckpt） |
| `configs/qwen3vl_8b_lora_smoketest.yaml` | 5 step / 5 样本自检 |

关键超参（train yaml）：

```yaml
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all
template: qwen3_vl_nothink       # SFT 数据不带 <think>，所以走 nothink template
cutoff_len: 4096
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
gradient_checkpointing: true
```

## 启动

```bash
# 自检（5 step，约 8 分钟）
bash scripts/launch_train.sh smoketest

# 正式训练（前台）
bash scripts/launch_train.sh

# 正式训练（nohup 后台，log 写到 training_results/train.log）
bash scripts/launch_train.sh nohup
```

`scripts/launch_train.sh` 会：
1. 把 `data/dataset_info_snippet.json` 合并到 `training_framework/data/dataset_info.json`，使 `dataset: apdd_aesthetic_thinking` 可用；
2. 进入 `training_framework/` 执行 `llamafactory-cli train <yaml>`。

## 推理 / 评测衔接

- `scripts/inference_test.py`：原始 transformers 推理冒烟脚本，用于验证基座模型可加载。
- 完成 LoRA 训练后，用仓库 `evaluation/merge_lora.py` 把 adapter merge 回基座，再喂给 `evaluation/run_vllm_offline.sh` 进行 ArtcomBench 评测。

## 训练 log / 曲线

仓库 `training_results/` 已包含本项目 `qwen3vl-8b-aesthetic-lora1` 正式训练的完整产物：

| 文件 | 内容 |
|---|---|
| `trainer_log.jsonl` / `trainer_state.json` | step-level loss / lr / eval 节点 |
| `training_loss.png` / `training_eval_loss.png` | 训练 / 验证 loss 曲线 |
| `all_results.json` / `train_results.json` / `eval_results.json` | 汇总指标 |
| `train.log` | LLaMA-Factory stdout 全程 log（442 KB） |
| `tensorboard/` | TensorBoard event 文件 |
| `model_card.md` | Trainer 自动生成的 model card（含 epoch × step × loss 表） |

训练关键统计（来自 `all_results.json` / `eval_results.json`）：

- epoch=3.0，total_steps=444，train_loss=1.472，**eval_loss=1.352**
- train_runtime ≈ 2 h 29 min（8 952 s）
- eval loss 走势（step 80→400）：1.591 → 1.432 → 1.380 → 1.359 → **1.352**

> LoRA adapter (`adapter_model.safetensors`, 167 MB) 与 5 个中间 `checkpoint-*` (合计 ≈ 2.5 GB) 体积过大，未入仓；如需可在自己的训练目录中找到对应文件。

## 注意事项

- **首次 I/O 慢**：autodl-fs 上加载 Qwen3-VL-8B 模型约 5 分钟，训练循环开始后不受影响。
- **Checkpoint 写到 autodl-tmp**：高速盘，避免 I/O 拖慢；重置容器会丢，正式训练后请复制到 autodl-fs。
- **Thinking 模型**：数据集若含 `<think>` 思维链，把 `template` 改成 `qwen3_vl`；否则用 `qwen3_vl_nothink`。
- **显存**：BF16 LoRA 约 18-22 GB；RTX 5090 (32 GB) / H20 (96 GB) 均可。
