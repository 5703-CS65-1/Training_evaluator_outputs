# Requirements / 环境复现

本目录提供两套独立环境的依赖清单：

| 文件 | 用途 |
|---|---|
| `environment.yml` | 训练用 Conda 环境完整导出（`conda env export --no-builds`），包含 Python 版本、conda 通道及 pip 依赖 |
| `training_requirements.txt` | 仅训练环境的 `pip freeze`，便于在已有 Python 中纯 pip 安装 |
| `eval_requirements.txt` | 评测 pipeline 需要的最小 Python 依赖（judge API 调用 + schema 校验） |

## 训练环境（LLaMA-Factory + Qwen3-VL）

推荐使用 conda 一键复现：

```bash
conda env create -f environment.yml -n llama_factory
conda activate llama_factory
# 进入仓库下的 LLaMA-Factory 源码做可编辑安装（避免重复装包）
cd ../training_framework && pip install -e ".[torch,metrics]" --no-deps
```

> 关键版本（来自 `training_requirements.txt`）：
> Python 3.11.15 · PyTorch 2.8.0+cu128 · transformers 5.2.0（含 Qwen3VL）· peft 0.18.1 · accelerate 1.11.0 · trl 0.24.0 · llamafactory 0.9.5.dev0
>
> 训练环境兼容 sm_90 (H20) 与 sm_120 (RTX 5090)。

## 评测环境（ArtcomBench）

评测 pipeline 默认走「裁判 API」+「本地推理后端」两条路：

```bash
pip install -r eval_requirements.txt
# 视使用的 Stage A 后端额外安装：
#   vLLM 路径（Qwen3-VL 系列）：参考 evaluation/install_vllm.sh
#   ArtiMuse 路径：pip install transformers>=4.46 torch>=2.4
```

API key 通过环境变量传入（参考 `evaluation/run_vllm_offline.sh` / `run_artimuse_offline.sh` 顶部）：

```bash
export JUDGE_API_KEY=...
export JUDGE_BASE_URL=...   # 如使用 DeepSeek / Qwen / OpenAI 兼容端点
```
