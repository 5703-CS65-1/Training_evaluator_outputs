"""Evaluation pipeline configuration."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_JSONL = Path("1.jsonl")
IMAGE_DIR  = Path("package_1_images")
OUTPUT_DIR = Path("outputs")

# ── Candidate backend (Stage A) ────────────────────────────────────────────
# "openai" → call an OpenAI-compatible API (default; vLLM / LLaMA-Factory / DashScope / OpenAI…)
# "local"  → load HuggingFace weights (+ optional LoRA adapter) and run inference in-process
CANDIDATE_BACKEND = "openai"

# Candidate model — OpenAI-compatible API path
CANDIDATE_MODEL    = "gpt-4o"
CANDIDATE_API_KEY: str | None  = None   # None → fall back to OPENAI_API_KEY env var
CANDIDATE_BASE_URL: str | None = None   # None → OpenAI official endpoint

# Candidate model — local HF path (only used when CANDIDATE_BACKEND == "local")
LOCAL_MODEL_PATH: str | None = None     # e.g. /root/autodl-fs/models/Qwen3-VL-8B-Thinking
LOCAL_LORA_PATH:  str | None = None     # e.g. /root/autodl-fs/output/qwen3vl-8b-aesthetic-lora; None → no LoRA
LOCAL_MERGE_LORA = True                 # merge LoRA into base weights for faster inference
LOCAL_DEVICE     = "auto"               # device_map for from_pretrained (auto / cuda / cuda:0 …)
LOCAL_DTYPE      = "bfloat16"           # bfloat16 / float16 / float32
LOCAL_MAX_NEW_TOKENS = 4096
LOCAL_DO_SAMPLE  = False                # greedy by default for reproducibility
LOCAL_TEMPERATURE = 1.0
LOCAL_TOP_P       = 1.0

# ── Judge model (Stage C) — always API-based ──────────────────────────────
JUDGE_MODEL       = "gpt-4o"
JUDGE_API_KEY: str | None  = None       # None → fall back to OPENAI_API_KEY env var
JUDGE_BASE_URL: str | None = None       # None → OpenAI official endpoint
JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS  = 16384
MAX_JSON_RETRIES  = 2

# ── Thinking mode
CANDIDATE_ENABLE_THINKING = False  # candidate 模型（Stage A）思考模式开关（仅 API 路径生效）
ENABLE_THINKING           = False  # judge 模型（Stage C）思考模式开关

# ── Judge image toggle ────────────────────────────────────────────────────
JUDGE_WITH_IMAGE = True

# ── Concurrency ────────────────────────────────────────────────────────────
# Separate semaphores so candidate (often GPU-bound when local) and judge (API) don't
# block each other.  When CANDIDATE_BACKEND == "local", CANDIDATE_CONCURRENCY is forced to 1.
CANDIDATE_CONCURRENCY = 4
JUDGE_CONCURRENCY     = 8
# Legacy alias kept for backwards compatibility with run_mine.sh / older callers.
MAX_CONCURRENT_REQUESTS = 8

# ── Fixed user prompt (from gold data spec) ────────────────────────────────
CANDIDATE_PROMPT: str = (_PROMPTS_DIR / "stage_a_candidate.md").read_text(encoding="utf-8")
