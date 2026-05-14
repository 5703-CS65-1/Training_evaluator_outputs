"""
ArtcomBench Faithfulness Evaluation Pipeline
=============================================
Stage A: Run candidate model on image + fixed prompt → candidate_text
Stage B: Extract atomic claims from candidate_text → candidate_claims
Stage C: Bidirectional claim-level judging → precision + recall
Stage D: Aggregate metrics → sample & corpus metrics
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

import config
from metrics import aggregate_corpus_metrics, aggregate_sample_metrics
from prompts import (
    JUDGING_SYSTEM,
    JUDGING_USER,
)
from schemas import (
    CandidateClaim,
    CandidateOutput,
    CorpusMetrics,
    GoldSample,
    JudgingResult,
    SampleMetrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
log = logging.getLogger(__name__)

candidate_client: AsyncOpenAI | None = None
judge_client: AsyncOpenAI | None = None
candidate_semaphore: asyncio.Semaphore | None = None
judge_semaphore: asyncio.Semaphore | None = None

def _get_candidate_client() -> AsyncOpenAI:
    global candidate_client
    if candidate_client is None:
        kwargs: dict = {}
        if config.CANDIDATE_API_KEY:
            kwargs["api_key"] = config.CANDIDATE_API_KEY
        if config.CANDIDATE_BASE_URL:
            kwargs["base_url"] = config.CANDIDATE_BASE_URL
        candidate_client = AsyncOpenAI(**kwargs)
    return candidate_client

def _get_judge_client() -> AsyncOpenAI:
    global judge_client
    if judge_client is None:
        kwargs: dict = {}
        if config.JUDGE_API_KEY:
            kwargs["api_key"] = config.JUDGE_API_KEY
        if config.JUDGE_BASE_URL:
            kwargs["base_url"] = config.JUDGE_BASE_URL
        judge_client = AsyncOpenAI(**kwargs)
    return judge_client

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_samples(path: Path) -> list[GoldSample]:
    samples: list[GoldSample] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(GoldSample.model_validate_json(line))
            except Exception as exc:
                log.warning("Skipping line %d: %s", lineno, exc)
    log.info("Loaded %d samples from %s", len(samples), path)
    return samples

def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode()

def image_media_type(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower()
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        ext, "image/png"
    )

def _strip_json_fences(text: str) -> str:
    """Remove optional ```json ... ``` fences the model sometimes emits."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()

async def _call_llm(
    messages: list[dict],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    use_judge: bool = False,
) -> str:
    """Call the OpenAI chat endpoint with semaphore throttling.

    use_judge=True  → judge_client  + judge_semaphore  + ENABLE_THINKING flag
    use_judge=False → candidate_client + candidate_semaphore
    """
    sem = judge_semaphore if use_judge else candidate_semaphore
    assert sem is not None
    c = _get_judge_client() if use_judge else _get_candidate_client()
    extra: dict = {}
    if use_judge and config.ENABLE_THINKING:
        extra["extra_body"] = {"enable_thinking": True}
    elif not use_judge and config.CANDIDATE_ENABLE_THINKING:
        extra["extra_body"] = {"enable_thinking": True}
    async with sem:
        resp = await c.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
    return resp.choices[0].message.content or ""

async def _call_llm_json(
    messages: list[dict],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    use_judge: bool = False,
    retries: int | None = None,
) -> dict[str, Any]:
    """Call LLM and parse response as JSON, with automatic retries."""
    if retries is None:
        retries = config.MAX_JSON_RETRIES
    for attempt in range(1 + retries):
        raw = await _call_llm(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, use_judge=use_judge,
        )
        try:
            return json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError as exc:
            if attempt < retries:
                log.warning("JSON parse failed (attempt %d), retrying: %s", attempt + 1, exc)
            else:
                log.error("JSON parse failed after %d attempts. Raw output:\n%s", retries + 1, raw[:500])
                raise

# ═══════════════════════════════════════════════════════════════════════════
# Stage A: Candidate generation
# ═══════════════════════════════════════════════════════════════════════════

async def run_candidate_model(sample: GoldSample) -> CandidateOutput:
    image_path = config.IMAGE_DIR / sample.image

    if config.CANDIDATE_BACKEND == "local":
        # In-process HF inference (LoRA-merged or base) — see local_inference.py
        assert candidate_semaphore is not None
        from local_inference import generate_local

        async with candidate_semaphore:
            text = await generate_local(
                image_path,
                config.CANDIDATE_PROMPT,
                max_new_tokens=config.LOCAL_MAX_NEW_TOKENS,
                do_sample=config.LOCAL_DO_SAMPLE,
                temperature=config.LOCAL_TEMPERATURE,
                top_p=config.LOCAL_TOP_P,
            )
    else:
        b64 = encode_image_base64(image_path)
        mt = image_media_type(sample.image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mt};base64,{b64}"},
                    },
                    {"type": "text", "text": config.CANDIDATE_PROMPT},
                ],
            }
        ]
        text = await _call_llm(
            messages,
            model=config.CANDIDATE_MODEL,
            temperature=0,
            max_tokens=config.JUDGE_MAX_TOKENS,
            use_judge=False,
        )

    predicted_score = parse_overall_score(text)
    return CandidateOutput(
        sample_id=sample.id,
        candidate_text=text,
        predicted_score=predicted_score,
    )


# ───────────────────── Score parsing ─────────────────────
_SCORE_RE = re.compile(
    r"Overall\s+aesthetic\s+score\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
    re.IGNORECASE,
)


def parse_overall_score(text: str) -> float | None:
    """Extract the predicted overall aesthetic score from candidate output.

    Looks for `Overall aesthetic score: X.XX/10` (matching the SFT training
    target format).  Returns None when the model didn't emit a score.
    """
    if not text:
        return None
    matches = _SCORE_RE.findall(text)
    if not matches:
        return None
    try:
        # Use the last match in case the model echoes the format earlier.
        return float(matches[-1])
    except ValueError:
        return None

# ═══════════════════════════════════════════════════════════════════════════
# Stage B: Rule-based claim extraction by 10 aesthetic dimensions
# ═══════════════════════════════════════════════════════════════════════════

DIMENSIONS = [
    "Layout and Composition",
    "Space and Perspective",
    "Light and Shadow",
    "Color",
    "Details and Texture",
    "Theme and Logic",
    "Mood",
    "The Overall",
    "Creativity",
    "Sense of Order",
]

_DIMENSION_PATTERN = "|".join(re.escape(d) for d in DIMENSIONS)
# Heading detector — matches both formats produced in practice:
#   • Markdown style on own line:  **1. Layout and Composition**  /  ## Color
#   • SFT inline style:            Layout and Composition: text...
# Anchored to start-of-line via re.MULTILINE; the trailing colon / decoration is
# optional, but if no colon is present we still require the next character to be
# whitespace or end-of-line so we don't false-match inside body sentences.
_HEADER_RE = re.compile(
    r"^[\s*#>\-]*\d{0,2}\.?\s*(?P<dim>" + _DIMENSION_PATTERN
    + r")\s*\**\s*(?:[:：]|$)",
    re.MULTILINE | re.IGNORECASE,
)

# Sections we strip BEFORE claim extraction so they don't pollute claim text:
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SCORE_LINE_RE = re.compile(
    r"\n?\s*Overall\s+aesthetic\s+score\s*[:：]\s*[0-9]+(?:\.[0-9]+)?\s*/\s*10\s*",
    re.IGNORECASE,
)


def _clean_for_claim_extraction(text: str) -> str:
    """Remove `<think>` blocks and the trailing score line (SFT format)."""
    text = _THINK_RE.sub("", text)
    text = _SCORE_LINE_RE.sub("\n", text)
    return text.strip()


def extract_claims(candidate_text: str) -> list[CandidateClaim]:
    """Split candidate text into claims by the 10 aesthetic dimension headings."""
    candidate_text = _clean_for_claim_extraction(candidate_text)
    matches = list(_HEADER_RE.finditer(candidate_text))

    if not matches:
        log.warning("No dimension headings found — returning entire text as one claim")
        text = candidate_text.strip()
        if text:
            return [CandidateClaim(cand_claim_id="cand_01", text=text)]
        return []

    claims: list[CandidateClaim] = []
    for i, m in enumerate(matches):
        dim_name = m.group("dim")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(candidate_text)
        content = candidate_text[start:end].strip()
        if content:
            claims.append(
                CandidateClaim(
                    cand_claim_id=f"cand_{i + 1:02d}",
                    text=f"[{dim_name}] {content}",
                )
            )

    if not claims:
        log.warning("Dimension headings found but all sections empty — fallback")
        text = candidate_text.strip()
        if text:
            return [CandidateClaim(cand_claim_id="cand_01", text=text)]

    return claims

# ═══════════════════════════════════════════════════════════════════════════
# Stage C: Bidirectional claim-level judging
# ═══════════════════════════════════════════════════════════════════════════

async def judge_claims(
    sample: GoldSample,
    candidate_claims: list[CandidateClaim],
    candidate_text: str,
) -> JudgingResult:
    obs_json = json.dumps(
        [o.model_dump() for o in sample.observations], ensure_ascii=False
    )
    claims_json = json.dumps(
        [c.model_dump() for c in sample.claims], ensure_ascii=False
    )
    final_json = json.dumps(
        sample.final_outputs.model_dump(), ensure_ascii=False
    )
    hn_json = json.dumps(
        [h.model_dump() for h in sample.hard_negatives], ensure_ascii=False
    )
    cand_json = json.dumps(
        [c.model_dump() for c in candidate_claims], ensure_ascii=False
    )

    user_text = JUDGING_USER.format(
        observations_json=obs_json,
        claims_json=claims_json,
        final_outputs_json=final_json,
        hard_negatives_json=hn_json,
        candidate_claims_json=cand_json,
        candidate_text=candidate_text,
    )

    if config.JUDGE_WITH_IMAGE:
        image_path = config.IMAGE_DIR / sample.image
        b64 = encode_image_base64(image_path)
        mt = image_media_type(sample.image)
        user_content: str | list = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mt};base64,{b64}"},
            },
            {"type": "text", "text": user_text},
        ]
    else:
        user_text = user_text.replace(
            "You are given a painting image together with calibrated gold reference data",
            "You are given calibrated gold reference data",
            1,
        )
        user_content = user_text

    messages = [
        {"role": "system", "content": JUDGING_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    data = await _call_llm_json(
        messages,
        model=config.JUDGE_MODEL,
        temperature=config.JUDGE_TEMPERATURE,
        max_tokens=config.JUDGE_MAX_TOKENS,
        use_judge=True,
    )
    return JudgingResult.model_validate(data)

# ═══════════════════════════════════════════════════════════════════════════
# Per-sample orchestration
# ═══════════════════════════════════════════════════════════════════════════

async def process_sample(
    sample: GoldSample,
    *,
    skip_candidate: bool = False,
    existing_candidate: CandidateOutput | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for one sample. Returns all intermediate artifacts."""
    sid = sample.id

    # Stage A
    if existing_candidate is not None:
        cand = existing_candidate
    elif skip_candidate:
        raise ValueError(f"No existing candidate for {sid} and skip_candidate=True")
    else:
        log.info("[A] Generating candidate for %s", sid)
        cand = await run_candidate_model(sample)

    # Stage B
    log.info("[B] Extracting claims for %s", sid)
    candidate_claims = extract_claims(cand.candidate_text)
    if not candidate_claims:
        log.warning("No claims extracted for %s — skipping judging", sid)
        return {
            "sample_id": sid,
            "candidate_output": cand.model_dump(),
            "candidate_claims": [],
            "judging_result": None,
            "sample_metrics": None,
            "error": "no_claims_extracted",
        }

    # Stage C
    log.info("[C] Judging %d claims for %s", len(candidate_claims), sid)
    judging = await judge_claims(sample, candidate_claims, cand.candidate_text)

    # Stage D (per-sample)
    metrics = aggregate_sample_metrics(
        sid, judging, len(sample.claims),
        gt_score=sample.gt_score,
        predicted_score=cand.predicted_score,
    )

    return {
        "sample_id": sid,
        "candidate_output": cand.model_dump(),
        "candidate_claims": [c.model_dump() for c in candidate_claims],
        "judging_result": judging.model_dump(),
        "sample_metrics": metrics.model_dump(),
    }

# ═══════════════════════════════════════════════════════════════════════════
# Full run
# ═══════════════════════════════════════════════════════════════════════════

async def run_all(
    data_path: Path = config.DATA_JSONL,
    output_dir: Path = config.OUTPUT_DIR,
    limit: int | None = None,
) -> CorpusMetrics:
    global candidate_semaphore, judge_semaphore

    if config.CANDIDATE_BACKEND == "local":
        cand_conc = 1  # GPU-bound: serialize candidate inference
        if not config.LOCAL_MODEL_PATH:
            raise ValueError(
                "CANDIDATE_BACKEND='local' requires LOCAL_MODEL_PATH "
                "(pass --local-model-path or set in config.py)."
            )
        from local_inference import init_local_model
        log.info("Initializing local candidate model …")
        init_local_model(
            config.LOCAL_MODEL_PATH,
            config.LOCAL_LORA_PATH,
            device=config.LOCAL_DEVICE,
            dtype=config.LOCAL_DTYPE,
            merge_lora=config.LOCAL_MERGE_LORA,
        )
    else:
        cand_conc = config.CANDIDATE_CONCURRENCY

    candidate_semaphore = asyncio.Semaphore(cand_conc)
    judge_semaphore = asyncio.Semaphore(config.JUDGE_CONCURRENCY)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(data_path)
    if limit:
        samples = samples[:limit]

    t0 = time.time()
    log.info("Starting evaluation on %d samples …", len(samples))

    results: list[dict[str, Any]] = []
    all_sample_metrics: list[SampleMetrics] = []

    # Stream results to disk so partial progress is preserved on crash
    cand_f    = open(output_dir / "candidate_texts.jsonl",  "w", encoding="utf-8")
    claims_f  = open(output_dir / "candidate_claims.jsonl", "w", encoding="utf-8")
    judged_f  = open(output_dir / "judged_results.jsonl",   "w", encoding="utf-8")
    sample_m_f = open(output_dir / "sample_metrics.jsonl",  "w", encoding="utf-8")

    async def _wrapped(s: GoldSample) -> dict[str, Any] | None:
        sid = s.id
        try:
            # Stage A — 完成即落盘，防止后续阶段失败时丢失
            log.info("[A] Generating candidate for %s", sid)
            cand = await run_candidate_model(s)
            cand_f.write(json.dumps(cand.model_dump(), ensure_ascii=False) + "\n")
            cand_f.flush()

            # Stage B — 完成即落盘
            log.info("[B] Extracting claims for %s", sid)
            candidate_claims = extract_claims(cand.candidate_text)
            claims_f.write(
                json.dumps(
                    {"sample_id": sid, "candidate_claims": [c.model_dump() for c in candidate_claims]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            claims_f.flush()

            if not candidate_claims:
                log.warning("No claims extracted for %s — skipping judging", sid)
                return None

            # Stage C
            log.info("[C] Judging %d claims for %s", len(candidate_claims), sid)
            judging = await judge_claims(s, candidate_claims, cand.candidate_text)

            judged_f.write(
                json.dumps({"sample_id": sid, **judging.model_dump()}, ensure_ascii=False) + "\n"
            )
            judged_f.flush()

            # Stage D (per-sample)
            metrics = aggregate_sample_metrics(
                sid, judging, len(s.claims),
                gt_score=s.gt_score,
                predicted_score=cand.predicted_score,
            )
            sample_m_f.write(json.dumps(metrics.model_dump(), ensure_ascii=False) + "\n")
            sample_m_f.flush()

            return {
                "sample_id": sid,
                "candidate_output": cand.model_dump(),
                "candidate_claims": [c.model_dump() for c in candidate_claims],
                "judging_result": judging.model_dump(),
                "sample_metrics": metrics.model_dump(),
            }
        except Exception:
            log.exception("Failed processing sample %s", sid)
            return None

    tasks = [_wrapped(s) for s in samples]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is None:
            continue
        results.append(result)
        sm = SampleMetrics.model_validate(result["sample_metrics"])
        all_sample_metrics.append(sm)

    for fh in (cand_f, claims_f, judged_f, sample_m_f):
        fh.close()

    # Corpus-level aggregation
    corpus = aggregate_corpus_metrics(all_sample_metrics, results)
    (output_dir / "corpus_metrics.json").write_text(
        json.dumps(corpus.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.time() - t0
    log.info(
        "Evaluation complete: %d samples in %.1fs\n"
        "  FP_soft=%.4f  EGS=%.4f  LF-P=%.4f  LF-C=%.4f  LF-E=%.4f  CR=%.4f",
        len(all_sample_metrics),
        elapsed,
        corpus.faithful_precision_soft,
        corpus.evidence_grounding_score,
        corpus.lf_perception,
        corpus.lf_cognition,
        corpus.lf_emotion,
        corpus.claim_recall,
    )
    if corpus.num_scored:
        log.info(
            "  Score: n=%d  MAE=%.4f  Spearman=%s",
            corpus.num_scored,
            corpus.score_mae if corpus.score_mae is not None else float("nan"),
            f"{corpus.score_spearman:.4f}" if corpus.score_spearman is not None else "N/A",
        )
    return corpus

# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ArtcomBench Faithfulness Eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data",              type=Path, default=config.DATA_JSONL)
    parser.add_argument("--image-dir",         type=Path, default=None)
    parser.add_argument("--output",            type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--limit",             type=int,  default=None)
    parser.add_argument("--candidate-model",   type=str,  default=None)
    parser.add_argument("--judge-model",       type=str,  default=None)
    parser.add_argument("--judge-max-tokens",  type=int,  default=None)
    parser.add_argument("--concurrency",       type=int,  default=None,
                        help="Legacy alias for --judge-concurrency.")
    parser.add_argument("--candidate-concurrency", type=int, default=None)
    parser.add_argument("--judge-concurrency",     type=int, default=None)
    parser.add_argument("--candidate-api-key", type=str,  default=None)
    parser.add_argument("--candidate-base-url",type=str,  default=None)
    parser.add_argument("--judge-api-key",     type=str,  default=None)
    parser.add_argument("--judge-base-url",    type=str,  default=None)
    parser.add_argument("--enable-thinking",   action="store_true", default=False)
    parser.add_argument("--candidate-enable-thinking", action="store_true", default=False)
    parser.add_argument("--max-json-retries",  type=int,  default=None)
    parser.add_argument("--no-judge-image",    action="store_true", default=False)

    # ── Local-backend options ─────────────────────────────────────────────
    parser.add_argument("--backend", choices=["openai", "local"], default=None,
                        help="Stage-A candidate backend. Default: openai (config.CANDIDATE_BACKEND).")
    parser.add_argument("--local-model-path", type=str, default=None,
                        help="HF model dir for the candidate base model (when --backend local).")
    parser.add_argument("--local-lora-path", type=str, default=None,
                        help="Optional LoRA adapter dir to load on top of the base model.")
    parser.add_argument("--local-no-merge-lora", action="store_true", default=False,
                        help="Keep LoRA as PEFT module instead of merging into base weights.")
    parser.add_argument("--local-device", type=str, default=None,
                        help="device_map for from_pretrained (auto / cuda / cuda:0 …).")
    parser.add_argument("--local-dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--local-max-new-tokens", type=int, default=None)
    parser.add_argument("--local-do-sample", action="store_true", default=False)
    parser.add_argument("--local-temperature", type=float, default=None)
    parser.add_argument("--local-top-p",       type=float, default=None)

    args = parser.parse_args()

    if args.candidate_model:    config.CANDIDATE_MODEL    = args.candidate_model
    if args.judge_model:        config.JUDGE_MODEL        = args.judge_model
    if args.judge_max_tokens:   config.JUDGE_MAX_TOKENS   = args.judge_max_tokens
    if args.concurrency:
        # Legacy: applies to BOTH semaphores unless explicit per-side flag is given.
        config.MAX_CONCURRENT_REQUESTS = args.concurrency
        config.CANDIDATE_CONCURRENCY   = args.concurrency
        config.JUDGE_CONCURRENCY       = args.concurrency
    if args.candidate_concurrency: config.CANDIDATE_CONCURRENCY = args.candidate_concurrency
    if args.judge_concurrency:     config.JUDGE_CONCURRENCY     = args.judge_concurrency
    if args.image_dir:          config.IMAGE_DIR          = args.image_dir
    if args.candidate_api_key:  config.CANDIDATE_API_KEY  = args.candidate_api_key
    if args.candidate_base_url: config.CANDIDATE_BASE_URL = args.candidate_base_url
    if args.judge_api_key:      config.JUDGE_API_KEY      = args.judge_api_key
    if args.judge_base_url:     config.JUDGE_BASE_URL     = args.judge_base_url
    if args.enable_thinking:    config.ENABLE_THINKING    = True
    if args.candidate_enable_thinking: config.CANDIDATE_ENABLE_THINKING = True
    if args.max_json_retries is not None: config.MAX_JSON_RETRIES = args.max_json_retries
    if args.no_judge_image:     config.JUDGE_WITH_IMAGE   = False

    if args.backend:                config.CANDIDATE_BACKEND = args.backend
    if args.local_model_path:       config.LOCAL_MODEL_PATH  = args.local_model_path
    if args.local_lora_path:        config.LOCAL_LORA_PATH   = args.local_lora_path
    if args.local_no_merge_lora:    config.LOCAL_MERGE_LORA  = False
    if args.local_device:           config.LOCAL_DEVICE      = args.local_device
    if args.local_dtype:            config.LOCAL_DTYPE       = args.local_dtype
    if args.local_max_new_tokens:   config.LOCAL_MAX_NEW_TOKENS = args.local_max_new_tokens
    if args.local_do_sample:        config.LOCAL_DO_SAMPLE   = True
    if args.local_temperature is not None: config.LOCAL_TEMPERATURE = args.local_temperature
    if args.local_top_p       is not None: config.LOCAL_TOP_P       = args.local_top_p

    asyncio.run(run_all(data_path=args.data, output_dir=args.output, limit=args.limit))

if __name__ == "__main__":
    main()
