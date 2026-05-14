"""Offline ArtiMuse evaluation pipeline.

Stage A: ArtiMuse offline batch inference — load InternVLChatModel once,
         loop over images, run bench 10-dim chat + score head, write
         candidate_texts.jsonl.
Stage B/C/D: reuse `eval_pipeline` (claim extraction → judging → metrics).

Differences vs `run_vllm_offline.py`:
  - candidate backend = ArtiMuse (transformers, not vLLM)
  - 没有 vLLM 引擎参数（max-model-len / gpu-memory-utilization / tp-size 等）
  - 采样固定 greedy（ArtiMuse 官方 README 做法），只暴露 max_new_tokens / device
  - 提供 --artimuse-repo 让 src/ 进 sys.path
  - 其它 resume / Stage B/C/D / metrics 全部沿用 run_vllm_offline.py 行为

用法:
  bash run_artimuse_offline.sh
或直接:
  /root/envs/artimuse/bin/python run_artimuse_offline.py \\
      --model-path /root/ArtiMuse/checkpoints/ArtiMuse \\
      --artimuse-repo /root/ArtiMuse \\
      --data data/eval_text/1.jsonl \\
      --image-dir data/eval_images/package_1_images \\
      --output outputs/artimuse-all1 \\
      --judge-api-key $JUDGE_KEY \\
      --judge-base-url https://api.deepseek.com/v1 \\
      --judge-model deepseek-chat \\
      --no-judge-image
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import config
from artimuse_offline import ArtiMuseCandidateEngine
from eval_pipeline import (
    extract_claims,
    judge_claims,
    load_samples,
)
from metrics import aggregate_corpus_metrics, aggregate_sample_metrics, _spearman
from schemas import CandidateClaim, CandidateOutput, GoldSample, SampleMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
log = logging.getLogger(__name__)


# ───────────────────────── Score parsing (mirrors eval_pipeline) ──────────────
_SCORE_RE = re.compile(
    r"Overall\s+aesthetic\s+score\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
    re.IGNORECASE,
)


def _parse_score(text: str) -> float | None:
    if not text:
        return None
    matches = _SCORE_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


# ───────────────────────── Score metrics (after Stage A) ─────────────────────
def compute_and_save_score_metrics(
    samples: list[GoldSample],
    candidates: dict[str, CandidateOutput],
    output_dir: Path,
    *,
    force: bool = False,
) -> None:
    """Compute MAE / Spearman from Stage-A results.

    Idempotent: if `score_metrics.json` already exists and its `num_scored`
    matches the current candidate set, we don't recompute.  Pass `force=True`
    to override (e.g. when Stage A produced new candidates this run).
    """
    out_path = output_dir / "score_metrics.json"

    paired: list[tuple[float, float]] = []
    for s in samples:
        cand = candidates.get(s.id)
        if cand is None or cand.predicted_score is None or s.gt_score is None:
            continue
        paired.append((cand.predicted_score, s.gt_score))

    n = len(paired)
    if n == 0:
        log.info("Score metrics: no (predicted_score, gt_score) pairs found — skipping.")
        return

    if not force and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("num_scored") == n:
                log.info(
                    "Score metrics: %s already up-to-date (num_scored=%d) — skipping recompute.",
                    out_path, n,
                )
                return
        except Exception:
            pass

    preds = [p for p, _ in paired]
    gts = [g for _, g in paired]
    mae = sum(abs(p - g) for p, g in paired) / n
    spearman = _spearman(preds, gts)

    result = {
        "num_scored": n,
        "score_mae": round(mae, 6),
        "score_spearman": round(spearman, 6) if spearman is not None else None,
    }
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "Score metrics (n=%d): MAE=%.4f  Spearman=%s",
        n, mae,
        f"{spearman:.4f}" if spearman is not None else "N/A",
    )


# ───────────────────────── Stage A: offline ArtiMuse batch ────────────────────
def _load_existing_candidates(path: Path) -> dict[str, CandidateOutput]:
    out: dict[str, CandidateOutput] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                co = CandidateOutput.model_validate_json(line)
                out[co.sample_id] = co
            except Exception as exc:
                log.warning("[resume] bad line %d in %s: %s", lineno, path, exc)
    log.info("[resume] loaded %d existing candidates from %s", len(out), path)
    return out


def run_stage_a_artimuse(
    samples: list[GoldSample],
    image_dir: Path,
    prompt: str,
    engine_factory,
    output_dir: Path,
    resume: bool,
) -> tuple[dict[str, CandidateOutput], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cand_path = output_dir / "candidate_texts.jsonl"

    existing = _load_existing_candidates(cand_path) if resume else {}

    items: list[tuple[str, Path]] = []
    skipped_missing: list[str] = []
    skipped_resume: list[str] = []
    for s in samples:
        if s.id in existing:
            skipped_resume.append(s.id)
            continue
        p = image_dir / s.image
        if not p.exists():
            log.warning("Image missing for %s: %s — skipping", s.id, p)
            skipped_missing.append(s.id)
            continue
        items.append((s.id, p))

    if skipped_missing:
        log.warning("Stage A: %d sample(s) skipped (missing images).", len(skipped_missing))
    if skipped_resume:
        log.info("Stage A: %d sample(s) skipped (already in candidate_texts.jsonl).", len(skipped_resume))

    new_outputs: dict[str, CandidateOutput] = {}
    if items:
        engine = engine_factory()
        raw = engine.run_batch(items, prompt)
        for sid, (text, reasoning) in raw.items():
            new_outputs[sid] = CandidateOutput(
                sample_id=sid,
                candidate_text=text,
                predicted_score=_parse_score(text),
                reasoning_text=reasoning,
            )
        # Append (resume) or rewrite — same behavior as run_vllm_offline.py.
        mode = "a" if resume and cand_path.exists() else "w"
        with cand_path.open(mode, encoding="utf-8") as fh:
            if mode == "w":
                for co in existing.values():
                    fh.write(co.model_dump_json() + "\n")
            for co in new_outputs.values():
                fh.write(co.model_dump_json() + "\n")
        log.info("Stage A: wrote %d new candidate(s) to %s", len(new_outputs), cand_path)
    else:
        log.info("Stage A: nothing to do (all samples already have candidates).")

    merged = {**existing, **new_outputs}
    return merged, len(new_outputs)


# ───────────────────────── Stage B: local claim extraction (resumable) ───────
def _load_existing_claims(path: Path) -> dict[str, list[CandidateClaim]]:
    """Read candidate_claims.jsonl from a previous run.

    Returns {sid: [CandidateClaim, ...]} for every sample whose claims have
    already been written.  Empty-list entries are preserved (means we already
    tried and got zero claims — don't re-extract).
    """
    out: dict[str, list[CandidateClaim]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj["sample_id"]
                claims = [CandidateClaim.model_validate(c) for c in obj.get("candidate_claims", [])]
                out[sid] = claims
            except Exception as exc:
                log.warning("[resume] bad claim line %d in %s: %s", lineno, path, exc)
    return out


def run_stage_b(
    samples: list[GoldSample],
    candidates: dict[str, CandidateOutput],
    output_dir: Path,
    resume: bool,
) -> dict[str, list[CandidateClaim]]:
    """Extract candidate claims for every sample (Stage B). 本地无 API。

    Idempotent: 已经在 `candidate_claims.jsonl` 里的 sample 直接复用，不重抽。
    仅对缺失的 sample 抽取并 append。如果一个 sample 都没漏，函数直接 short-circuit
    返回，不会触碰 jsonl 文件（也不会改 mtime）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    claims_path = output_dir / "candidate_claims.jsonl"

    existing = _load_existing_claims(claims_path) if resume else {}
    if existing:
        log.info("[resume] %d sample(s) already have extracted claims.", len(existing))

    todo: list[GoldSample] = [
        s for s in samples
        if s.id in candidates and s.id not in existing
    ]

    if not todo:
        log.info("Stage B: nothing to do — claims for all %d sample(s) are cached.", len(existing))
        return existing

    log.info("Stage B: extracting claims for %d new sample(s) …", len(todo))
    # 用 append 保护已有内容；resume=False 走 'w' 也只在这条路径上覆写（与旧行为一致）。
    mode = "a" if (resume and claims_path.exists()) else "w"
    with claims_path.open(mode, encoding="utf-8") as fh:
        if mode == "w":
            for sid, claims in existing.items():
                fh.write(json.dumps(
                    {"sample_id": sid, "candidate_claims": [c.model_dump() for c in claims]},
                    ensure_ascii=False,
                ) + "\n")
        for s in todo:
            cand = candidates[s.id]
            try:
                claims = extract_claims(cand.candidate_text)
            except Exception:
                log.exception("[B] extract_claims failed for %s", s.id)
                continue
            existing[s.id] = claims
            fh.write(json.dumps(
                {"sample_id": s.id, "candidate_claims": [c.model_dump() for c in claims]},
                ensure_ascii=False,
            ) + "\n")
            fh.flush()
    log.info("Stage B done: %d total sample(s) with claims on disk.", len(existing))
    return existing


# ───────────────────────── Stage C/D async per sample ────────────────────────
async def process_one(
    sample: GoldSample,
    cand: CandidateOutput,
    candidate_claims: list[CandidateClaim],
    files: dict[str, Any],
) -> dict[str, Any] | None:
    sid = sample.id
    try:
        if not candidate_claims:
            log.warning("No claims for %s — skipping judging", sid)
            return None

        log.info("[C] Judging %d claims for %s", len(candidate_claims), sid)
        judging = await judge_claims(sample, candidate_claims, cand.candidate_text)
        files["judged"].write(
            json.dumps({"sample_id": sid, **judging.model_dump()}, ensure_ascii=False) + "\n"
        )
        files["judged"].flush()

        metrics = aggregate_sample_metrics(
            sid, judging, len(sample.claims),
            gt_score=sample.gt_score,
            predicted_score=cand.predicted_score,
        )
        files["sample_m"].write(json.dumps(metrics.model_dump(), ensure_ascii=False) + "\n")
        files["sample_m"].flush()

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


def _load_done_sample_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "sample_id" in obj:
                    done.add(obj["sample_id"])
            except Exception:
                pass
    return done


def _load_existing_sample_metrics(path: Path) -> list[SampleMetrics]:
    out: list[SampleMetrics] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(SampleMetrics.model_validate_json(line))
            except Exception:
                pass
    return out


def _load_existing_judged(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "sample_id" in obj:
                    out[obj["sample_id"]] = obj
            except Exception:
                pass
    return out


async def run_stage_cd(
    samples: list[GoldSample],
    candidates: dict[str, CandidateOutput],
    all_claims: dict[str, list[CandidateClaim]],
    output_dir: Path,
    judge_concurrency: int,
    resume: bool,
) -> None:
    """Stage C (judge API) + Stage D (aggregation). Resume-aware on `judged_results.jsonl`.

    Corpus metrics is always (re)written at the end as long as there exists at
    least one sample we'd want to aggregate (either newly judged this run or
    already on disk).  If absolutely nothing needs judging *and* corpus_metrics
    already exists, we leave it alone too.
    """
    import eval_pipeline

    eval_pipeline.judge_semaphore = asyncio.Semaphore(judge_concurrency)
    eval_pipeline.candidate_semaphore = asyncio.Semaphore(1)  # unused but checked

    output_dir.mkdir(parents=True, exist_ok=True)

    judged_path = output_dir / "judged_results.jsonl"
    sample_m_path = output_dir / "sample_metrics.jsonl"
    corpus_path = output_dir / "corpus_metrics.json"

    done_ids: set[str] = _load_done_sample_ids(judged_path) if resume else set()
    if done_ids:
        log.info("[resume] %d sample(s) already judged — will skip Stage C.", len(done_ids))

    todo_samples = [
        s for s in samples
        if s.id in candidates and s.id in all_claims and s.id not in done_ids
    ]

    if not todo_samples:
        log.info("Stage C: nothing to do — all samples already judged.")
        if corpus_path.exists():
            log.info("Corpus metrics already on disk (%s) — leaving as-is.", corpus_path)
            return
        # Corpus metrics 缺失但 judge 全跑过了 → 从落盘文件重建
        log.info("Rebuilding corpus_metrics.json from existing sample_metrics.jsonl …")
        existing_metrics = _load_existing_sample_metrics(sample_m_path)
        existing_judged = _load_existing_judged(judged_path)
        results = []
        for s in samples:
            if s.id not in candidates or s.id not in existing_judged:
                continue
            j = existing_judged[s.id]
            results.append({
                "sample_id": s.id,
                "candidate_output": candidates[s.id].model_dump(),
                "candidate_claims": [c.model_dump() for c in all_claims.get(s.id, [])],
                "judging_result": j,
                "sample_metrics": next(
                    (m.model_dump() for m in existing_metrics if m.sample_id == s.id),
                    None,
                ),
            })
        if existing_metrics:
            corpus = aggregate_corpus_metrics(existing_metrics, results)
            corpus_path.write_text(
                json.dumps(corpus.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return

    mode = "a" if resume else "w"
    files = {
        "judged":   open(judged_path,   mode, encoding="utf-8"),
        "sample_m": open(sample_m_path, mode, encoding="utf-8"),
    }

    tasks = [
        process_one(s, candidates[s.id], all_claims[s.id], files)
        for s in todo_samples
    ]

    new_results: list[dict[str, Any]] = []
    new_metrics: list[SampleMetrics] = []
    for coro in asyncio.as_completed(tasks):
        r = await coro
        if r is None:
            continue
        new_results.append(r)
        new_metrics.append(SampleMetrics.model_validate(r["sample_metrics"]))

    for fh in files.values():
        fh.close()

    # Stage D: 用本次 + 历史的 sample_metrics 一起算 corpus
    existing_metrics = _load_existing_sample_metrics(sample_m_path) if resume else new_metrics
    all_metrics = existing_metrics if existing_metrics else new_metrics
    # 反序列化已存在的 judged 结果，让 aggregate_corpus_metrics 拿到完整 results
    existing_judged = _load_existing_judged(judged_path)
    results: list[dict[str, Any]] = []
    for s in samples:
        if s.id not in candidates or s.id not in existing_judged:
            continue
        j = existing_judged[s.id]
        sm = next((m for m in all_metrics if m.sample_id == s.id), None)
        if sm is None:
            continue
        results.append({
            "sample_id": s.id,
            "candidate_output": candidates[s.id].model_dump(),
            "candidate_claims": [c.model_dump() for c in all_claims.get(s.id, [])],
            "judging_result": j,
            "sample_metrics": sm.model_dump(),
        })

    if not all_metrics:
        log.warning("No sample_metrics to aggregate — skipping corpus_metrics.json.")
        return

    corpus = aggregate_corpus_metrics(all_metrics, results)
    corpus_path.write_text(
        json.dumps(corpus.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info(
        "Eval done: %d scored | FP_soft=%.4f EGS=%.4f LF-P=%.4f LF-C=%.4f LF-E=%.4f CR=%.4f",
        len(all_metrics),
        corpus.faithful_precision_soft,
        corpus.evidence_grounding_score,
        corpus.lf_perception, corpus.lf_cognition, corpus.lf_emotion,
        corpus.claim_recall,
    )
    if corpus.num_scored:
        log.info(
            "  Score: n=%d MAE=%.4f Spearman=%s",
            corpus.num_scored,
            corpus.score_mae if corpus.score_mae is not None else float("nan"),
            f"{corpus.score_spearman:.4f}" if corpus.score_spearman is not None else "N/A",
        )


# ─────────────────────────────────── main ─────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ArtcomBench offline ArtiMuse eval")

    # data / paths
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)

    # ArtiMuse model
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to the ArtiMuse checkpoint dir (e.g. /root/ArtiMuse/checkpoints/ArtiMuse).")
    p.add_argument("--artimuse-repo", type=str, required=True,
                   help="Path to the ArtiMuse repo root (so its src/ can be put on sys.path).")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument("--chat-mode", choices=["batch", "single"], default="batch",
                   help="batch: 10 dims in one forward (fast, slight quality trade-off per ArtiMuse README); "
                        "single: dim-by-dim chat (better quality, slower).")
    p.add_argument("--image-batch-size", type=int, default=4,
                   help="Number of images per batch_chat call (K images × 10 dims). "
                        "A800 80GB: 4 is safe, try 6 for higher throughput.")

    # Judge (Stage C) — OpenAI-compatible API
    p.add_argument("--judge-model", type=str, default=None)
    p.add_argument("--judge-api-key", type=str, default=None)
    p.add_argument("--judge-base-url", type=str, default=None)
    p.add_argument("--judge-max-tokens", type=int, default=None)
    p.add_argument("--judge-concurrency", type=int, default=8)
    p.add_argument("--judge-temperature", type=float, default=None)
    p.add_argument("--enable-thinking", action="store_true",
                   help="Pass enable_thinking=True to judge model (Qwen3 series).")
    p.add_argument("--no-judge-image", action="store_true",
                   help="Don't send image to judge (required for text-only judges like deepseek-chat).")
    p.add_argument("--max-json-retries", type=int, default=None)

    # Run-control
    p.add_argument("--stage-a-only", action="store_true")
    p.add_argument("--resume", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Wire CLI → eval_pipeline.config
    if args.judge_model:        config.JUDGE_MODEL        = args.judge_model
    if args.judge_api_key:      config.JUDGE_API_KEY      = args.judge_api_key
    if args.judge_base_url:     config.JUDGE_BASE_URL     = args.judge_base_url
    if args.judge_max_tokens:   config.JUDGE_MAX_TOKENS   = args.judge_max_tokens
    if args.judge_temperature is not None: config.JUDGE_TEMPERATURE = args.judge_temperature
    if args.no_judge_image:     config.JUDGE_WITH_IMAGE   = False
    if args.enable_thinking:    config.ENABLE_THINKING    = True
    if args.max_json_retries is not None: config.MAX_JSON_RETRIES = args.max_json_retries
    config.IMAGE_DIR = args.image_dir

    if not args.stage_a_only and not config.JUDGE_API_KEY:
        raise SystemExit(
            "[ERROR] judge api key is empty — pass --judge-api-key, set "
            "OPENAI_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY, or run with --stage-a-only"
        )

    samples = load_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]
    log.info("Loaded %d samples (limit=%s, resume=%s, stage_a_only=%s)",
             len(samples), args.limit, args.resume, args.stage_a_only)

    # ── Stage A: ArtiMuse offline batch (with resume support)
    t_a = time.time()

    def _make_engine() -> ArtiMuseCandidateEngine:
        return ArtiMuseCandidateEngine(
            model_path=args.model_path,
            artimuse_repo=args.artimuse_repo,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            use_flash_attn=not args.no_flash_attn,
            chat_mode=args.chat_mode,
            image_batch_size=args.image_batch_size,
        )

    candidates, num_new_candidates = run_stage_a_artimuse(
        samples, args.image_dir, config.CANDIDATE_PROMPT,
        engine_factory=_make_engine,
        output_dir=args.output,
        resume=args.resume,
    )
    log.info("Stage A done: %d total candidate(s) available (+%d new this run) in %.1fs",
             len(candidates), num_new_candidates, time.time() - t_a)

    # 仅在 Stage A 有新产物 或 score_metrics.json 还不存在时重算（idempotent）。
    compute_and_save_score_metrics(
        samples, candidates, args.output,
        force=(num_new_candidates > 0),
    )

    if args.stage_a_only:
        log.info("--stage-a-only set; exiting before Stage B/C/D.")
        return

    # ── Stage B: local claim extraction (no API, resumable on candidate_claims.jsonl)
    t_b = time.time()
    all_claims = run_stage_b(samples, candidates, args.output, resume=args.resume)
    log.info("Stage B done in %.1fs", time.time() - t_b)

    # ── Stage C/D async (resumable on judged_results.jsonl)
    t_cd = time.time()
    asyncio.run(run_stage_cd(
        samples, candidates, all_claims, args.output,
        judge_concurrency=args.judge_concurrency,
        resume=args.resume,
    ))
    log.info("Stage C/D done in %.1fs", time.time() - t_cd)


if __name__ == "__main__":
    main()
