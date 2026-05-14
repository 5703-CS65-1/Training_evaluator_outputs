"""
Metric aggregation for ArtcomBench Faithfulness Evaluation.

Precision-side (per candidate claim):
    Faithful-Precision (soft) = mean(support_score)
    Evidence Grounding Score  = mean(support_score * evidence_score)
    LF-Perception             = mean(support_score) over perception claims
    LF-Cognition              = mean(support_score) over cognition claims
    LF-Emotion                = mean(support_score) over emotion claims

Recall-side (per gold claim):
    Claim-Recall = mean(recall_score)

Aesthetic-scoring (per sample, optional):
    abs_error      = |pred - gt|
Corpus-level scoring metrics:
    score_mae       = mean abs_error
    score_spearman  = Spearman rank correlation between (pred, gt)

All faithfulness corpus metrics use micro-average (pool all claims, then mean).
"""

from __future__ import annotations

import math
from typing import Any

from schemas import (
    CorpusMetrics,
    JudgedCandidateClaim,
    JudgingResult,
    RecallJudgment,
    SampleMetrics,
)


def aggregate_sample_metrics(
    sample_id: str,
    judging: JudgingResult,
    num_gold_claims: int,
    *,
    gt_score: float | None = None,
    predicted_score: float | None = None,
) -> SampleMetrics:
    pj = judging.precision_judgments
    rj = judging.recall_judgments

    abs_err = _score_abs_error(gt_score, predicted_score)

    m = len(pj)
    if m == 0:
        return SampleMetrics(
            sample_id=sample_id,
            num_candidate_claims=0,
            num_gold_claims=num_gold_claims,
            faithful_precision_soft=0.0,
            evidence_grounding_score=0.0,
            lf_perception=None,
            lf_cognition=None,
            lf_emotion=None,
            claim_recall=_mean_recall(rj),
            gt_score=gt_score,
            predicted_score=predicted_score,
            score_abs_error=abs_err,
        )

    fp_soft = sum(c.support_score for c in pj) / m
    egs = sum(c.support_score * c.evidence_score for c in pj) / m

    return SampleMetrics(
        sample_id=sample_id,
        num_candidate_claims=m,
        num_gold_claims=num_gold_claims,
        faithful_precision_soft=fp_soft,
        evidence_grounding_score=egs,
        lf_perception=_level_fp(pj, "perception"),
        lf_cognition=_level_fp(pj, "cognition"),
        lf_emotion=_level_fp(pj, "emotion"),
        claim_recall=_mean_recall(rj),
        gt_score=gt_score,
        predicted_score=predicted_score,
        score_abs_error=abs_err,
    )


def _score_abs_error(
    gt: float | None, pred: float | None
) -> float | None:
    if gt is None or pred is None:
        return None
    return abs(float(pred) - float(gt))


def _level_fp(
    claims: list[JudgedCandidateClaim], level: str
) -> float | None:
    subset = [c for c in claims if c.assigned_level == level]
    if not subset:
        return None
    return sum(c.support_score for c in subset) / len(subset)


def _mean_recall(rj: list[RecallJudgment]) -> float:
    if not rj:
        return 0.0
    return sum(r.recall_score for r in rj) / len(rj)


# ═══════════════════════════════════════════════════════════════════════════
# Score-correlation helpers (no scipy dependency)
# ═══════════════════════════════════════════════════════════════════════════

def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _ranks(xs: list[float]) -> list[float]:
    """Average-rank tie handling, like scipy.stats.rankdata(method='average')."""
    indexed = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed average
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


# ═══════════════════════════════════════════════════════════════════════════
# Corpus-level aggregation (micro-average)
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_corpus_metrics(
    sample_metrics_list: list[SampleMetrics],
    raw_results: list[dict[str, Any]],
) -> CorpusMetrics:
    """Micro-average across ALL claims from all samples."""

    all_pj: list[JudgedCandidateClaim] = []
    all_rj: list[RecallJudgment] = []

    for r in raw_results:
        jr = r.get("judging_result")
        if jr is None:
            continue
        result = JudgingResult.model_validate(jr)
        all_pj.extend(result.precision_judgments)
        all_rj.extend(result.recall_judgments)

    n_cand = len(all_pj)
    n_gold = len(all_rj)
    n_samples = len(sample_metrics_list)

    if n_cand == 0:
        fp_soft = egs = lf_p = lf_c = lf_e = 0.0
    else:
        fp_soft = sum(c.support_score for c in all_pj) / n_cand
        egs = sum(c.support_score * c.evidence_score for c in all_pj) / n_cand

        p_claims = [c for c in all_pj if c.assigned_level == "perception"]
        c_claims = [c for c in all_pj if c.assigned_level == "cognition"]
        e_claims = [c for c in all_pj if c.assigned_level == "emotion"]

        lf_p = (sum(c.support_score for c in p_claims) / len(p_claims)) if p_claims else 0.0
        lf_c = (sum(c.support_score for c in c_claims) / len(c_claims)) if c_claims else 0.0
        lf_e = (sum(c.support_score for c in e_claims) / len(e_claims)) if e_claims else 0.0

    cr = (sum(r.recall_score for r in all_rj) / n_gold) if n_gold else 0.0

    # ── Aesthetic scoring aggregates ──────────────────────────────────────
    paired = [
        (sm.predicted_score, sm.gt_score)
        for sm in sample_metrics_list
        if sm.predicted_score is not None and sm.gt_score is not None
    ]
    if paired:
        preds = [p for p, _ in paired]
        gts = [g for _, g in paired]
        score_mae = sum(abs(p - g) for p, g in paired) / len(paired)
        score_spearman = _spearman(preds, gts)
    else:
        score_mae = score_spearman = None

    return CorpusMetrics(
        num_samples=n_samples,
        num_total_candidate_claims=n_cand,
        num_total_gold_claims=n_gold,
        faithful_precision_soft=fp_soft,
        evidence_grounding_score=egs,
        lf_perception=lf_p,
        lf_cognition=lf_c,
        lf_emotion=lf_e,
        claim_recall=cr,
        num_scored=len(paired),
        score_mae=score_mae,
        score_spearman=score_spearman,
    )
