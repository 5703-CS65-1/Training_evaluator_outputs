"""Pydantic schemas for gold data, intermediate results, and judge outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# Gold data schemas (input)
# ═══════════════════════════════════════════════════════════════════════════

class Observation(BaseModel):
    id: str
    text: str
    boxes: list[list[int]]
    visual_type: str
    certainty: Literal["high", "medium", "low"]

class GoldClaim(BaseModel):
    id: str
    level: Literal["perception", "cognition", "emotion"]
    dimension: str
    text: str
    support_label: Literal["supported", "weakly_supported", "insufficient_evidence"]
    source_obs_ids: list[str]

class FinalOutputs(BaseModel):
    thinking_chain: str
    commentary: str
    short_summary: str

class HardNegative(BaseModel):
    text: str
    label: Literal["unsupported", "unverifiable"]
    error_type: str

class GoldSample(BaseModel):
    id: str
    image: str
    observations: list[Observation]
    claims: list[GoldClaim]
    final_outputs: FinalOutputs
    hard_negatives: list[HardNegative]
    gt_score: float | None = None  # 10-point overall aesthetic ground-truth score

# ═══════════════════════════════════════════════════════════════════════════
# Stage A output
# ═══════════════════════════════════════════════════════════════════════════

class CandidateOutput(BaseModel):
    sample_id: str
    candidate_text: str
    predicted_score: float | None = None  # parsed from "Overall aesthetic score: X.XX/10"
    reasoning_text: str | None = None     # split-out <think>...</think> contents (vLLM offline only)

# ═══════════════════════════════════════════════════════════════════════════
# Stage B output – claim extraction
# ═══════════════════════════════════════════════════════════════════════════

class CandidateClaim(BaseModel):
    cand_claim_id: str
    text: str

class ClaimExtractionResult(BaseModel):
    candidate_claims: list[CandidateClaim]

# ═══════════════════════════════════════════════════════════════════════════
# Stage C output – bidirectional judging
# ═══════════════════════════════════════════════════════════════════════════

class JudgedCandidateClaim(BaseModel):
    cand_claim_id: str
    text: str
    assigned_level: Literal["perception", "cognition", "emotion"]
    matched_gold_claim_ids: list[str] = Field(default_factory=list)
    matched_obs_ids: list[str] = Field(default_factory=list)
    support_score: Literal[1.0, 0.5, 0.0]
    evidence_score: Literal[1.0, 0.75, 0.5, 0.25, 0.0]
    reason: str

class RecallJudgment(BaseModel):
    gold_claim_id: str
    recall_score: Literal[1.0, 0.5, 0.0]
    matched_cand_claim_ids: list[str] = Field(default_factory=list)
    reason: str

class JudgingResult(BaseModel):
    precision_judgments: list[JudgedCandidateClaim]
    recall_judgments: list[RecallJudgment]

# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

class SampleMetrics(BaseModel):
    sample_id: str
    num_candidate_claims: int
    num_gold_claims: int
    faithful_precision_soft: float
    evidence_grounding_score: float
    lf_perception: float | None = None
    lf_cognition: float | None = None
    lf_emotion: float | None = None
    claim_recall: float
    # ── Aesthetic scoring (optional, only present when gt_score & predicted_score available) ──
    gt_score: float | None = None
    predicted_score: float | None = None
    score_abs_error: float | None = None

class CorpusMetrics(BaseModel):
    num_samples: int
    num_total_candidate_claims: int
    num_total_gold_claims: int
    faithful_precision_soft: float
    evidence_grounding_score: float
    lf_perception: float
    lf_cognition: float
    lf_emotion: float
    claim_recall: float
    # ── Aesthetic scoring metrics ──
    num_scored: int = 0          # # samples with both gt_score and predicted_score
    score_mae: float | None = None
    score_spearman: float | None = None
