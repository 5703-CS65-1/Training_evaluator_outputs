"""Judge prompt templates for claim extraction and bidirectional scoring.

All prompt content lives in the prompts/ directory as Markdown files.
User-facing templates use Python .format() syntax:
    - {variable_name}  → replaced at runtime
    - {{ / }}          → literal { / } in the final string (e.g. JSON examples)
"""

from pathlib import Path

_DIR = Path(__file__).parent / "prompts"

def _load(filename: str) -> str:
    return (_DIR / filename).read_text(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════
# Stage B: Claim Extraction (no gold, no image)
# ═══════════════════════════════════════════════════════════════════════════

CLAIM_EXTRACTION_SYSTEM: str = _load("stage_b_claim_extraction_system.md")
CLAIM_EXTRACTION_USER: str = _load("stage_b_claim_extraction_user.md")

# ═══════════════════════════════════════════════════════════════════════════
# Stage C: Bidirectional Claim-Level Judging
# ═══════════════════════════════════════════════════════════════════════════

JUDGING_SYSTEM: str = _load("stage_c_judging_system.md")
JUDGING_USER: str = _load("stage_c_judging_user.md")
