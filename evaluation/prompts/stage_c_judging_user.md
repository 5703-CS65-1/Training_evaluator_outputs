You are given a painting image together with calibrated gold reference data and a set of candidate claims extracted from a model's commentary.

## Gold reference (all equally authoritative)

observations, claims, and final_outputs below have been human-calibrated and are ALL fully trustworthy. They jointly serve as the unified gold reference for EVERY scoring dimension (support, evidence grounding, and recall).

### Gold observations
{observations_json}

### Gold claims
{claims_json}

### Gold final_outputs
{final_outputs_json}

### Hard negatives (examples of unsupported / unverifiable content)
{hard_negatives_json}

## Candidate data

### Candidate claims (medium-grained evaluative claims)
{candidate_claims_json}

### Candidate raw text (for recall evaluation)
{candidate_text}

───────────────────────────────────────────────────
## Task A — Precision scoring

For EACH candidate claim, you MUST first complete the following three steps before assigning any score. Include this reasoning in the "reason" field.

**Step 1 — Match check**: Identify the most relevant gold claim and/or observation. Write its ID. If none exist, write "no match".
**Step 2 — Cap check**: If the matched gold claim has support_label = "weakly_supported", the support_score ceiling is 0.5. State explicitly: "Cap applies: YES / NO".
**Step 3 — Score decision**: Based on Steps 1–2, assign support_score. If you have no match from Step 1, support_score = 0.0 unless a gold observation directly supports the claim.

Then evaluate:

1. **assigned_level**: perception / cognition / emotion
2. **matched_gold_claim_ids**: which gold claim(s) align with this candidate claim (empty list if none)
3. **matched_obs_ids**: which observation(s) relate (empty list if none)
4. **support_score** (1.0 / 0.5 / 0.0):
   - 1.0 = The claim is clearly and directly supported by at least one gold claim (support_label = "supported") or by one or more gold observations. The core judgment matches without requiring inferential leaps.
   - 0.5 = Use ONLY in one of these specific situations:
     - (a) The candidate claim matches a gold claim whose support_label = "weakly_supported" — in this case 0.5 is the MAXIMUM allowed score.
     - (b) The candidate claim captures the correct general direction of a gold claim or observation, but omits a key qualifier, conflates two distinct aspects, or uses hedged language where the gold is definitive.
     - (c) The claim is partially grounded — one part is supported but another part within the same claim is unverifiable or speculative.
   - 0.0 = The claim is unsupported, hallucinated, speculative, involves author intent / art-historical inference not present in the gold reference, or directly resembles a hard negative example.
   - IMPORTANT — Cap rule: if the closest matching gold claim has support_label = "weakly_supported", the candidate's support_score MUST NOT exceed 0.5, even if the candidate's phrasing is confident.
   - Do NOT use 0.5 simply because you are uncertain. If there is no clear positive match to any gold claim or observation, default to 0.0.
5. **evidence_score** (1.0 / 0.75 / 0.5 / 0.25 / 0.0):
   - 1.0 = directly, clearly anchored to image + observations
   - 0.75 = well grounded, requires synthesizing multiple references
   - 0.5 = moderate grounding with some inferential leap
   - 0.25 = weak grounding
   - 0.0 = no grounding
   - If support_score is 0.0, evidence_score MUST be 0.0 or 0.25 only.
   - Do NOT let evidence_score exceed support_score by more than one tier.
6. **reason**: one-sentence justification that includes your Step 1–3 reasoning.

Observations CAN rescue a faithful paraphrase that does not match any gold claim verbatim. A candidate claim grounded in the gold reference should still receive credit even without exact wording overlap.

**SCORING PROHIBITIONS — violating any of these is an error:**
- Do NOT give support_score = 0.5 simply because you are uncertain or the claim seems "somewhat relevant". 0.5 requires a specific justification matching sub-case (a), (b), or (c) above.
- Do NOT give support_score > 0.0 for any claim that references the artist's intention, biography, or makes art-historical attributions not present in the gold reference.
- Do NOT give support_score > 0.0 for any claim that directly resembles a hard negative example.

───────────────────────────────────────────────────
## Task B — Recall scoring

For EACH gold claim, judge whether the candidate's RAW TEXT (not just the extracted claims) covers it:

1. **gold_claim_id**: the id of the gold claim
2. **recall_score** (1.0 / 0.5 / 0.0):
   - 1.0 = The candidate raw text explicitly addresses the core evaluative judgment of this gold claim. The key subject, predicate, and evaluative direction must all be present, though exact wording is not required.
   - 0.5 = Use ONLY in one of these specific situations:
     - (a) The candidate text mentions the relevant dimension or visual element but stops short of making the evaluative judgment the gold claim makes (e.g., gold claim says "the composition creates a strong upward rhythm" and candidate only says "the composition is structured").
     - (b) The gold claim's judgment is implied by the candidate's overall argument but never stated directly — the reader would need to infer it.
     - (c) The candidate covers only part of a compound gold claim (e.g., gold claim addresses both light and mood together, candidate addresses only light).
   - 0.0 = The gold claim's subject or evaluative judgment is entirely absent from the candidate text. Do not award 0.5 out of charity — if you cannot identify specific sentences in the candidate that correspond to this gold claim, score 0.0.
3. **matched_cand_claim_ids**: which candidate claim(s) contribute to the coverage (empty list if none)
4. **reason**: one-sentence justification. If scoring 0.5, state which sub-case (a/b/c) applies.

───────────────────────────────────────────────────
## Scoring examples

**Example 1 — support_score = 1.0**
Candidate: "The lower section features dense, overlapping rock formations."
Step 1: Matches obs_03 ("overlapping layered rocks occupy the lower third"). Step 2: Cap applies: NO. Step 3: 1.0.
→ support_score = 1.0, evidence_score = 1.0

**Example 2 — support_score = 0.5 (sub-case a)**
Candidate: "The mood feels somewhat melancholic and withdrawn."
Step 1: Matches claim_08 (support_label = "weakly_supported"). Step 2: Cap applies: YES — ceiling is 0.5. Step 3: 0.5.
→ support_score = 0.5, evidence_score = 0.5

**Example 3 — support_score = 0.0**
Candidate: "The artist intended this work as a meditation on mortality."
Step 1: No match — author intent not in gold. Step 2: N/A. Step 3: 0.0.
→ support_score = 0.0, evidence_score = 0.0

**Example 4 — recall_score = 0.5 (sub-case a)**
Gold claim: "The diagonal composition creates a dynamic upward tension."
Candidate contains: "the composition is organized along diagonal lines" — mentions the element but omits the evaluative judgment about tension.
→ recall_score = 0.5

**Example 5 — recall_score = 0.0**
Gold claim: "Dry-brush textures in the upper canopy evoke weathered resilience."
Candidate: no mention of dry-brush technique or canopy texture anywhere.
→ recall_score = 0.0

───────────────────────────────────────────────────
## Output format

Return valid JSON only, with no markdown fencing. Use this exact schema:

{{
  "precision_judgments": [
    {{
      "cand_claim_id": "cand_01",
      "text": "...",
      "assigned_level": "perception",
      "matched_gold_claim_ids": ["claim_02"],
      "matched_obs_ids": ["obs_03"],
      "support_score": 1.0,
      "evidence_score": 1.0,
      "reason": "..."
    }}
  ],
  "recall_judgments": [
    {{
      "gold_claim_id": "claim_01",
      "recall_score": 1.0,
      "matched_cand_claim_ids": ["cand_02"],
      "reason": "..."
    }}
  ]
}}
