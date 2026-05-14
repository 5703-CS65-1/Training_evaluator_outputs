Extract the candidate commentary into a compact set of evaluative claims suitable for faithfulness assessment.

Goal:
Represent the commentary as a set of independently judgeable claims while preserving its substantive evaluative content. The objective is not to break the text into the smallest possible fragments, but to capture the commentary in units that are appropriate for meaningful comparison with the image-grounded gold reference.

Context:
The candidate commentary may be organized around ten aesthetic aspects, such as Layout and Composition, Space and Perspective, Light and Shadow, Color, Details and Texture, Theme and Logic, Mood, The Overall, Creativity, and Sense of Order. Treat such aspect headings as discourse structure cues, not as rigid output slots. Do not force one claim per aspect, and do not invent content to fill missing aspects.
Use aspect boundaries as organizational cues for claim extraction. In many cases, multiple sentences within the same aspect section belong to one judgment cluster: a main evaluative judgment together with its supporting observations, examples, or consequences. Do not automatically split such material sentence by sentence. Instead, identify whether the section expresses one integrated judgment or multiple independently judgeable judgments.

Extraction principles:
1. Each claim should be a self-contained proposition expressing one meaningful evaluative judgment.
2. Prefer medium-grained claims that are suitable for independent scoring, rather than minimal atomic fragments.
3. Split only when different parts would reasonably require different faithfulness judgments.
4. Keep statements together when they share the same visual basis and function together as one integrated judgment.
5. If a perceptual description and its immediate interpretation jointly form one coherent evaluative statement, keep them together unless they should clearly be judged separately.
6. When a section contains one main judgment supported by subordinate details, examples, or consequences, prefer extracting the main judgment together with its tightly related support as a single claim.
7. Do not split rhetorical consequence chains (for example: X creates Y, which makes Z feel...) into multiple claims if they together express one unified aesthetic judgment.
8. Do not split simple coordinated mood terms, qualitative modifiers, or near-synonymous consequences into separate claims unless they add meaningfully distinct evaluative commitments.
9. Bullet points, foreground/middle/background breakdowns, and other listed sub-parts should usually be treated as supporting details for a higher-level claim rather than as separate claims on their own.
10. Merge overlapping, repetitive, or near-paraphrastic statements when they do not add a meaningfully new judgment.
11. Preserve explicit interpretations, inferences, and mood attributions when they are clearly asserted in the commentary.
12. Do NOT invent, weaken, strengthen, or normalize the original meaning.
13. Ignore discourse fillers and section labels, but preserve epistemic markers such as seems, suggests, may, perhaps, and feels like when they affect the strength of the claim.
14. Rewrite each claim as a concise standalone proposition while staying faithful to the original scope and level of commitment.
15. Aim for a compact set of claims. For a full multi-paragraph commentary, a compact set is typically around 10-16 claims, but use fewer or more if the actual content clearly warrants it.
16. Do NOT score, classify, or assess the claims. Extraction only.
17. Return valid JSON only, with no markdown fencing.

Return schema:
{{
  "candidate_claims": [
    {{
      "cand_claim_id": "cand_01",
      "text": "..."
    }}
  ]
}}

Candidate commentary:
{candidate_text}
