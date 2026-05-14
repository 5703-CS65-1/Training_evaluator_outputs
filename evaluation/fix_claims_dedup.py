"""
fix_claims_dedup.py
====================
对 candidate_claims.jsonl 做维度去重：
  - 每个维度只保留最后一次出现的 claim（对应模型最终给出的答案）
  - 对有 </think> 的样本：直接从 </think> 之后重新提取，最干净
  - 对没有 </think> 的样本：在已有 claims 里按维度保留最后一条

用法:
  python fix_claims_dedup.py --output-dir outputs/base-all1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

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
_HEADER_RE = re.compile(
    r"^[\s*#>\-]*\d{0,2}\.?\s*(?P<dim>" + _DIMENSION_PATTERN + r")\s*\**\s*(?:[:：]|$)",
    re.MULTILINE | re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CLOSE_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)
_SCORE_LINE_RE = re.compile(
    r"\n?\s*Overall\s+aesthetic\s+score\s*[:：]\s*[0-9]+(?:\.[0-9]+)?\s*/\s*10\s*",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _SCORE_LINE_RE.sub("\n", text)
    return text.strip()


def extract_claims_from_text(text: str) -> list[dict]:
    """从 candidate_text 中提取，优先只取 </think> 之后部分。"""
    # 优先截断到 </think> 之后
    if "</think>" in text:
        after = _CLOSE_THINK_RE.sub("", text, count=1)
        text_to_use = after.strip()
    else:
        text_to_use = text

    text_to_use = _clean(text_to_use)
    matches = list(_HEADER_RE.finditer(text_to_use))

    if not matches:
        # fallback：整段当一个 claim
        t = text_to_use.strip()
        if t:
            return [{"cand_claim_id": "cand_01", "text": t}]
        return []

    claims = []
    for i, m in enumerate(matches):
        dim_name = m.group("dim")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text_to_use)
        content = text_to_use[start:end].strip()
        if content:
            claims.append({"dim": dim_name, "text": f"[{dim_name}] {content}"})

    # 每个维度只保留最后一次（去重）
    last_by_dim: dict[str, dict] = {}
    for c in claims:
        last_by_dim[c["dim"].lower()] = c

    # 按 DIMENSIONS 顺序排列，重新编号
    ordered = []
    for dim in DIMENSIONS:
        key = dim.lower()
        if key in last_by_dim:
            ordered.append(last_by_dim[key]["text"])

    return [
        {"cand_claim_id": f"cand_{i+1:02d}", "text": t}
        for i, t in enumerate(ordered)
    ]


def fix_output_dir(output_dir: Path) -> None:
    texts_path  = output_dir / "candidate_texts.jsonl"
    claims_path = output_dir / "candidate_claims.jsonl"

    if not texts_path.exists():
        print(f"[ERROR] 找不到 {texts_path}", file=sys.stderr)
        sys.exit(1)

    # 读取所有 candidate texts
    texts: dict[str, str] = {}
    with texts_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts[obj["sample_id"]] = obj["candidate_text"]

    print(f"读取 candidate_texts: {len(texts)} 条")

    # 重新提取 claims
    fixed: list[dict] = []
    count_before: list[int] = []
    count_after:  list[int] = []

    # 先统计旧 claims 数量（如果存在）
    old_counts: dict[str, int] = {}
    if claims_path.exists():
        with claims_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                old_counts[obj["sample_id"]] = len(obj.get("candidate_claims", []))

    for sid, text in texts.items():
        old_n = old_counts.get(sid, -1)
        new_claims = extract_claims_from_text(text)
        count_before.append(old_n)
        count_after.append(len(new_claims))
        fixed.append({"sample_id": sid, "candidate_claims": new_claims})

    # 统计
    changed = sum(1 for a, b in zip(count_before, count_after) if a != b)
    print(f"去重前 claims 均值: {sum(count_before)/len(count_before):.1f}  "
          f"max={max(count_before)}  min={min(count_before)}")
    print(f"去重后 claims 均值: {sum(count_after)/len(count_after):.1f}  "
          f"max={max(count_after)}  min={min(count_after)}")
    print(f"发生变化的样本: {changed} / {len(fixed)}")

    # 备份旧文件
    if claims_path.exists():
        bak = claims_path.with_suffix(".jsonl.bak")
        claims_path.rename(bak)
        print(f"旧文件已备份至 {bak}")

    # 写入新文件
    with claims_path.open("w", encoding="utf-8") as f:
        for row in fixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✓ 已写入修复后的 {claims_path}")
    print()
    print("下一步：删除旧 Stage C/D 产物，然后重跑评测脚本")
    print(f"  rm {output_dir}/judged_results.jsonl "
          f"{output_dir}/sample_metrics.jsonl "
          f"{output_dir}/score_metrics.json")
    print(f"  # 然后运行 bash run_vllm_offline.sh（需确认 OUTPUT_DIR=outputs/base-all1 且 RESUME=true）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="评测输出目录，如 outputs/base-all1")
    args = parser.parse_args()
    fix_output_dir(Path(args.output_dir))


if __name__ == "__main__":
    main()
