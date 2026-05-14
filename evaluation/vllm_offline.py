"""vLLM offline batch inference engine for the Stage-A candidate model.

不起 OpenAI server，直接在进程内加载 vLLM `LLM`，用 `LLM.chat()` 一次性
batch 跑完所有 candidate 推理，返回 dict[sample_id -> CandidateOutput]。

为什么走离线 batch:
  - 评测场景：所有 prompt 一次性已知，无需 streaming / 多客户端
  - 单进程 in-process，省掉 server 启停 / 端口占用 / 健康检查
  - vLLM 内部仍然 continuous-batch + paged-attention，吞吐与 server 一致
  - 采样参数、reasoning 后处理一处搞定

修复了之前 server 路径上的退化问题:
  - Qwen3-VL Thinking 官方推荐采样: temperature=0.6, top_p=0.95, top_k=20,
    repetition_penalty>=1.05；之前 server 路径走 temperature=0 导致严重退化
  - chat_template.jinja 已删 `<think>\\n` prefix，与 LLaMA-Factory 训练侧
    `qwen3_vl` 模板对齐
  - prompt 已删字面 `...` 占位符
  - 通过正则后处理把 `<think>...</think>` 拆出来到 reasoning_text，
    candidate_text 同时保留原文，下游 stage B 已经会自动剥 think
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _split_reasoning(text: str) -> tuple[str | None, str]:
    """Return (reasoning, full_text). full_text is left intact for downstream stages.

    Stage B's `_clean_for_claim_extraction` already strips `<think>...</think>` from
    the candidate text, so we keep `full_text` as-is and only surface `reasoning`
    separately for logging / debugging.
    """
    m = _THINK_RE.search(text)
    if not m:
        return None, text
    return m.group(1).strip(), text


def _image_media_type(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower()
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(
        ext, "image/png"
    )


def _encode_image_data_url(image_path: Path) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:{_image_media_type(image_path.name)};base64,{b64}"


class VllmCandidateEngine:
    """Wrapper around vllm.LLM for offline batched candidate generation."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        limit_mm_per_prompt: dict[str, int] | None = None,
        trust_remote_code: bool = True,
        seed: int = 42,
        # Sampling defaults — Qwen3 thinking-model official recommendation
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
        presence_penalty: float = 0.0,
        max_new_tokens: int = 4096,
    ) -> None:
        from vllm import LLM, SamplingParams

        log.info(
            "[vllm-offline] Loading model: %s (dtype=%s, max_len=%d, mem_util=%.2f, tp=%d)",
            model_path, dtype, max_model_len, gpu_memory_utilization, tensor_parallel_size,
        )
        self._llm = LLM(
            model=model_path,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt=limit_mm_per_prompt or {"image": 1},
            seed=seed,
        )
        self._sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_new_tokens,
            seed=seed,
            # Stop only at the canonical chat-template end-of-turn token; do NOT
            # include `</think>` here — the model is supposed to emit it.
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        log.info(
            "[vllm-offline] Sampling: temp=%.2f top_p=%.2f top_k=%d rep_pen=%.2f max_tokens=%d",
            temperature, top_p, top_k, repetition_penalty, max_new_tokens,
        )

    def _build_messages(self, image_path: Path, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_image_data_url(image_path)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def run_batch(
        self,
        items: list[tuple[str, Path]],
        prompt: str,
    ) -> dict[str, tuple[str, str | None]]:
        """Run vLLM on a batch of (sample_id, image_path) tuples.

        Returns dict[sample_id -> (full_text, reasoning_or_None)].
        Order of inputs is preserved by vLLM but we re-key by sample_id for safety.
        """
        if not items:
            return {}

        message_lists = [self._build_messages(p, prompt) for _sid, p in items]

        log.info("[vllm-offline] Submitting batch of %d prompts to vLLM …", len(items))
        outputs = self._llm.chat(message_lists, sampling_params=self._sampling)

        if len(outputs) != len(items):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(items)} prompts — order mismatch"
            )

        results: dict[str, tuple[str, str | None]] = {}
        for (sid, _path), out in zip(items, outputs, strict=True):
            text = out.outputs[0].text if out.outputs else ""
            reasoning, full = _split_reasoning(text)
            results[sid] = (full, reasoning)
        log.info("[vllm-offline] Batch done; %d candidate texts ready.", len(results))
        return results
