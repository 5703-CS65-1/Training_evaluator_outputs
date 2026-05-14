"""ArtiMuse offline batch inference engine for the Stage-A candidate model.

ArtiMuse 是基于 InternVL-3 的自定义美学模型（custom modeling_artimuse.py），
不能用 vLLM 加载。它对外暴露两条 API：

  - model.score(...)  → 直接出一个浮点美学分（专门的 aes-token score head，
                         返回 0-100 区间，本文件把它归一到 0-10 与 bench 对齐）
  - model.chat(...)   → 对单个美学维度生成文字评价

为了和现有 pipeline 的 Stage B/C/D 完全兼容（按 10 个 bench heading 切 claims、
regex 抓 `Overall aesthetic score: X.XX/10`），本引擎做的事：

  1. 对每张图加载一次 pixel_values（448×448 单 tile，ImageNet norm，bf16）
  2. 用 batch_chat 一次性对该图跑 bench 的 10 个维度 prompt，拿到 10 段文字
  3. 调用 model.score 拿一个 0-100 的浮点分
  4. 把 10 段按 bench 模板拼成 candidate_text，并在末尾追加
        Overall aesthetic score: X.XX/10
  5. 返回 dict[sample_id -> (candidate_text, reasoning_or_None)]

接口和 VllmCandidateEngine.run_batch 完全对齐，方便 run_artimuse_offline.py
直接 drop-in。

Note: 推理是 GPU-bound 串行（一次一张图，10 个维度走 batch_chat 一次拿完），
没必要再外面起多 worker，单进程内部已经 GPU 满载。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

log = logging.getLogger(__name__)


# Bench 强制的 10 个维度 heading（顺序与 prompts/stage_a_candidate.md 完全一致）。
BENCH_DIMENSIONS = [
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _load_image_tensor(image_path: Path, dtype: torch.dtype, device: str, input_size: int = 448) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    px = _build_transform(input_size)(img).unsqueeze(0)  # [1,3,H,W]
    return px.to(dtype).to(device)


def _ensure_artimuse_on_path(artimuse_repo: Path) -> None:
    """Make `from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel` work.

    ArtiMuse README 要求 `sys.path.append("src")` + `sys.path.append("src/artimuse")`，
    我们这里照搬，避免污染调用方。
    """
    src = artimuse_repo / "src"
    if not src.exists():
        raise FileNotFoundError(
            f"ArtiMuse repo 'src/' not found under {artimuse_repo}. "
            "请确认 --artimuse-repo 指向 ArtiMuse 仓库根目录。"
        )
    for p in (src, src / "artimuse"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


class ArtiMuseCandidateEngine:
    """Wrapper around ArtiMuse's InternVLChatModel for batched-per-image inference.

    Mirrors vllm_offline.VllmCandidateEngine's run_batch signature so the rest of
    the pipeline doesn't care which backend produced the candidate text.
    """

    def __init__(
        self,
        model_path: str,
        artimuse_repo: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 4096,
        score_max_new_tokens: int = 16,
        use_flash_attn: bool = True,
        chat_mode: str = "batch",
        image_batch_size: int = 4,
    ) -> None:
        _ensure_artimuse_on_path(Path(artimuse_repo))
        from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel  # noqa: E402

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        log.info("[artimuse] Loading model: %s (dtype=%s, device=%s, flash_attn=%s)",
                 model_path, dtype, device, use_flash_attn)
        self._model = InternVLChatModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
        ).eval().to(device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False,
        )

        # Aesthetic-aware generation: greedy 是 ArtiMuse README 的官方做法；
        # max_new_tokens 给大一些以容纳每个维度的详细分析。
        self._gen_cfg = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        # score 输出只是两个字母（如 'ea'），16 个 token 绰绰有余。
        self._score_gen_cfg = dict(
            max_new_tokens=score_max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        self._device = device
        self._dtype = torch_dtype
        if chat_mode not in {"batch", "single"}:
            raise ValueError(f"chat_mode must be 'batch' or 'single', got {chat_mode!r}")
        self._chat_mode = chat_mode
        self._image_batch_size = image_batch_size
        log.info("[artimuse] Ready. chat_mode=%s  max_new_tokens=%d  image_batch_size=%d",
                 chat_mode, max_new_tokens, image_batch_size)

    # ────────────────────────────────────────────────────────────────────
    # Per-image inference
    # ────────────────────────────────────────────────────────────────────
    def _eval_one(self, image_path: Path) -> tuple[str, float | None]:
        """Run 10-dim chat + score on a single image. Returns (candidate_text, score_0_10)."""
        pixel_values = _load_image_tensor(image_path, self._dtype, self._device)

        prompts = [
            f"Please evaluate the aesthetic quality of this image from the aspect of {dim}."
            for dim in BENCH_DIMENSIONS
        ]

        if self._chat_mode == "batch":
            n = len(BENCH_DIMENSIONS)
            pv_batch = torch.cat((pixel_values,) * n, dim=0)
            num_patches_list = [pixel_values.size(0)] * n
            responses = self._model.batch_chat(
                self._device,
                self._tokenizer,
                pv_batch,
                num_patches_list=num_patches_list,
                questions=prompts,
                generation_config=dict(self._gen_cfg),
            )
        else:
            responses = [
                self._model.chat(
                    self._device, self._tokenizer, pixel_values, q, dict(self._gen_cfg),
                )
                for q in prompts
            ]

        # Aesthetic score (0-100) → 0-10 to match bench.
        try:
            raw_score = self._model.score(
                self._device, self._tokenizer, pixel_values, dict(self._score_gen_cfg),
            )
            score_0_10: float | None = float(raw_score) / 10.0
            # 保险 clamp，避免极端 logit 让分数越界。
            if score_0_10 < 0.0:
                score_0_10 = 0.0
            elif score_0_10 > 10.0:
                score_0_10 = 10.0
        except Exception:
            log.exception("[artimuse] score() failed for %s", image_path)
            score_0_10 = None

        # Assemble bench-formatted candidate text.
        paragraphs = []
        for dim, resp in zip(BENCH_DIMENSIONS, responses):
            body = (resp or "").strip()
            # 如果模型自己重复了 heading（比如以 "Layout and Composition:" 开头），
            # 去掉它，避免 Stage B 重复 heading 干扰。
            lowered_head = f"{dim}:".lower()
            if body.lower().startswith(lowered_head):
                body = body[len(lowered_head):].lstrip()
            paragraphs.append(f"{dim}: {body}" if body else f"{dim}: (no response)")

        text = "\n\n".join(paragraphs)
        if score_0_10 is not None:
            text += f"\n\nOverall aesthetic score: {score_0_10:.2f}/10"

        return text, score_0_10

    # ────────────────────────────────────────────────────────────────────
    # Multi-image batch inference (K images × 10 dims in one batch_chat)
    # ────────────────────────────────────────────────────────────────────
    def _eval_multi(self, image_paths: list[Path]) -> list[tuple[str, float | None]]:
        """Run 10-dim chat + score on K images in a single batch_chat call.

        Arranges the batch as: [img0×dim0, img0×dim1, …, img0×dim9,
                                 img1×dim0, …, img1×dim9, …, imgK×dim9]
        so responses[k*10:(k+1)*10] map back to image k.
        score() is still called per-image (max_new_tokens=16, negligible).
        """
        prompts = [
            f"Please evaluate the aesthetic quality of this image from the aspect of {dim}."
            for dim in BENCH_DIMENSIONS
        ]
        n_dims = len(BENCH_DIMENSIONS)

        pv_list = [_load_image_tensor(p, self._dtype, self._device) for p in image_paths]

        pv_parts: list[torch.Tensor] = []
        num_patches_list: list[int] = []
        questions: list[str] = []
        for pv in pv_list:
            for q in prompts:
                pv_parts.append(pv)
                num_patches_list.append(pv.size(0))
                questions.append(q)

        pv_batch = torch.cat(pv_parts, dim=0)  # [K*10, 3, H, W]
        responses = self._model.batch_chat(
            self._device,
            self._tokenizer,
            pv_batch,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=dict(self._gen_cfg),
        )

        results: list[tuple[str, float | None]] = []
        for k, (image_path, pv) in enumerate(zip(image_paths, pv_list)):
            img_responses = responses[k * n_dims : (k + 1) * n_dims]

            try:
                raw_score = self._model.score(
                    self._device, self._tokenizer, pv, dict(self._score_gen_cfg),
                )
                score_0_10: float | None = float(raw_score) / 10.0
                score_0_10 = max(0.0, min(10.0, score_0_10))
            except Exception:
                log.exception("[artimuse] score() failed for %s", image_path)
                score_0_10 = None

            paragraphs = []
            for dim, resp in zip(BENCH_DIMENSIONS, img_responses):
                body = (resp or "").strip()
                lowered_head = f"{dim}:".lower()
                if body.lower().startswith(lowered_head):
                    body = body[len(lowered_head):].lstrip()
                paragraphs.append(f"{dim}: {body}" if body else f"{dim}: (no response)")

            text = "\n\n".join(paragraphs)
            if score_0_10 is not None:
                text += f"\n\nOverall aesthetic score: {score_0_10:.2f}/10"

            results.append((text, score_0_10))

        return results

    # ────────────────────────────────────────────────────────────────────
    # Batched API (mirrors VllmCandidateEngine.run_batch)
    # ────────────────────────────────────────────────────────────────────
    def run_batch(
        self,
        items: list[tuple[str, Path]],
        prompt: str,  # unused — kept for signature compatibility
    ) -> dict[str, tuple[str, str | None]]:
        """Run ArtiMuse on a list of (sample_id, image_path).

        Images are processed in chunks of image_batch_size.  Each chunk issues
        one batch_chat call with batch = chunk_size × 10 dims, which keeps the
        GPU busier than the original one-image-at-a-time loop.
        """
        if not items:
            return {}
        del prompt  # explicit: bench prompt is N/A for ArtiMuse

        results: dict[str, tuple[str, str | None]] = {}
        total = len(items)
        bs = self._image_batch_size

        for chunk_start in range(0, total, bs):
            chunk = items[chunk_start : chunk_start + bs]
            chunk_ids   = [sid  for sid,  _ in chunk]
            chunk_paths = [path for _,  path in chunk]
            log.info(
                "[artimuse] images %d-%d / %d  (batch_chat batch=%d)",
                chunk_start + 1, chunk_start + len(chunk), total, len(chunk) * 10,
            )
            try:
                outputs = self._eval_multi(chunk_paths)
            except Exception:
                log.exception(
                    "[artimuse] batch inference failed for images %d-%d",
                    chunk_start + 1, chunk_start + len(chunk),
                )
                continue
            for sid, (text, _score) in zip(chunk_ids, outputs):
                results[sid] = (text, None)

        log.info("[artimuse] Batch done; %d/%d candidate texts ready.", len(results), total)
        return results
