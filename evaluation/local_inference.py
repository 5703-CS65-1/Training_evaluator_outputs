"""Local HuggingFace inference engine for the Stage-A candidate model.

Loads a Qwen3-VL (or compatible) base model + optional LoRA adapter once at
process start, then exposes an async-friendly `generate_local()` helper that the
existing pipeline can call instead of an OpenAI-compatible HTTP API.

The async wrapper offloads the (blocking) `model.generate` call to a worker
thread via `asyncio.to_thread`; an inner `threading.Lock` serializes GPU work,
so candidate inference happens one sample at a time even if the caller schedules
multiple coroutines concurrently.  This keeps GPU memory pressure bounded while
still letting Stage-C judge requests fire in parallel against the API.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Module-level singletons populated by `init_local_model()`.
_MODEL: Any = None
_PROCESSOR: Any = None
_GEN_LOCK = threading.Lock()
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def init_local_model(
    model_path: str,
    lora_path: str | None = None,
    *,
    device: str = "auto",
    dtype: str = "bfloat16",
    merge_lora: bool = True,
) -> None:
    """Load base model (and merge LoRA if provided).  Idempotent."""
    global _MODEL, _PROCESSOR, _INITIALIZED

    with _INIT_LOCK:
        if _INITIALIZED:
            return

        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as _ModelCls
        except ImportError:  # older transformers fallback
            from transformers import AutoModelForCausalLM as _ModelCls  # type: ignore

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype, torch.bfloat16)

        log.info(
            "[local] Loading base model: %s (dtype=%s, device_map=%s)",
            model_path, dtype, device,
        )
        model = _ModelCls.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )

        if lora_path:
            from peft import PeftModel

            log.info("[local] Loading LoRA adapter: %s", lora_path)
            model = PeftModel.from_pretrained(model, lora_path)
            if merge_lora:
                log.info("[local] Merging LoRA weights into base model")
                model = model.merge_and_unload()

        model.eval()

        log.info("[local] Loading processor: %s", model_path)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        _MODEL = model
        _PROCESSOR = processor
        _INITIALIZED = True
        log.info("[local] Model ready.")


def _strip_image_placeholders(text: str) -> str:
    """Remove SFT-style image placeholder tokens from prompt text.

    The training data contains a literal `<image>` (LLaMA-Factory convention) at
    the end of the user message; for HF inference the image is supplied via the
    chat template `{type: image}` content, so any in-text placeholder must be
    removed to avoid double-insertion.
    """
    for tok in ("<|image|>", "<image>", "<|vision_start|><|image_pad|><|vision_end|>"):
        text = text.replace(tok, "")
    return text.strip()


def _generate_sync(
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    assert _INITIALIZED and _MODEL is not None and _PROCESSOR is not None

    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    user_text = _strip_image_placeholders(prompt)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    # Render text part of the chat template, then run the processor with the
    # image side-by-side.  This is the documented Qwen-VL inference pattern and
    # is robust across recent transformers versions.
    chat_text = _PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _PROCESSOR(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(_MODEL.device)

    with _GEN_LOCK:  # serialize GPU usage across concurrent coroutines
        with torch.inference_mode():
            generated = _MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                pad_token_id=getattr(_PROCESSOR.tokenizer, "pad_token_id", None)
                or getattr(_PROCESSOR.tokenizer, "eos_token_id", None),
            )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, input_len:]
    text = _PROCESSOR.batch_decode(new_tokens, skip_special_tokens=False)[0]

    # Strip Qwen-style end markers but keep `<think>...</think>` content.
    for tok in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(tok, "")
    return text.strip()


async def generate_local(
    image_path: Path,
    prompt: str,
    *,
    max_new_tokens: int = 4096,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """Async-compatible wrapper around the blocking HF generate call."""
    import asyncio

    return await asyncio.to_thread(
        _generate_sync,
        image_path,
        prompt,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
    )


def is_initialized() -> bool:
    return _INITIALIZED
