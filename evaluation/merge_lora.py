"""Merge a PEFT LoRA adapter into its base model and persist the merged weights.

Why merge ahead of time:
    • vLLM's runtime LoRA loading for vision-language models is uneven; the most
      reliable path is to serve a fully-merged checkpoint.
    • You only pay the merge cost once, then every subsequent vLLM startup goes
      straight from disk → GPU.

Usage (defaults match the layout in this project):

    /root/autodl-fs/conda_envs/llama_factory/bin/python merge_lora.py \
        --base  /root/autodl-fs/models/Qwen3-VL-8B-Thinking \
        --lora  /root/autodl-fs/output/qwen3vl-8b-aesthetic-lora \
        --out   /root/autodl-fs/models/qwen3vl-8b-aesthetic-merged

Notes:
    • Run this in the env with the **newer** transformers/peft (llama_factory env
      has transformers 5.2 + peft 0.18, which matches the LoRA training env).
    • Output dir will contain the model weights + processor/tokenizer files,
      so vLLM can be pointed at it directly.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
log = logging.getLogger("merge_lora")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    p.add_argument("--base", required=True, type=Path,
                   help="Base model dir (e.g. /root/autodl-fs/models/Qwen3-VL-8B-Thinking)")
    p.add_argument("--lora", required=True, type=Path,
                   help="LoRA adapter dir (peft adapter_config.json + adapter_model.safetensors)")
    p.add_argument("--out",  required=True, type=Path,
                   help="Output dir for the merged model (will be created)")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device-map", default="cpu",
                   help="device_map for from_pretrained. 'cpu' is safest (no GPU OOM risk).")
    p.add_argument("--overwrite", action="store_true",
                   help="If --out already exists, remove it first.")
    args = p.parse_args()

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"Output dir {args.out} already exists. Use --overwrite to replace.")
        log.warning("Removing existing %s", args.out)
        shutil.rmtree(args.out)

    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelCls  # type: ignore

    from peft import PeftModel

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[args.dtype]

    t0 = time.time()
    log.info("Loading base model from %s (dtype=%s, device_map=%s) …",
             args.base, args.dtype, args.device_map)
    base = ModelCls.from_pretrained(
        args.base,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    log.info("  base loaded in %.1fs", time.time() - t0)

    t0 = time.time()
    log.info("Attaching LoRA adapter from %s …", args.lora)
    peft_model = PeftModel.from_pretrained(base, args.lora)
    log.info("  adapter attached in %.1fs", time.time() - t0)

    t0 = time.time()
    log.info("Merging LoRA into base weights (merge_and_unload) …")
    merged = peft_model.merge_and_unload()
    log.info("  merged in %.1fs", time.time() - t0)

    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info("Saving merged model → %s", args.out)
    merged.save_pretrained(args.out, safe_serialization=True)
    log.info("  weights saved in %.1fs", time.time() - t0)

    # Persist the processor/tokenizer so vLLM finds preprocessor_config.json etc.
    log.info("Saving processor / tokenizer from base …")
    proc = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    proc.save_pretrained(args.out)

    # chat_template.jinja from the LoRA dir takes precedence if present (training-time template)
    src_chat = args.lora / "chat_template.jinja"
    if src_chat.is_file():
        shutil.copy(src_chat, args.out / "chat_template.jinja")
        log.info("Copied chat_template.jinja from LoRA dir")

    log.info("✓ Done. Merged model directory: %s", args.out)


if __name__ == "__main__":
    main()
