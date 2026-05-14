"""快速推理测试：加载 Qwen3-VL-8B-Thinking 跑一张图，验证模型可加载并生成。
Usage:
    python inference_test.py
"""
import time
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_PATH = "/root/autodl-fs/models/Qwen3-VL-8B-Thinking"
TEST_IMAGE = "/root/autodl-fs/code/LLaMA-Factory/data/mllm_demo_data/1.jpg"


def main():
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"Free GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    print("=" * 60)

    print("\n[1/3] 加载 processor...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"     done ({time.time() - t0:.1f}s)")

    print("\n[2/3] 加载模型 (BF16, ~16GB)...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"     done ({time.time() - t0:.1f}s)")
    print(f"     GPU mem after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n[3/3] 推理测试...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": TEST_IMAGE},
                {"type": "text", "text": "用一句话描述这张图片。"},
            ],
        }
    ]
    t0 = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    out = processor.batch_decode(
        gen_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print(f"     done ({time.time() - t0:.1f}s)")
    print(f"\n=== 模型输出 ===\n{out[0]}\n")
    print("=" * 60)
    print("推理测试通过 ✓")


if __name__ == "__main__":
    main()
