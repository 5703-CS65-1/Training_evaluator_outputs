"""
从 HuggingFace 下载评测图片，按 1.jsonl 中的文件名保存到 package_1_images/
用法: python download_images.py
"""
import json
from pathlib import Path
from datasets import load_dataset

JSONL_PATH = Path("data/eval_text/1.jsonl")
OUTPUT_DIR = Path("data/eval_text/package_1_images")
HF_DATASET  = "A111LEn/5703-test"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取 jsonl，按顺序提取期望的文件名
    image_names = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                image_names.append(json.loads(line)["image"])

    print(f"需要下载 {len(image_names)} 张图片")

    # 加载 HuggingFace 数据集（流式，省内存）
    print("正在连接 HuggingFace 数据集...")
    ds = load_dataset(HF_DATASET, split="test")

    assert len(ds) == len(image_names), \
        f"数据集行数不匹配: HF={len(ds)}, jsonl={len(image_names)}"

    # 按索引保存图片
    skipped = 0
    for i, (row, fname) in enumerate(zip(ds, image_names)):
        out_path = OUTPUT_DIR / fname
        if out_path.exists():
            skipped += 1
            continue
        img = row["image"]  # PIL.Image
        img.save(out_path)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(image_names)} 完成")

    print(f"下载完成！新保存 {len(image_names)-skipped} 张，跳过已存在 {skipped} 张")
    print(f"图片目录: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
