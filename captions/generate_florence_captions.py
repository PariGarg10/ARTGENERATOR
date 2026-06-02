"""
Auto-caption images with Florence-2 and prepend trigger token.

Outputs one .txt caption file beside each image in `--image_dir`.

Usage:
python captions/generate_florence_captions.py --image_dir ./dataset/processed --trigger_token mystyle
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import pipeline

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(image_dir: Path) -> Iterable[Path]:
    for file_path in image_dir.rglob("*"):
        if file_path.suffix.lower() in VALID_EXTENSIONS and file_path.is_file():
            yield file_path


def clean_caption(text: str) -> str:
    text = " ".join(text.strip().split())
    return text.replace("|", ",")


def generate_captions(image_dir: Path, trigger_token: str, model_id: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(device)
    except AttributeError as error:
        raise RuntimeError(
            "Florence-2 failed to load. This is usually a transformers/Python compatibility issue.\n"
            "Fix steps:\n"
            "1) Use Python 3.10 or 3.11 (recommended)\n"
            "2) Install pinned deps from requirements.txt (transformers==4.41.2)\n"
            f"Current Python: {platform.python_version()}\n"
            "Then re-run captioning."
        ) from error
    except ImportError as error:
        missing_flash_attn = "flash_attn" in str(error)
        if not missing_flash_attn:
            raise

        print("Florence-2 requires flash_attn on this environment.")
        print("Falling back to BLIP captioning so you can continue locally.")
        captioner = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=0 if torch.cuda.is_available() else -1,
        )

        count = 0
        for image_path in list_images(image_dir):
            image = Image.open(image_path).convert("RGB")
            result = captioner(image, max_new_tokens=64)
            caption = clean_caption(result[0]["generated_text"])
            final_caption = f"{trigger_token}, {caption}"
            caption_path = image_path.with_suffix(".txt")
            caption_path.write_text(final_caption, encoding="utf-8")
            count += 1

        print(f"Captioned {count} images in {image_dir} (BLIP fallback)")
        return

    task_prompt = "<MORE_DETAILED_CAPTION>"
    count = 0
    for image_path in list_images(image_dir):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, dtype=dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=96,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(generated_text, task=task_prompt, image_size=image.size)
        caption = clean_caption(parsed[task_prompt])
        final_caption = f"{trigger_token}, {caption}"

        caption_path = image_path.with_suffix(".txt")
        caption_path.write_text(final_caption, encoding="utf-8")
        count += 1

    print(f"Captioned {count} images in {image_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--trigger_token", type=str, default="mystyle")
    parser.add_argument("--model_id", type=str, default="microsoft/Florence-2-base")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_captions(args.image_dir, args.trigger_token, args.model_id)
