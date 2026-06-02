"""
Dataset cleaning + resize script.

What it does:
1) Scans an input folder for images.
2) Drops corrupted files.
3) Center-crops and resizes to 1024x1024.
4) Saves cleaned images to output folder.

Usage:
python dataset/preprocess_dataset.py --input_dir ./dataset/raw --output_dir ./dataset/processed --size 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(input_dir: Path) -> Iterable[Path]:
    for file_path in input_dir.rglob("*"):
        if file_path.suffix.lower() in VALID_EXTENSIONS and file_path.is_file():
            yield file_path


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((size, size), Image.Resampling.LANCZOS)


def preprocess(input_dir: Path, output_dir: Path, size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    removed = 0

    for image_path in list_images(input_dir):
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = center_crop_resize(image, size)
                output_path = output_dir / f"{image_path.stem}.png"
                image.save(output_path, format="PNG", optimize=True)
                processed += 1
        except (UnidentifiedImageError, OSError, ValueError):
            removed += 1

    print(f"Done. Processed: {processed} | Removed/Skipped corrupted: {removed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.input_dir, args.output_dir, args.size)
