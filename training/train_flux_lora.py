"""
Train FLUX LoRA using the official Diffusers dreambooth LoRA script.

Includes two ready-to-run profiles:
- free_colab: FLUX.1-schnell, 512px, rank 4, memory-saving flags (T4/L4 ~15GB VRAM)
- high_vram:  FLUX.1-dev, 1024px, rank 8 (A100 / 24GB+ VRAM)

Usage (free Colab):
python training/train_flux_lora.py \
  --dataset_dir ./dataset/processed \
  --output_dir ./training/output \
  --trigger_token mystyle \
  --profile free_colab
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


FREE_COLAB_MODEL = "black-forest-labs/FLUX.1-schnell"
HIGH_VRAM_MODEL = "black-forest-labs/FLUX.1-dev"

PROFILE_DEFAULTS = {
    "free_colab": {
        "base_model": FREE_COLAB_MODEL,
        "resolution": 512,
        "batch_size": 1,
        "rank": 4,
        "epochs": 10,
        "learning_rate": 1e-4,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 1,
        "memory_saving": True,
    },
    "high_vram": {
        "base_model": HIGH_VRAM_MODEL,
        "resolution": 1024,
        "batch_size": 1,
        "rank": 8,
        "epochs": 10,
        "learning_rate": 1e-4,
        "mixed_precision": "bf16",
        "gradient_accumulation_steps": 4,
        "memory_saving": True,
    },
}


def ensure_diffusers_script(script_path: Path) -> None:
    if script_path.exists():
        return
    script_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux.py",
            "-o",
            str(script_path),
        ],
        check=True,
    )


def count_training_images(dataset_dir: Path) -> int:
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sum(1 for path in dataset_dir.iterdir() if path.suffix.lower() in extensions)


def resolve_profile(args: argparse.Namespace) -> dict:
    profile = PROFILE_DEFAULTS[args.profile]
    return {
        "base_model": args.base_model or profile["base_model"],
        "resolution": args.resolution or profile["resolution"],
        "batch_size": args.batch_size or profile["batch_size"],
        "rank": args.rank or profile["rank"],
        "epochs": args.epochs or profile["epochs"],
        "learning_rate": args.learning_rate or profile["learning_rate"],
        "mixed_precision": args.mixed_precision or profile["mixed_precision"],
        "gradient_accumulation_steps": args.gradient_accumulation_steps or profile["gradient_accumulation_steps"],
        "memory_saving": profile["memory_saving"] and not args.no_memory_saving,
    }


def run_training(args: argparse.Namespace) -> None:
    script_path = Path("./training/_hf_scripts/train_dreambooth_lora_flux.py")
    ensure_diffusers_script(script_path)

    settings = resolve_profile(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_count = max(count_training_images(Path(args.dataset_dir)), 1)
    steps_per_epoch = args.steps_per_epoch or image_count
    max_train_steps = args.max_train_steps or (settings["epochs"] * steps_per_epoch)

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    command = [
        "accelerate",
        "launch",
        "--mixed_precision",
        settings["mixed_precision"],
        str(script_path),
        "--pretrained_model_name_or_path",
        settings["base_model"],
        "--instance_data_dir",
        str(args.dataset_dir),
        "--output_dir",
        str(output_dir),
        "--instance_prompt",
        f"{args.trigger_token} style",
        "--resolution",
        str(settings["resolution"]),
        "--train_batch_size",
        str(settings["batch_size"]),
        "--gradient_accumulation_steps",
        str(settings["gradient_accumulation_steps"]),
        "--learning_rate",
        str(settings["learning_rate"]),
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--max_train_steps",
        str(max_train_steps),
        "--rank",
        str(settings["rank"]),
        "--mixed_precision",
        settings["mixed_precision"],
        "--guidance_scale",
        "1",
        "--seed",
        "42",
    ]

    if settings["memory_saving"]:
        command.extend(
            [
                "--gradient_checkpointing",
                "--use_8bit_adam",
                "--cache_latents",
            ]
        )

    print("Training profile:", args.profile)
    print("Base model:", settings["base_model"])
    print("Resolution:", settings["resolution"])
    print("Rank:", settings["rank"])
    print("Max train steps:", max_train_steps)
    print("Running command:\n", " ".join(command))
    subprocess.run(command, check=True, env=env)

    target = output_dir / "mystyle_flux_lora.safetensors"
    candidate = output_dir / "pytorch_lora_weights.safetensors"
    if candidate.exists():
        candidate.replace(target)
        print(f"Saved LoRA weights to: {target}")
    else:
        print("Training finished. Could not find default LoRA file to rename automatically.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("./training/output"))
    parser.add_argument("--trigger_token", type=str, default="mystyle")
    parser.add_argument(
        "--profile",
        type=str,
        choices=("free_colab", "high_vram"),
        default="free_colab",
        help="free_colab = FLUX.1-schnell @ 512px; high_vram = FLUX.1-dev @ 1024px",
    )

    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument(
        "--no_memory_saving",
        action="store_true",
        help="Disable gradient checkpointing, 8-bit Adam, and latent caching.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
