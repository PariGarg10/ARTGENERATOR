"""
Train FLUX.1-dev LoRA using Diffusers training script.

This is a lightweight launcher that calls the official Diffusers FLUX LoRA trainer
with free-Colab-friendly defaults.

Usage example:
python training/train_flux_lora.py \
  --dataset_dir ./dataset/processed \
  --output_dir ./training/output \
  --trigger_token mystyle
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


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


def run_training(args: argparse.Namespace) -> None:
    script_path = Path("./training/_hf_scripts/train_dreambooth_lora_flux.py")
    ensure_diffusers_script(script_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    command = [
        "accelerate",
        "launch",
        str(script_path),
        "--pretrained_model_name_or_path",
        "black-forest-labs/FLUX.1-dev",
        "--instance_data_dir",
        str(args.dataset_dir),
        "--output_dir",
        str(output_dir),
        "--instance_prompt",
        f"{args.trigger_token} style",
        "--resolution",
        str(args.resolution),
        "--train_batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        str(args.learning_rate),
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--max_train_steps",
        str(args.epochs * args.steps_per_epoch),
        "--rank",
        str(args.rank),
        "--mixed_precision",
        "fp16",
        "--seed",
        "42",
    ]

    print("Running command:\n", " ".join(command))
    subprocess.run(command, check=True, env=env)

    # Rename to requested filename for easier downstream use.
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

    # Free Colab defaults
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
