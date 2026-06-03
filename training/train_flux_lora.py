"""
Train FLUX LoRA using the official Diffusers dreambooth LoRA script.

Includes two ready-to-run profiles:
- free_colab: FLUX.1-schnell, 384px, rank 2, 4-bit NF4 transformer (12GB RAM / 15GB VRAM Colab)
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

# Must match requirements-colab.txt; training script is downloaded from this tag.
DIFFUSERS_TRAIN_TAG = "v0.34.0"
DIFFUSERS_TRAIN_SCRIPT = (
    f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS_TRAIN_TAG}/"
    "examples/dreambooth/train_dreambooth_lora_flux.py"
)

PROFILE_DEFAULTS = {
    "free_colab": {
        "base_model": FREE_COLAB_MODEL,
        "resolution": 384,
        "batch_size": 1,
        "rank": 2,
        "epochs": 6,
        "learning_rate": 1e-4,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 1,
        "memory_saving": True,
        "low_ram": True,
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


def bundled_training_script() -> Path:
    """Use the script committed in-repo (diffusers v0.34.0 compatible)."""
    return Path(__file__).resolve().parent / "_hf_scripts" / "train_dreambooth_lora_flux.py"


def script_is_compatible(content: str) -> bool:
    return (
        'check_min_version("0.34.0")' in content
        and "_collate_lora_metadata" in content
    )


def write_compatible_script(script_path: Path, content: str) -> None:
    if not script_is_compatible(content):
        raise RuntimeError(
            "Downloaded training script is not compatible with diffusers 0.34.0. "
            "Re-clone the repo or run: git pull"
        )
    script_path.write_text(content, encoding="utf-8")
    (script_path.parent / ".diffusers_train_tag").write_text(DIFFUSERS_TRAIN_TAG, encoding="utf-8")


def ensure_diffusers_script(script_path: Path) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)

    if script_path.exists():
        existing = script_path.read_text(encoding="utf-8")
        if script_is_compatible(existing):
            return

    bundled = bundled_training_script()
    if bundled.exists():
        bundled_text = bundled.read_text(encoding="utf-8")
        if script_is_compatible(bundled_text):
            write_compatible_script(script_path, bundled_text)
            print(f"Using bundled training script: {bundled}")
            return

    script_path.unlink(missing_ok=True)
    subprocess.run(
        ["curl", "-L", DIFFUSERS_TRAIN_SCRIPT, "-o", str(script_path)],
        check=True,
    )
    downloaded = script_path.read_text(encoding="utf-8")
    write_compatible_script(script_path, downloaded)
    print(f"Downloaded training script from {DIFFUSERS_TRAIN_TAG}")


def verify_hf_model_access(model_id: str) -> None:
    """Fail early if FLUX weights cannot be downloaded (login or license missing)."""
    try:
        from huggingface_hub import whoami
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub: pip install huggingface_hub") from exc

    try:
        whoami()
    except Exception as exc:
        raise RuntimeError(
            "Not logged in to Hugging Face. In Colab run the login cell first:\n"
            "  from huggingface_hub import login\n"
            "  login()\n"
            "Also accept the model license on the model page."
        ) from exc

    try:
        from transformers import CLIPTokenizer

        CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        print(f"Hugging Face access OK for: {model_id}")
    except OSError as exc:
        raise RuntimeError(
            f"Cannot download tokenizer for {model_id}.\n"
            "Fix:\n"
            f"  1) Open https://huggingface.co/{model_id} and click 'Agree and access repository'\n"
            "  2) Create a token at https://huggingface.co/settings/tokens (read access)\n"
            "  3) Run: from huggingface_hub import login; login()\n"
            "  4) Re-run training"
        ) from exc


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
        "low_ram": profile.get("low_ram", False),
    }


def run_training(args: argparse.Namespace) -> None:
    script_path = Path("./training/_hf_scripts/train_dreambooth_lora_flux.py")
    ensure_diffusers_script(script_path)

    settings = resolve_profile(args)
    verify_hf_model_access(settings["base_model"])
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

    if settings["low_ram"]:
        command.append("--low_ram")

    print("Training profile:", args.profile)
    if settings["low_ram"]:
        print("Low-RAM mode: 4-bit NF4 transformer (for 12GB system RAM Colab)")
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
