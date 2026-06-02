"""
Inference utility for FLUX + trained LoRA.

Usage (free Colab / schnell profile):
python inference/generate.py \
  --prompt "a cozy cafe street in rain" \
  --lora_path ./training/output/mystyle_flux_lora.safetensors \
  --base_model black-forest-labs/FLUX.1-schnell \
  --width 512 --height 512 \
  --output ./inference/outputs/sample.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import FluxPipeline

SCHNELL_MODEL = "black-forest-labs/FLUX.1-schnell"
DEV_MODEL = "black-forest-labs/FLUX.1-dev"


def load_pipeline(lora_path: Path, base_model: str = SCHNELL_MODEL) -> FluxPipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.load_lora_weights(str(lora_path))
    pipe.to(device)
    return pipe


def default_inference_settings(base_model: str) -> dict:
    if "schnell" in base_model.lower():
        return {
            "guidance_scale": 0.0,
            "num_inference_steps": 4,
            "width": 512,
            "height": 512,
        }
    return {
        "guidance_scale": 3.5,
        "num_inference_steps": 30,
        "width": 1024,
        "height": 1024,
    }


def generate(
    pipe: FluxPipeline,
    user_prompt: str,
    trigger_token: str = "mystyle",
    negative_prompt: str | None = None,
    guidance_scale: float | None = None,
    num_inference_steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    base_model: str = SCHNELL_MODEL,
):
    defaults = default_inference_settings(base_model)
    full_prompt = f"{trigger_token}, {user_prompt}"
    return pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale if guidance_scale is not None else defaults["guidance_scale"],
        num_inference_steps=num_inference_steps if num_inference_steps is not None else defaults["num_inference_steps"],
        width=width if width is not None else defaults["width"],
        height=height if height is not None else defaults["height"],
    ).images[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--lora_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./inference/outputs/generated.png"))
    parser.add_argument("--trigger_token", type=str, default="mystyle")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=SCHNELL_MODEL)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(args.lora_path, base_model=args.base_model)
    image = generate(
        pipeline,
        args.prompt,
        trigger_token=args.trigger_token,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        width=args.width,
        height=args.height,
        base_model=args.base_model,
    )
    image.save(args.output)
    print(f"Saved image: {args.output}")
