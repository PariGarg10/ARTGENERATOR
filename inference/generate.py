"""
Inference utility for FLUX.1-dev + trained LoRA.

Usage:
python inference/generate.py \
  --prompt "a cozy cafe street in rain" \
  --lora_path ./training/output/mystyle_flux_lora.safetensors \
  --output ./inference/outputs/sample.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import FluxPipeline


def load_pipeline(lora_path: Path) -> FluxPipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=dtype)
    pipe.load_lora_weights(str(lora_path))
    pipe.to(device)
    return pipe


def generate(
    pipe: FluxPipeline,
    user_prompt: str,
    trigger_token: str = "mystyle",
    negative_prompt: str | None = None,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 30,
    width: int = 1024,
    height: int = 1024,
):
    full_prompt = f"{trigger_token}, {user_prompt}"
    return pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
    ).images[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--lora_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./inference/outputs/generated.png"))
    parser.add_argument("--trigger_token", type=str, default="mystyle")
    parser.add_argument("--negative_prompt", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(args.lora_path)
    image = generate(pipeline, args.prompt, trigger_token=args.trigger_token, negative_prompt=args.negative_prompt)
    image.save(args.output)
    print(f"Saved image: {args.output}")
