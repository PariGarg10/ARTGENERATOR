# Minimal FLUX.1 + LoRA Generator

This project is a lightweight custom-style image generator:
- train **once** with your dataset using LoRA on FLUX (Colab-friendly),
- then generate unlimited images from prompts in that learned style.

**Free Colab profile (recommended):** `FLUX.1-schnell`, 512px, rank 4, memory-saving flags.
**High-VRAM profile:** `FLUX.1-dev`, 1024px, rank 8 (A100 / 24GB+).

## Folder Structure

```text
project/
├── dataset/
│   └── preprocess_dataset.py
├── captions/
│   └── generate_florence_captions.py
├── training/
│   └── train_flux_lora.py
├── inference/
│   ├── __init__.py
│   └── generate.py
├── gradio_app.py
├── requirements.txt
├── train_flux_lora.ipynb
├── train_flux_lora_high_vram.ipynb
└── README.md
```

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Prepare dataset

Put raw images in `dataset/raw/`, then run:

```bash
python dataset/preprocess_dataset.py --input_dir ./dataset/raw --output_dir ./dataset/processed --size 1024
```

This script:
- removes corrupted/unreadable images,
- center-crops and resizes to `1024x1024`.

## 3) Auto-caption with Florence-2

```bash
python captions/generate_florence_captions.py --image_dir ./dataset/processed --trigger_token mystyle
```

This creates one `.txt` caption per image and prepends the trigger token, for example:

```text
mystyle, a hot air balloon in a bright sky
```

## 4) Train LoRA on Colab (recommended)

| Notebook | When to use |
|----------|-------------|
| `train_flux_lora.ipynb` | Free Colab (~12GB RAM / 15GB VRAM) — schnell, 384px, rank 2, 4-bit |
| `train_flux_lora_high_vram.ipynb` | A100 / 24GB+ VRAM — FLUX.1-dev, 1024px, rank 8 (original quality) |

Open the matching notebook in Google Colab and run all cells.

The notebook uses the **free Colab profile** (12GB RAM / 15GB VRAM):
- Base model: `FLUX.1-schnell`
- Resolution: 384
- Batch size: 1, LoRA rank: 2
- 4-bit NF4 transformer (`--low_ram`)
- Memory flags: gradient checkpointing, fp16, 8-bit Adam, latent caching
- Libraries: install from `requirements-colab.txt` (pinned versions)

For high-VRAM GPUs, use `--profile high_vram` instead.

Expected output:

```text
training/output/mystyle_flux_lora.safetensors
```

## 5) Local inference (script)

```bash
python inference/generate.py \
  --prompt "a cozy cafe in rain" \
  --lora_path ./training/output/mystyle_flux_lora.safetensors \
  --base_model black-forest-labs/FLUX.1-schnell \
  --width 384 --height 384 \
  --output ./inference/outputs/sample.png
```

Generation prompt is automatically prefixed:
- `mystyle + user prompt`

## 6) Gradio app (simple UI)

```bash
python gradio_app.py
```

UI includes:
- Prompt textbox
- Generate button
- Image output
- Download image button

## Notes

- Free Colab (~15 GB VRAM): use `FLUX.1-schnell` and `--profile free_colab` (default).
- FLUX.1-dev often OOMs on free T4; use `--profile high_vram` only on A100/L4 with 24GB+.
- Accept the Hugging Face license and log in before training gated FLUX models.
- If VRAM errors persist, keep resolution at 512 and rank at 4; do not increase batch size.
