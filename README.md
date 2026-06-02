# Minimal FLUX.1 + LoRA Generator

This project is a lightweight custom-style image generator:
- train **once** with your dataset using LoRA on FLUX.1-dev (Colab-friendly),
- then generate unlimited images from prompts in that learned style.

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

Open `train_flux_lora.ipynb` in Google Colab and run all cells.

Default training settings (free Colab optimized):
- Resolution: 1024
- Batch size: 1
- LoRA rank: 8
- Epochs: 10
- Learning rate: 1e-4

Expected output:

```text
training/output/mystyle_flux_lora.safetensors
```

## 5) Local inference (script)

```bash
python inference/generate.py --prompt "a cozy cafe in rain" --lora_path ./training/output/mystyle_flux_lora.safetensors --output ./inference/outputs/sample.png
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

- FLUX.1-dev is heavy; free Colab VRAM can be tight. Keep dataset small and clean.
- If VRAM errors happen, reduce effective training steps and keep rank/batch unchanged.
