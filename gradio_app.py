from __future__ import annotations

from pathlib import Path
from datetime import datetime

import gradio as gr
from inference.generate import SCHNELL_MODEL, generate, load_pipeline

DEFAULT_LORA_PATH = Path("./training/output/mystyle_flux_lora.safetensors")
DEFAULT_BASE_MODEL = SCHNELL_MODEL
OUTPUT_DIR = Path("./inference/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipe = None


def ensure_pipeline(lora_path: str, base_model: str):
    global pipe
    path = Path(lora_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LoRA not found: {path}")
    if pipe is None:
        pipe = load_pipeline(path, base_model=base_model)
    return pipe


def run_generation(prompt: str, negative_prompt: str, lora_path: str, trigger_token: str, base_model: str):
    pipeline = ensure_pipeline(lora_path, base_model)
    image = generate(
        pipeline,
        user_prompt=prompt,
        trigger_token=trigger_token,
        negative_prompt=negative_prompt or None,
        base_model=base_model,
    )
    output_path = OUTPUT_DIR / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(output_path)
    return image, str(output_path)


with gr.Blocks(title="Minimal FLUX LoRA Generator") as demo:
    gr.Markdown("## Minimal FLUX.1 + LoRA Image Generator")
    gr.Markdown("Train once in Colab, then generate locally with prompts.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="A fantasy castle at sunrise")
            negative_prompt = gr.Textbox(label="Negative Prompt (optional)", placeholder="blurry, low quality")
            trigger_token = gr.Textbox(label="Trigger Token", value="mystyle")
            base_model = gr.Textbox(label="Base Model", value=DEFAULT_BASE_MODEL)
            lora_path = gr.Textbox(label="LoRA Path", value=str(DEFAULT_LORA_PATH))
            generate_button = gr.Button("Generate", variant="primary")
        with gr.Column():
            image_output = gr.Image(label="Generated Image", type="pil")
            file_output = gr.File(label="Download Image")

    generate_button.click(
        fn=run_generation,
        inputs=[prompt, negative_prompt, lora_path, trigger_token, base_model],
        outputs=[image_output, file_output],
    )

if __name__ == "__main__":
    demo.launch()
