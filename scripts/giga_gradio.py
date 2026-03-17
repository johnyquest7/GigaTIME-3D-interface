"""
GigaTIME Gradio Interface
=========================
Upload an H&E pathology slide image and generate virtual mIF protein channel maps
using Microsoft's GigaTIME model (NestedUNet architecture).

Requirements: Install the gigatime conda environment from the repo, then:
  pip install gradio

Usage:
  1. Place this file in the GigaTIME/scripts/ directory (alongside archs.py)
  2. Set your HuggingFace token: export HF_TOKEN=<your_token>
  3. Run: python gigatime_gradio_app.py
  4. Open the URL shown in terminal
"""

import os
import sys
import numpy as np
import torch
import gradio as gr
from PIL import Image
from huggingface_hub import snapshot_download

# Ensure scripts dir is on path so we can import archs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import archs

# ── Constants ──────────────────────────────────────────────────────────────────

CHANNEL_NAMES = [
    'DAPI',
    'TRITC',        # background - excluded from display
    'Cy5',          # background - excluded from display
    'PD-1',
    'CD14',
    'CD4',
    'T-bet',
    'CD34',
    'CD68',
    'CD16',
    'CD11c',
    'CD138',
    'CD20',
    'CD3',
    'CD8',
    'PD-L1',
    'CK',
    'Ki67',
    'Tryptase',
    'Actin-D',
    'Caspase3-D',
    'PHH3-B',
    'Transgelin',
]

EXCLUDE_CHANNELS = {'TRITC', 'Cy5'}

# ImageNet normalization (same as the notebook)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

NUM_CLASSES = 23
INPUT_CHANNELS = 3
INPUT_SIZE = 512        # model expects 512x512 (resized), inference in 256 windows
WINDOW_SIZE = 256
THRESHOLD = 0.5

# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model():
    """Download weights from HuggingFace and load the GigaTIME model."""
    print("Loading GigaTIME model...")
    model = archs.gigatime(NUM_CLASSES, INPUT_CHANNELS)

    repo_id = "prov-gigatime/GigaTIME"
    local_dir = snapshot_download(repo_id=repo_id)
    weights_path = os.path.join(local_dir, "model.pth")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model, device


MODEL, DEVICE = load_model()

# ── Inference ─────────────────────────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Resize to INPUT_SIZE, normalize with ImageNet stats, return [1,3,H,W] tensor."""
    img = pil_image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0          # [H,W,3] in [0,1]
    img_np = (img_np - MEAN) / STD                              # normalize
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float()  # [1,3,H,W]
    return img_tensor


def sliding_window_inference(img_tensor: torch.Tensor) -> torch.Tensor:
    """Run inference with 256×256 sliding window, matching the notebook approach."""
    b, c, h, w = img_tensor.shape
    output_logits = torch.zeros(b, NUM_CLASSES, h, w, device=img_tensor.device)
    with torch.no_grad():
        for i in range(0, h, WINDOW_SIZE):
            for j in range(0, w, WINDOW_SIZE):
                window = img_tensor[:, :, i:i + WINDOW_SIZE, j:j + WINDOW_SIZE]
                logits = MODEL(window)
                output_logits[:, :, i:i + WINDOW_SIZE, j:j + WINDOW_SIZE] = logits
    return output_logits


def run_gigatime(input_image: Image.Image):
    """
    Main prediction function.
    Returns a list of (PIL.Image, label) tuples for Gradio Gallery.
    """
    if input_image is None:
        return []

    img_tensor = preprocess_image(input_image).to(DEVICE)
    logits = sliding_window_inference(img_tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]              # [23, H, W]
    preds = (probs > THRESHOLD).astype(np.float32)

    # Also create the un-normalized H&E for display
    he_display = input_image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

    gallery_items = []
    # First item: the H&E input
    gallery_items.append((he_display, "H&E Input"))

    # Add each protein channel (skip background)
    for idx, name in enumerate(CHANNEL_NAMES):
        if name in EXCLUDE_CHANNELS:
            continue
        channel_map = preds[idx]                                # [H, W] binary
        prob_map = probs[idx]                                   # [H, W] continuous

        # Create a colored heatmap: green channel for probability overlay
        heatmap = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        # Use a colormap: intensity encodes probability
        intensity = (prob_map * 255).clip(0, 255).astype(np.uint8)
        # Use cyan-ish coloring for positive signal on dark background
        heatmap[:, :, 0] = (intensity * 0.2).astype(np.uint8)   # slight red
        heatmap[:, :, 1] = intensity                              # green dominant
        heatmap[:, :, 2] = (intensity * 0.8).astype(np.uint8)   # blue

        pil_heatmap = Image.fromarray(heatmap)
        gallery_items.append((pil_heatmap, name))

    return gallery_items


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="GigaTIME – Virtual mIF from H&E",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🔬 GigaTIME: Virtual Multiplex Immunofluorescence from H&E
        Upload an H&E pathology tile (ideally 256×256 or 512×512 px) and GigaTIME will
        predict activation maps for **21 protein channels**. Use the gallery arrows to
        flip through each channel.

        > **Research use only.** GigaTIME is not intended for clinical decision-making.
        > [Paper (Cell)](https://aka.ms/gigatime-paper) ·
        > [Model Card](https://aka.ms/gigatime-model) ·
        > [GitHub](https://github.com/prov-gigatime/GigaTIME)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload H&E Tile",
                height=400,
            )
            run_btn = gr.Button("Generate Virtual mIF", variant="primary", size="lg")
            gr.Markdown(
                """
                **Protein channels predicted:**
                DAPI, PD-1, CD14, CD4, T-bet, CD34, CD68, CD16, CD11c, CD138,
                CD20, CD3, CD8, PD-L1, CK, Ki67, Tryptase, Actin-D,
                Caspase3-D, PHH3-B, Transgelin
                """
            )

        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Virtual mIF Channels (click/arrow to browse)",
                columns=4,
                rows=3,
                height=600,
                object_fit="contain",
                preview=True,           # enables large preview + arrow navigation
            )

    run_btn.click(fn=run_gigatime, inputs=input_image, outputs=gallery)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)