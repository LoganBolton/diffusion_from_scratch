from diffusion import *
import torchvision
import torch
import os
import time
import json
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from text_embedding import ClipTextEncoder
from dit import DiT
from flow_matching import FlowMatching

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A picture of a cat")
    parser.add_argument("--w", type=float, default=3.0)
    args = parser.parse_args()

    model = DiT(num_layers=12)
    model.load_state_dict(torch.load("checkpoints/flow_match_v1/model_best_epoch_517.pt"))
    model.eval()
    model.to("cuda")

    PROMPT = args.prompt
    TIMESTEPS = 1000
    w = args.w

    # constants = DiffusionConstants(t=TIMESTEPS, device="cuda")
    fm = FlowMatching()
    x_t = torch.randn(1, 3, 64, 64).to("cuda")

    current_sys_time = time.time()
    output_dir = f"outputs/{current_sys_time}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/meta.json", "w") as f:
        json.dump({"prompt": PROMPT, "w": w}, f)

    def save_frame(tensor, idx):
        scaled = ((tensor + 1) / 2).clamp(0, 1)
        path = f"{output_dir}/sample_{idx:03d}.png"
        torchvision.utils.save_image(scaled, path)
        return path

    text_encoder = ClipTextEncoder(device="cuda")
    text_embed = text_encoder.embed_text(PROMPT)

    # Continuous time walked from 1 (noise) -> 0 (data) via Euler ODE steps.
    NUM_STEPS = 20
    ts = torch.linspace(1.0, 0.0, NUM_STEPS + 1)

    frame_paths = [save_frame(x_t, 0)]      # frame 0 = the initial pure noise (t=1)
    frame_labels = [f"t={ts[0]:.2f}"]

    with torch.no_grad():
        for i in range(NUM_STEPS):
            t = ts[i].item()
            t_prev = ts[i + 1].item()
            x_t = fm.sample_step(model, x_t, t, TIMESTEPS, t_prev, text_embed, w)

            frame_paths.append(save_frame(x_t, i + 1))
            frame_labels.append(f"t={t_prev:.2f}")

    # Left to right: noisy -> clean
    images = [Image.open(p).convert("RGB") for p in frame_paths]

    _, axes = plt.subplots(1, len(images), figsize=(len(images) * 2, 2))

    if len(images) == 1:
        axes = [axes]

    for ax, img, label in zip(axes, images, frame_labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/denoising_left_to_right.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()