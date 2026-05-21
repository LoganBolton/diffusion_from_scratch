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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A picture of a cat")
    parser.add_argument("--w", type=float, default=3.0)
    args = parser.parse_args()

    model = UNet()
    model.load_state_dict(torch.load("checkpoints/text_condition/model_best_epoch_269.pt"))
    model.eval()
    model.to("cuda")

    PROMPT = args.prompt
    TIMESTEPS = 1000
    SAVE_EVERY = 200
    w = args.w

    constants = DiffusionConstants(t=TIMESTEPS, device="cuda")
    x_t = torch.randn(1, 3, 64, 64).to("cuda")

    current_sys_time = time.time()
    output_dir = f"outputs/{current_sys_time}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/meta.json", "w") as f:
        json.dump({"prompt": PROMPT, "w": w}, f)

    # Save the initial pure noise image at T=1000
    scaled_x_t = (x_t + 1) / 2
    scaled_x_t = scaled_x_t.clamp(0, 1)
    torchvision.utils.save_image(
        scaled_x_t,
        f"{output_dir}/sample_{TIMESTEPS}.png"
    )

    text_encoder = ClipTextEncoder(device="cuda")
    text_embed = text_encoder.embed_text(PROMPT)

    with torch.no_grad():
        for t in range(TIMESTEPS - 1, -1, -1):
            x_t = constants.sample_step(model, x_t, t, text_embed, w)

            if t % SAVE_EVERY == 0:
                print(f"Step {t}/{TIMESTEPS}")
                scaled_x_t = (x_t + 1) / 2
                scaled_x_t = scaled_x_t.clamp(0, 1)
                torchvision.utils.save_image(
                    scaled_x_t,
                    f"{output_dir}/sample_{t}.png"
                )

    # Left to right: noisy -> clean
    frame_ts = [TIMESTEPS] + list(range(TIMESTEPS - SAVE_EVERY, -1, -SAVE_EVERY))

    images = [
        Image.open(f"{output_dir}/sample_{t}.png").convert("RGB")
        for t in frame_ts
    ]

    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 2, 2))

    if len(images) == 1:
        axes = [axes]

    for ax, img, t in zip(axes, images, frame_ts):
        ax.imshow(img)
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/denoising_left_to_right.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()