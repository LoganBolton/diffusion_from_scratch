"""Stack all denoising_left_to_right.png images vertically with class labels."""

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


LABEL_WIDTH = 200


def stack_images(input_dir: str | None = None, output_path: str | None = None):
    if input_dir is None:
        input_dir = Path(__file__).parent / "text_conditioning_w_5"
    else:
        input_dir = Path(input_dir)
    subdirs = sorted(input_dir.iterdir())

    rows = []
    for subdir in subdirs:
        img_path = subdir / "denoising_left_to_right.png"
        meta_path = subdir / "meta.json"
        if not img_path.exists():
            continue
        label = ""
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            label = meta.get("prompt", "").replace("a photo of ", "")
        rows.append((label, Image.open(img_path)))

    if not rows:
        print(f"No denoising_left_to_right.png images found in {input_dir}")
        return

    max_img_width = max(img.width for _, img in rows)
    row_height = max(img.height for _, img in rows)
    total_width = LABEL_WIDTH + max_img_width
    total_height = row_height * len(rows)

    stacked = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(stacked)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    y = 0
    for label, img in rows:
        # Draw label centered vertically in the row
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (LABEL_WIDTH - text_w) // 2
        text_y = y + (row_height - text_h) // 2
        draw.text((text_x, text_y), label, fill="black", font=font)
        stacked.paste(img, (LABEL_WIDTH, y))
        y += row_height

    if output_path is None:
        output_path = input_dir / "stacked.png"

    stacked.save(output_path)
    print(f"Saved {len(rows)} images stacked to {output_path} ({total_width}x{total_height})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack denoising images vertically")
    parser.add_argument("--input-dir", default=None, help="Directory containing output subdirs (default: text_conditioning next to this script)")
    parser.add_argument("--output", default=None, help="Output file path (default: <input-dir>/stacked.png)")
    args = parser.parse_args()
    stack_images(args.input_dir, args.output)
