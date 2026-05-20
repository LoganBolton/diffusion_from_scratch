"""Stack all denoising_left_to_right.png images vertically into a single image."""

import argparse
from pathlib import Path
from PIL import Image


def stack_images(input_dir: str = "outputs", output_path: str | None = None):
    input_dir = Path(input_dir)
    subdirs = sorted(input_dir.iterdir())

    images = []
    for subdir in subdirs:
        img_path = subdir / "denoising_left_to_right.png"
        if img_path.exists():
            images.append(Image.open(img_path))

    if not images:
        print(f"No denoising_left_to_right.png images found in {input_dir}")
        return

    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)

    stacked = Image.new("RGBA", (max_width, total_height))
    y = 0
    for img in images:
        stacked.paste(img, (0, y))
        y += img.height

    if output_path is None:
        output_path = input_dir / "stacked.png"

    stacked.save(output_path)
    print(f"Saved {len(images)} images stacked to {output_path} ({max_width}x{total_height})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack denoising images vertically")
    parser.add_argument("--input-dir", default="outputs", help="Directory containing output subdirs")
    parser.add_argument("--output", default=None, help="Output file path (default: <input-dir>/stacked.png)")
    args = parser.parse_args()
    stack_images(args.input_dir, args.output)
