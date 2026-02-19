import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser("SAM mask generation for trayseg cell images")
    parser.add_argument("--tray_num", type=int, required=True)
    parser.add_argument("--input_root", type=str, default="/data/trayseg_output")
    parser.add_argument("--output_root", type=str, default="/data/mask_data")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    tray_num = str(args.tray_num)
    input_base = Path(args.input_root) / tray_num
    output_base = Path(args.output_root) / tray_num
    output_base.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_base.glob("*/cells/*.png"))
    if not png_files:
        raise RuntimeError(f"No input images found in {input_base} (expected: */cells/*.png)")

    generator = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device=args.device)

    for image_path in tqdm(png_files, desc=f"SAM tray {tray_num}"):
        target_name = image_path.stem
        merged_output_name = image_path.parent.parent.name
        img = Image.open(image_path).convert("RGB")

        out = generator(
            img,
            points_per_batch=64,
            points_per_side=32,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.9,
            min_mask_region_area=1,
            crop_n_layers=0,
            box_nms_thresh=0.2,
        )

        masks = out["masks"]
        scores = out["scores"].detach().cpu().numpy()

        mask_np = []
        areas = []
        for m in masks:
            arr = np.asarray(m)
            b = arr > 0
            mask_np.append(b)
            areas.append(int(b.sum()))

        areas = np.array(areas)
        h, w = img.size[1], img.size[0]
        img_area = h * w

        min_area = int(img_area * 0.002)
        max_area = int(img_area * 0.6)
        keep = (areas >= min_area) & (areas <= max_area)

        idx = np.where(keep)[0]
        idx = idx[np.argsort(scores[idx])[::-1]]
        idx = idx[:10]

        base = np.array(img).astype(np.float32)
        rng = np.random.default_rng(0)
        alpha = 0.45
        overlay = base.copy()
        label_positions = []

        for j, i in enumerate(idx):
            m = mask_np[i]
            color = rng.integers(0, 256, size=3, dtype=np.int32).astype(np.float32)
            overlay[m] = overlay[m] * (1 - alpha) + color * alpha

            ys, xs = np.where(m)
            if len(xs) > 0:
                cx = int(xs.mean())
                cy = int(ys.mean())
                label_positions.append((j + 1, cx, cy))

        out_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(out_img)
        font = ImageFont.load_default()
        for rank, x, y in label_positions:
            draw.text(
                (x, y),
                str(rank),
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
                font=font,
                anchor="mm",
            )

        save_path = output_base / "overlay" / merged_output_name / f"{target_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(save_path)

        save_path = output_base / "original" / merged_output_name / f"{target_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

        if len(idx) == 0:
            print(f"[skip] No masks left after area filtering: {target_name}")
            continue

        best_idx = int(idx[0])
        best_mask = np.asarray(masks[best_idx])
        binary_mask = (best_mask > 0.5).astype(np.uint8) * 255
        mask_img = Image.fromarray(binary_mask)

        save_path = output_base / "mask" / merged_output_name / f"{target_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(save_path)


if __name__ == "__main__":
    main()
