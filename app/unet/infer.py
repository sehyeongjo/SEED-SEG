import os
import glob
import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
import torch

from src.utils import ensure_dir
from src.module import SegLitModule


def load_model(cfg, ckpt_path: str, device: str):
    model = SegLitModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval()
    model.to(device)
    return model


def _resize_with_aspect(img_rgb: np.ndarray, target_h: int, target_w: int):
    """Resize keeping aspect ratio so longer side fits max(target_h, target_w)."""
    h, w = img_rgb.shape[:2]
    max_side = max(target_h, target_w)
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


@torch.no_grad()
def infer_one(model, img_path: str, img_h: int, img_w: int, threshold: float, device: str):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Keep aspect ratio then pad/crop to model size.
    resized = _resize_with_aspect(img_rgb, img_h, img_w)
    rh, rw = resized.shape[:2]
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    copy_h = min(img_h, rh)
    copy_w = min(img_w, rw)
    canvas[:copy_h, :copy_w] = resized[:copy_h, :copy_w]

    x = canvas.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    pred = (prob >= threshold).astype(np.uint8) * 255

    # Recover prediction to original image size.
    pred = pred[:copy_h, :copy_w]
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return pred, img_bgr


def make_overlay(img_bgr: np.ndarray, mask_255: np.ndarray, color=(0, 255, 0), alpha=0.45):
    overlay = img_bgr.copy()
    colored = np.zeros_like(img_bgr)
    colored[mask_255 > 0] = color
    overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)
    return overlay


def make_cutout_rgba(img_bgr: np.ndarray, mask_255: np.ndarray):
    # Transparent background, only segmented region visible.
    b, g, r = cv2.split(img_bgr)
    alpha = mask_255.copy()
    return cv2.merge([b, g, r, alpha])


def main():
    parser = argparse.ArgumentParser("Infer U-Net masks from trayseg cell images")
    parser.add_argument("--tray_num", type=int, required=True)
    parser.add_argument("--checkpoint_name", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--input_root", type=str, default="/data/trayseg_output")
    parser.add_argument("--output_root", type=str, default="/data/unet_output")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else (script_dir / "config.yaml")
    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else (script_dir / "checkpoints")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tray_num = str(args.tray_num)
    in_dir = Path(args.input_root) / tray_num
    out_dir = Path(args.output_root) / tray_num
    ckpt_path = checkpoints_dir / args.checkpoint_name
    overlay_color_bgr = (0, 255, 0)
    overlay_alpha = float(args.overlay_alpha)

    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ensure_dir(str(out_dir))

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(cfg, str(ckpt_path), device)

    img_h = int(cfg["train"]["img_h"])
    img_w = int(cfg["train"]["img_w"])
    threshold = float(cfg["train"]["threshold"])

    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"):
        files.extend(glob.glob(str(in_dir / "merged_*" / "cells" / ext)))
    files = sorted(files)
    if len(files) == 0:
        raise SystemExit(f"No images found in: {in_dir}/merged_*/cells")

    total = len(files)
    print(f"[INFO] tray={tray_num}, found {total} images for inference")

    for idx, p in enumerate(files, start=1):
        rel_file = os.path.relpath(p, str(in_dir))
        pct = (idx / total) * 100.0
        print(f"[{idx}/{total}] ({pct:6.2f}%) {rel_file}", flush=True)

        pred, img_bgr = infer_one(model, p, img_h=img_h, img_w=img_w, threshold=threshold, device=device)
        stem = os.path.splitext(os.path.basename(p))[0]
        merged_name = Path(p).parent.parent.name

        overlay_dir = out_dir / merged_name / "overlay"
        cutout_dir = out_dir / merged_name / "cutout"
        ensure_dir(str(overlay_dir))
        ensure_dir(str(cutout_dir))

        overlay_path = overlay_dir / f"{stem}.png"
        cutout_path = cutout_dir / f"{stem}.png"

        overlay = make_overlay(img_bgr, pred, color=overlay_color_bgr, alpha=overlay_alpha)
        cutout_rgba = make_cutout_rgba(img_bgr, pred)

        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(cutout_path), cutout_rgba)

    print(f"[DONE] Input: {in_dir}")
    print(f"[DONE] Saved overlays/cutouts under: {out_dir}")


if __name__ == "__main__":
    main()
