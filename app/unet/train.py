import os
import argparse
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from src.utils import seed_everything, ensure_dir
from src.data import build_pairs, SegDataModule
from src.module import SegLitModule


def _safe_num_workers(requested: int) -> int:
    shm_path = Path("/dev/shm")
    if not shm_path.exists():
        print("[WARN] /dev/shm is not available. Forcing num_workers=0.")
        return 0
    try:
        stat = os.statvfs(str(shm_path))
        shm_total = stat.f_frsize * stat.f_blocks
        if shm_total < 256 * 1024 * 1024:
            print(f"[WARN] /dev/shm is too small ({shm_total // (1024 * 1024)} MB). Forcing num_workers=0.")
            return 0
    except Exception:
        print("[WARN] Failed to inspect /dev/shm. Forcing num_workers=0.")
        return 0
    return max(0, int(requested))


def main():
    parser = argparse.ArgumentParser("Train U-Net with all trays under mask_data")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="/data/mask_data")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else (script_dir / "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(int(cfg["seed"]))

    data_root = Path(args.data_root)
    if not data_root.exists() or not data_root.is_dir():
        raise RuntimeError(f"data_root does not exist or is not a directory: {data_root}")

    ckpt_dir = str(script_dir / "checkpoints")
    ensure_dir(ckpt_dir)

    tray_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not tray_dirs:
        raise RuntimeError(f"No tray directories found under: {data_root}")

    pairs = []
    used_trays = []
    skipped_trays = []
    for tray_dir in tray_dirs:
        try:
            tray_pairs = build_pairs(
                root=str(tray_dir),
                original_dir=cfg["data"]["original_dir"],
                mask_dir=cfg["data"]["mask_dir"],
                img_ext=cfg["data"]["img_ext"],
                mask_ext=cfg["data"]["mask_ext"],
                merged_ids=cfg["data"]["merged_ids"] if cfg["data"]["merged_ids"] else None,
            )
        except RuntimeError:
            skipped_trays.append(tray_dir.name)
            continue
        pairs.extend(tray_pairs)
        used_trays.append(tray_dir.name)

    if not pairs:
        raise RuntimeError(f"No train pairs found in any tray under: {data_root}")

    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] used_trays={len(used_trays)} ({', '.join(used_trays)})")
    if skipped_trays:
        print(f"[INFO] skipped_trays={len(skipped_trays)} ({', '.join(skipped_trays)})")
    print(f"[INFO] total_pairs={len(pairs)}")

    requested_workers = int(cfg["train"]["num_workers"])
    safe_workers = _safe_num_workers(requested_workers)
    if safe_workers != requested_workers:
        print(f"[INFO] num_workers overridden: {requested_workers} -> {safe_workers}")

    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    dm = SegDataModule(
        pairs=pairs,
        img_h=int(cfg["train"]["img_h"]),
        img_w=int(cfg["train"]["img_w"]),
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=safe_workers,
        val_ratio=float(cfg["train"]["val_ratio"]),
        test_ratio=float(cfg["train"]["test_ratio"]),
        seed=int(cfg["seed"]),
        aug_enable=bool(cfg["aug"]["enable"]),
        rotate_limit=int(cfg["aug"]["rotate_limit"]),
        scale_limit=float(cfg["aug"]["scale_limit"]),
        brightness_contrast=float(cfg["aug"]["brightness_contrast"]),
    )
    dm.setup()

    model = SegLitModule(cfg)

    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val_dice:.4f}",
        monitor=cfg["logging"]["ckpt_monitor"],
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    es = EarlyStopping(monitor=cfg["logging"]["ckpt_monitor"], mode="max", patience=15)
    lrm = LearningRateMonitor(logging_interval="epoch")
    csv_logger = CSVLogger(save_dir=ckpt_dir, name="csv_logs")

    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        max_epochs=int(cfg["train"]["max_epochs"]),
        precision=cfg["train"]["precision"],
        accelerator="auto",
        devices="auto",
        logger=csv_logger,
        callbacks=[ckpt, es, lrm],
        log_every_n_steps=20,
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
    print(f"[DONE] Checkpoint dir: {ckpt_dir}")
    print(f"[DONE] Best checkpoint: {ckpt.best_model_path}")
    print(f"[DONE] Epoch metrics CSV: {csv_logger.log_dir}/metrics.csv")


if __name__ == "__main__":
    main()
