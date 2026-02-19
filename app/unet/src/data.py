import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _list_merged_folders(root: str, subdir: str) -> List[str]:
    # ex) /data/mask_data/<tray>/original/merged_*
    base = os.path.join(root, subdir)
    return sorted([p for p in glob.glob(os.path.join(base, "merged_*")) if os.path.isdir(p)])


def build_pairs(
    root: str,
    original_dir: str,
    mask_dir: str,
    img_ext: str = ".png",
    mask_ext: str = ".png",
    merged_ids: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Returns list of (orig_path, mask_path, key) where key = "merged_*/{B}.png"
    """
    orig_merged_paths = _list_merged_folders(root, original_dir)
    mask_merged_paths = _list_merged_folders(root, mask_dir)

    # map: merged_folder_name -> full path
    orig_map = {os.path.basename(p): p for p in orig_merged_paths}
    mask_map = {os.path.basename(p): p for p in mask_merged_paths}

    merged_names = sorted(set(orig_map.keys()) & set(mask_map.keys()))

    if merged_ids:
        wanted_prefixes = tuple(f"merged_{mid}" for mid in merged_ids)
        merged_names = [m for m in merged_names if m.startswith(wanted_prefixes)]

    pairs = []
    for merged in merged_names:
        o_dir = orig_map[merged]
        m_dir = mask_map[merged]

        # files: .../merged_x_output/{B}.png
        o_files = glob.glob(os.path.join(o_dir, f"*{img_ext}"))
        # build quick lookup for mask
        m_lookup = {os.path.basename(p): p for p in glob.glob(os.path.join(m_dir, f"*{mask_ext}"))}

        for o_path in sorted(o_files):
            fname = os.path.basename(o_path)
            m_path = m_lookup.get(fname)
            if m_path is None:
                continue
            key = os.path.join(merged, fname)
            pairs.append((o_path, m_path, key))

    if len(pairs) == 0:
        raise RuntimeError(
            f"No pairs found. Check paths like {root}/{original_dir}/merged_* and {root}/{mask_dir}/merged_*"
        )
    return pairs


def make_transforms(img_h: int, img_w: int, aug_enable: bool, rotate_limit: int, scale_limit: float, brightness_contrast: float):
    # after optional augmentations, force final tensor size to (img_h, img_w)
    def _fix_size():
        return [
            A.PadIfNeeded(
                min_height=img_h,
                min_width=img_w,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            A.CenterCrop(height=img_h, width=img_w),
        ]

    train_tfms = [A.LongestMaxSize(max_size=max(img_h, img_w))]
    if aug_enable:
        train_tfms += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=rotate_limit, p=0.6, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.RandomScale(scale_limit=scale_limit, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=brightness_contrast, contrast_limit=brightness_contrast, p=0.5),
        ]

    train_tfms += _fix_size() + [A.Normalize(), ToTensorV2()]

    val_tfms = [A.LongestMaxSize(max_size=max(img_h, img_w))] + _fix_size() + [A.Normalize(), ToTensorV2()]
    return A.Compose(train_tfms), A.Compose(val_tfms)


class SegPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, str]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, key = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # mask: 0/255 or arbitrary => binarize to {0,1}
        mask = (mask > 127).astype(np.uint8)

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img_t = out["image"]               # [3,H,W] float
            mask_t = out["mask"].unsqueeze(0)  # [1,H,W] uint8 -> ToTensorV2 gives torch tensor
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        mask_t = mask_t.float()  # ensure float for loss
        return {"image": img_t, "mask": mask_t, "key": key, "img_path": img_path, "mask_path": mask_path}


@dataclass
class SplitConfig:
    val_ratio: float = 0.15
    test_ratio: float = 0.0
    seed: int = 42


def split_pairs(pairs: List[Tuple[str, str, str]], cfg: SplitConfig):
    # Shuffle before split to preserve shape diversity even in a single domain.
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)

    n = len(pairs)
    n_test = int(n * cfg.test_ratio)
    n_val = int(n * cfg.val_ratio)
    n_train = n - n_val - n_test

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train = [pairs[i] for i in train_idx]
    val = [pairs[i] for i in val_idx]
    test = [pairs[i] for i in test_idx]
    return train, val, test


class SegDataModule:
    def __init__(
        self,
        pairs,
        img_h: int,
        img_w: int,
        batch_size: int,
        num_workers: int,
        val_ratio: float,
        test_ratio: float,
        seed: int,
        aug_enable: bool,
        rotate_limit: int,
        scale_limit: float,
        brightness_contrast: float,
    ):
        self.pairs = pairs
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.aug_enable = aug_enable
        self.rotate_limit = rotate_limit
        self.scale_limit = scale_limit
        self.brightness_contrast = brightness_contrast

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self):
        train_pairs, val_pairs, test_pairs = split_pairs(
            self.pairs, SplitConfig(self.val_ratio, self.test_ratio, self.seed)
        )
        train_tfms, val_tfms = make_transforms(
            self.img_h, self.img_w, self.aug_enable, self.rotate_limit, self.scale_limit, self.brightness_contrast
        )

        self.train_ds = SegPairDataset(train_pairs, transform=train_tfms)
        self.val_ds = SegPairDataset(val_pairs, transform=val_tfms)
        self.test_ds = SegPairDataset(test_pairs, transform=val_tfms) if len(test_pairs) else None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

