import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def binarize_mask(mask_float01, threshold=0.5):
    return (mask_float01 >= threshold).astype(np.uint8)