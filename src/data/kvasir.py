from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png")
MSK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _first_existing(root: str, sub: str, stem: str, exts: Tuple[str, ...]) -> str | None:
    for ext in exts:
        p = os.path.join(root, sub, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None


class KvasirSeg(Dataset):
    """Kvasir-SEG style loader for flat folders.

    Expected layout:
        root/
          images/*.jpg|*.jpeg|*.png
          masks/*.png|*.jpg|*.jpeg|*.bmp|*.tif|*.tiff  (binary mask: 0 background, >0 foreground)

    Returns:
        img:  torch.FloatTensor [3,H,W] in [0,1]
        mask: torch.LongTensor  [H,W]   with values {0,1}
    """
    def __init__(self, root: str, ids: List[str], transform=None):
        self.root = root
        self.transform = transform

        # Build (img, mask, id) pairs only for stems that have both files present/readable
        pairs: List[Tuple[str, str, str]] = []
        skipped = 0
        for stem in list(ids):
            ip = _first_existing(root, "images", stem, IMG_EXTS)
            mp = _first_existing(root, "masks", stem, MSK_EXTS)
            if ip is None or mp is None:
                skipped += 1
                continue
            # Quick readability check (avoid dying inside __getitem__)
            if cv2.imread(ip, cv2.IMREAD_COLOR) is None or cv2.imread(mp, cv2.IMREAD_GRAYSCALE) is None:
                skipped += 1
                continue
            pairs.append((ip, mp, stem))

        if not pairs:
            raise FileNotFoundError(
                f"No valid (image,mask) pairs under '{root}'. "
                f"Checked {len(ids)} ids; ensure matching files exist in images/ and masks/."
            )
        if skipped > 0:
            print(f"[WARN] KvasirSeg: filtered out {skipped} ids without usable masks/images at '{root}'.")

        self.img_paths = [p for p, _, _ in pairs]
        self.msk_paths = [m for _, m, _ in pairs]
        self.ids = [s for _, _, s in pairs]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        ip = self.img_paths[idx]
        mp = self.msk_paths[idx]

        img_cv = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img_cv is None:
            raise FileNotFoundError(f"Image not found or unreadable: {ip}")
        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        mask_cv = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask_cv is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {mp}")
        # Binary mask {0,1}
        mask = (mask_cv > 0).astype(np.uint8)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        # Robust tensor conversion
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        else:
            img = img.float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return img, mask
