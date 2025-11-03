from __future__ import annotations
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class KvasirSeg(Dataset):
    """Kvasir-SEG loader.

    Expected layout:
        root/
          images/*.jpg|*.png
          masks/*.png  (binary mask: 0 background, >0 foreground)

    Returns:
        img:  torch.FloatTensor [3,H,W] in [0,1]
        mask: torch.LongTensor  [H,W]   with values {0,1}
    """
    def __init__(self, root: str, ids: List[str], transform=None):
        self.root = root
        self.ids = list(ids)  # copy to avoid accidental external mutation
        self.transform = transform
        self.img_paths = [self._img_path(i) for i in self.ids]
        self.msk_paths = [self._msk_path(i) for i in self.ids]

    def _img_path(self, i: str) -> str:
        # try jpg then png
        p_jpg = os.path.join(self.root, "images", f"{i}.jpg")
        if os.path.exists(p_jpg):
            return p_jpg
        p_png = os.path.join(self.root, "images", f"{i}.png")
        return p_png

    def _msk_path(self, i: str) -> str:
        return os.path.join(self.root, "masks", f"{i}.png")

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
        mask = (mask_cv > 0).astype(np.uint8)  # {0,1}

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        # Robust tensor conversion:
        # - If Albumentations+ToTensorV2 ran, these are already torch.Tensors.
        # - If not, convert from numpy.
        if isinstance(img, np.ndarray):
            # [H,W,3] uint8 -> [3,H,W] float32 in [0,1]
            img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        else:
            # ensure float32
            img = img.float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return img, mask
