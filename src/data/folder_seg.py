from __future__ import annotations
import os
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class FolderSeg(Dataset):
    """Generic images/masks dataset.
    Returns:
        img: FloatTensor [3,H,W] in [0,1]
        mask: LongTensor [H,W]
    """
    def __init__(self, root: str, ids: List[str], transform=None):
        self.root = root
        self.ids = list(ids)
        self.transform = transform
        self.imgs = [self._img_path(i) for i in self.ids]
        self.msks = [self._msk_path(i) for i in self.ids]

    def _img_path(self, i: str) -> str:
        jpg = os.path.join(self.root, "images", f"{i}.jpg")
        if os.path.exists(jpg): return jpg
        png = os.path.join(self.root, "images", f"{i}.png")
        return png

    def _msk_path(self, i: str) -> str:
        png = os.path.join(self.root, "masks", f"{i}.png")
        if os.path.exists(png): return png
        jpg = os.path.join(self.root, "masks", f"{i}.jpg")
        return jpg

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx: int):
        ip, mp = self.imgs[idx], self.msks[idx]

        img_cv = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img_cv is None:
            raise FileNotFoundError(f"Image not found: {ip}")
        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        mask_cv = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask_cv is None:
            raise FileNotFoundError(f"Mask not found: {mp}")
        mask = mask_cv.astype(np.int64)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        else:
            img = img.float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return img, mask
