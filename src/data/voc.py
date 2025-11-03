from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

IGNORE_INDEX = 255


class VOCSeg(Dataset):
    """Pascal VOC 2012 semantic segmentation.

    Expected layout:
        root/VOCdevkit/VOC2012/
          JPEGImages/{id}.jpg
          SegmentationClass/{id}.png
          ImageSets/Segmentation/{split}.txt

    Returns:
        img:  torch.FloatTensor [3,H,W] in [0,1]
        mask: torch.LongTensor  [H,W] with values in {0..20, 255(ignore)}
    """
    def __init__(
        self,
        root: str,
        split: str,
        ids_subset: Optional[List[str]] = None,
        transform=None,
    ):
        self.root = os.path.join(root, "VOCdevkit", "VOC2012")
        self.transform = transform

        list_file = os.path.join(self.root, "ImageSets", "Segmentation", f"{split}.txt")
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Split file not found: {list_file}")

        with open(list_file) as f:
            all_ids = [x.strip() for x in f.readlines()]

        subset = set(ids_subset) if ids_subset is not None else None
        self.ids = [i for i in all_ids if (subset is None or i in subset)]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        _id = self.ids[idx]
        img_p = os.path.join(self.root, "JPEGImages", f"{_id}.jpg")
        msk_p = os.path.join(self.root, "SegmentationClass", f"{_id}.png")

        if not os.path.exists(img_p):
            raise FileNotFoundError(f"Image not found: {img_p}")
        if not os.path.exists(msk_p):
            raise FileNotFoundError(f"Mask not found: {msk_p}")

        img = Image.open(img_p).convert("RGB")
        mask = Image.open(msk_p)

        img = np.asarray(img)                # [H,W,3], uint8
        mask = np.asarray(mask, dtype=np.int64)  # [H,W], {0..20, 255(ignore)}

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        # Robust conversion in case ToTensorV2 wasn't used
        if isinstance(img, np.ndarray):
            # [H,W,3] -> [3,H,W], float32 in [0,1]
            img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        else:
            img = img.float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return img, mask
