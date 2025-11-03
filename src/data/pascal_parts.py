from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class PascalParts(Dataset):
    """PASCAL-Parts (semantic part parsing) with PNG masks (integer ids)."""
    def __init__(self, root: str, split: str, transform=None, ignore_index: int | None = 255):
        self.root = root
        self.transform = transform
        self.ignore_index = ignore_index
        with open(os.path.join(root, "ImageSets", "Part", f"{split}.txt")) as f:
            self.ids = [x.strip() for x in f.readlines()]

    def __len__(self): 
        return len(self.ids)

    def __getitem__(self, idx: int):
        _id = self.ids[idx]
        img = Image.open(os.path.join(self.root, "JPEGImages", f"{_id}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.root, "SegmentationPart", f"{_id}.png"))
        img = np.array(img)
        mask = np.array(mask, dtype=np.int64)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        return img.float(), torch.from_numpy(mask).long()
