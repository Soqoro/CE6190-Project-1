from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

# Prefer names that match each dataset layout; VOC "slim" first for images/masks
IMG_DIR_CANDIDATES = ("JPEGImages", "images", "img")
# NEW: add person-part mask dirs while preserving VOC-first precedence
MSK_DIR_CANDIDATES = (
    "SegmentationClass",                 # VOC slim (21-class)
    "pascal_person_parts_gt",            # Person-Part (7-class)
    "pascal_person_part_gt",             # common variant name
    "PartMasks7",                        # another variant name
    "SegmentationPart",                  # some bundles use this name
    "masks",                             # flat generic
    "ann",                               # split generic
)

IMG_EXTS = (".jpg", ".jpeg", ".png")
# Keep .png first so palette masks are preferred over any stray jpgs
MSK_EXTS = (".png", ".jpg", ".jpeg")


def _pick_subdir(root: Path, candidates: Tuple[str, ...]) -> Optional[Path]:
    for name in candidates:
        p = root / name
        if p.exists() and p.is_dir():
            return p
    return None


def _list_stems(dirpath: Path, exts: Tuple[str, ...]) -> List[str]:
    stems: List[str] = []
    if not dirpath.exists():
        return stems
    for p in sorted(dirpath.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            stems.append(p.stem)
    return stems


def _to_tensor_image(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Return image as [C,H,W] float32.

    IMPORTANT:
    - If input is integer-typed (uint8/int*), scale to [0,1] by dividing by 255.
    - If input is already float (e.g., from Albumentations ToTensorV2/Normalize),
      DO NOT rescale. This avoids double-scaling that crushes activations.
    """
    if isinstance(x, torch.Tensor):
        t = x
        # If HWC tensor, move to CHW
        if t.ndim == 3 and t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
            t = t.permute(2, 0, 1).contiguous()
        if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            return t.to(torch.float32).div_(255.0)
        return t.to(torch.float32)

    # NumPy path
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[:, :, None]  # gray -> add channel
    # HWC -> CHW
    chw = np.transpose(arr, (2, 0, 1))  # C,H,W
    if np.issubdtype(arr.dtype, np.integer):
        t = torch.from_numpy(np.ascontiguousarray(chw)).float().div_(255.0)
    else:
        t = torch.from_numpy(np.ascontiguousarray(chw)).float()
    return t


def _to_tensor_mask(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Ensure a mask tensor [H,W] int64 of class indices.

    Accepts:
      - np.ndarray [H,W] or [H,W,1] or [H,W,3] (uint8/int)  -> converts to [H,W]
      - torch.Tensor [H,W] or [1,H,W] or [H,W,1] or [3,H,W] -> converts to [H,W]
    """
    if isinstance(x, torch.Tensor):
        m = x
        if m.ndim == 3:
            # [1,H,W] -> [H,W]
            if m.shape[0] == 1:
                m = m.squeeze(0)
            # [H,W,1] -> [H,W]
            if m.ndim == 3 and m.shape[-1] == 1:
                m = m.squeeze(-1)
            # If someone produced 3-channel mask, take first channel
            if m.ndim == 3 and (m.shape[0] == 3 or m.shape[-1] == 3):
                if m.shape[0] == 3:
                    m = m[0]
                else:
                    m = m[..., 0]
        return m.to(dtype=torch.long)

    m_np = np.asarray(x)
    if m_np.ndim == 3:
        # [H,W,1] -> [H,W]
        if m_np.shape[-1] == 1:
            m_np = m_np[..., 0]
        # [1,H,W] -> [H,W]
        elif m_np.shape[0] == 1:
            m_np = m_np[0]
        # If RGB accidentally, take first channel
        elif m_np.shape[-1] == 3:
            m_np = m_np[..., 0]
        elif m_np.shape[0] == 3:
            m_np = m_np[0]
    return torch.from_numpy(m_np.astype(np.int64, copy=False))


class FolderSeg(Dataset):
    """
    Generic segmentation dataset for folder layouts:
      - Flat:   root/{images,masks}
      - Split:  root/{img,ann}   (e.g., root is data/voc/train)
      - VOC slim: root/{JPEGImages, SegmentationClass}
      - Person-Part: root/{JPEGImages, pascal_person_parts_gt}

    Args:
      root: split root or dataset root containing the two subfolders
      ids:  optional list of filestems to use; if None, use all pairs in root
      transform: albumentations transform taking image,mask numpy arrays
    """
    def __init__(self, root: str | Path, ids: Optional[List[str]] = None, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.img_dir = _pick_subdir(self.root, IMG_DIR_CANDIDATES)
        self.msk_dir = _pick_subdir(self.root, MSK_DIR_CANDIDATES)
        if self.img_dir is None or self.msk_dir is None:
            raise FileNotFoundError(
                f"Expected image/mask folders under {self.root} "
                f"(tried {IMG_DIR_CANDIDATES} and {MSK_DIR_CANDIDATES})"
            )

        if ids is None:
            # derive IDs by intersection of available pairs (image-led)
            img_stems = set(_list_stems(self.img_dir, IMG_EXTS))
            msk_stems = {
                stem for stem in img_stems
                if any((self.msk_dir / f"{stem}{e}").exists() for e in MSK_EXTS)
            }
            self.ids = sorted(msk_stems)
        else:
            self.ids = list(ids)

        if len(self.ids) == 0:
            raise RuntimeError(f"No (image,mask) pairs found under {self.root}")

    def __len__(self):
        return len(self.ids)

    def _img_path(self, stem: str) -> Path:
        # prefer common extensions
        for ext in IMG_EXTS:
            p = self.img_dir / f"{stem}{ext}"
            if p.exists():
                return p
        # fall back: first that exists with same stem
        for p in self.img_dir.iterdir():
            if p.is_file() and p.stem == stem:
                return p
        raise FileNotFoundError(f"Image not found for id='{stem}' in {self.img_dir}")

    def _msk_path(self, stem: str) -> Path:
        for ext in MSK_EXTS:
            p = self.msk_dir / f"{stem}{ext}"
            if p.exists():
                return p
        for p in self.msk_dir.iterdir():
            if p.is_file() and p.stem == stem:
                return p
        raise FileNotFoundError(f"Mask not found for id='{stem}' in {self.msk_dir}")

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        img_p = self._img_path(stem)
        msk_p = self._msk_path(stem)

        # Read image with OpenCV (BGR) then convert to RGB; fallback to PIL if cv2 fails
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if img is None:
            # fallback path to avoid crashes on occasional unreadable files
            img = np.array(Image.open(img_p).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask with PIL to preserve palette indices (VOC/Person-Part: 0..C-1, 255=ignore)
        msk = np.array(Image.open(msk_p), dtype=np.uint8)
        if msk.ndim == 3 and msk.shape[-1] == 3:
            # If someone saved RGB masks, fall back to first channel
            msk = msk[..., 0]

        # Albumentations expects NumPy; transforms may return NumPy or Torch (with ToTensorV2).
        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]

        # Normalize types/shapes (no double scaling)
        img_t = _to_tensor_image(img)
        msk_t = _to_tensor_mask(msk)

        return img_t, msk_t
