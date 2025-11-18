from __future__ import annotations
import os, sys, time
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
        t_start = time.time()
        print(f"[KvasirSeg][INIT] root={root}")
        print(f"[KvasirSeg][INIT] received ids={len(ids)}")
        sys.stdout.flush()

        self.root = root
        self.transform = transform

        # Build (img, mask, id) pairs only for stems that have both files present/readable
        pairs: List[Tuple[str, str, str]] = []
        skipped = 0

        for i, stem in enumerate(list(ids)):
            if i % 50 == 0:
                print(f"[KvasirSeg][INIT] scanning id {i+1}/{len(ids)} -> '{stem}'")
                sys.stdout.flush()

            ip = _first_existing(root, "images", stem, IMG_EXTS)
            mp = _first_existing(root, "masks", stem, MSK_EXTS)

            if ip is None:
                print(f"[KvasirSeg][INIT][WARN] no image file for '{stem}' in {os.path.join(root, 'images')}")
                skipped += 1
                continue
            if mp is None:
                print(f"[KvasirSeg][INIT][WARN] no mask file for '{stem}' in {os.path.join(root, 'masks')}")
                skipped += 1
                continue

            # Quick readability check (avoid dying inside __getitem__)
            img_probe = cv2.imread(ip, cv2.IMREAD_COLOR)
            if img_probe is None:
                print(f"[KvasirSeg][INIT][WARN] unreadable image: {ip}")
                skipped += 1
                continue
            else:
                print(f"[KvasirSeg][INIT] image ok: '{stem}' path={ip} shape={img_probe.shape} dtype={img_probe.dtype}")

            msk_probe = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if msk_probe is None:
                print(f"[KvasirSeg][INIT][WARN] unreadable mask: {mp}")
                skipped += 1
                continue
            else:
                # Print a tiny summary of mask values without heavy ops
                u = np.unique(msk_probe[: min(64, msk_probe.shape[0]), : min(64, msk_probe.shape[1])])
                print(f"[KvasirSeg][INIT] mask ok:   '{stem}' path={mp} shape={msk_probe.shape} dtype={msk_probe.dtype} sample_unique={u.tolist()}")

            pairs.append((ip, mp, stem))

            # Flush frequently to see progress live
            if i % 10 == 0:
                sys.stdout.flush()

        if not pairs:
            print(f"[KvasirSeg][INIT][ERROR] No valid (image,mask) pairs under '{root}'. Checked {len(ids)} ids.")
            sys.stdout.flush()
            raise FileNotFoundError(
                f"No valid (image,mask) pairs under '{root}'. "
                f"Checked {len(ids)} ids; ensure matching files exist in images/ and masks/."
            )

        if skipped > 0:
            print(f"[KvasirSeg][INIT][WARN] filtered out {skipped} ids without usable masks/images at '{root}'.")

        self.img_paths = [p for p, _, _ in pairs]
        self.msk_paths = [m for _, m, _ in pairs]
        self.ids = [s for _, _, s in pairs]

        print(f"[KvasirSeg][INIT] built pairs={len(self.ids)} | skipped={skipped} | elapsed_init={time.time() - t_start:.2f}s")
        if len(self.ids) > 0:
            print(f"[KvasirSeg][INIT] first id='{self.ids[0]}'")
            print(f"[KvasirSeg][INIT] first image='{self.img_paths[0]}'")
            print(f"[KvasirSeg][INIT] first mask ='{self.msk_paths[0]}'")
        sys.stdout.flush()

    def __len__(self):
        n = len(self.ids)
        print(f"[KvasirSeg][LEN] {n}")
        sys.stdout.flush()
        return n

    def __getitem__(self, idx: int):
        t0 = time.time()
        if idx < 0 or idx >= len(self.ids):
            print(f"[KvasirSeg][GET][ERROR] idx out of range: idx={idx}, len={len(self.ids)}")
            sys.stdout.flush()
            raise IndexError("Index out of range")

        stem = self.ids[idx]
        ip = self.img_paths[idx]
        mp = self.msk_paths[idx]
        print(f"[KvasirSeg][GET] idx={idx} id='{stem}'")
        print(f"[KvasirSeg][GET] reading image: {ip}")
        sys.stdout.flush()

        img_cv = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img_cv is None:
            print(f"[KvasirSeg][GET][ERROR] Image not found or unreadable: {ip}")
            sys.stdout.flush()
            raise FileNotFoundError(f"Image not found or unreadable: {ip}")
        print(f"[KvasirSeg][GET] image read ok: shape={img_cv.shape} dtype={img_cv.dtype}")

        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        print(f"[KvasirSeg][GET] image converted BGR->RGB: shape={img.shape} dtype={img.dtype}")

        print(f"[KvasirSeg][GET] reading mask:  {mp}")
        sys.stdout.flush()
        mask_cv = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask_cv is None:
            print(f"[KvasirSeg][GET][ERROR] Mask not found or unreadable: {mp}")
            sys.stdout.flush()
            raise FileNotFoundError(f"Mask not found or unreadable: {mp}")
        print(f"[KvasirSeg][GET] mask read ok: shape={mask_cv.shape} dtype={mask_cv.dtype} min={int(mask_cv.min())} max={int(mask_cv.max())}")

        # Binary mask {0,1}
        mask = (mask_cv > 0).astype(np.uint8)
        print(f"[KvasirSeg][GET] mask binarized -> unique={np.unique(mask).tolist()}")

        if self.transform:
            print(f"[KvasirSeg][GET] applying transform ...")
            sys.stdout.flush()
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
            # Log post-transform types/shapes
            if isinstance(img, np.ndarray):
                print(f"[KvasirSeg][GET] transform ok: image ndarray shape={img.shape} dtype={img.dtype}")
            else:
                # Albumentations ToTensorV2 returns torch.Tensor
                try:
                    print(f"[KvasirSeg][GET] transform ok: image tensor shape={tuple(img.shape)} dtype={img.dtype}")
                except Exception:
                    print(f"[KvasirSeg][GET] transform ok: image type={type(img)}")
            if isinstance(mask, np.ndarray):
                print(f"[KvasirSeg][GET] transform ok: mask ndarray shape={mask.shape} dtype={mask.dtype}")
            else:
                try:
                    print(f"[KvasirSeg][GET] transform ok: mask tensor shape={tuple(mask.shape)} dtype={mask.dtype}")
                except Exception:
                    print(f"[KvasirSeg][GET] transform ok: mask type={type(mask)}")
            sys.stdout.flush()

        # Robust tensor conversion
        if isinstance(img, np.ndarray):
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
            print(f"[KvasirSeg][GET] image -> tensor: shape={tuple(img_t.shape)} dtype={img_t.dtype} "
                  f"min={float(img_t.min()):.3f} max={float(img_t.max()):.3f}")
        else:
            img_t = img.float()
            try:
                print(f"[KvasirSeg][GET] image already tensor: shape={tuple(img_t.shape)} dtype={img_t.dtype} "
                      f"min={float(img_t.min()):.3f} max={float(img_t.max()):.3f}")
            except Exception:
                print(f"[KvasirSeg][GET] image already tensor: dtype={getattr(img_t, 'dtype', 'unknown')}")

        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(mask).long()
            # Keep this cheap: show a small crop stats
            uniq = torch.unique(mask_t[: min(32, mask_t.shape[0]), : min(32, mask_t.shape[1])]).tolist()
            print(f"[KvasirSeg][GET] mask -> tensor: shape={tuple(mask_t.shape)} dtype={mask_t.dtype} sample_unique={uniq}")
        else:
            mask_t = mask.long()
            try:
                uniq = torch.unique(mask_t[: min(32, mask_t.shape[0]), : min(32, mask_t.shape[1])]).tolist()
            except Exception:
                uniq = "n/a"
            print(f"[KvasirSeg][GET] mask already tensor: shape={tuple(mask_t.shape)} dtype={mask_t.dtype} sample_unique={uniq}")

        print(f"[KvasirSeg][GET] done idx={idx} elapsed={time.time() - t0:.3f}s")
        sys.stdout.flush()

        return img_t, mask_t
