from __future__ import annotations
import os, json, yaml, csv, time, glob, shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from src.data.transforms import make_transforms
from src.data.kvasir import KvasirSeg
from src.data.voc import VOCSeg, IGNORE_INDEX as VOC_IGNORE
from src.data.pascal_parts import PascalParts
from src.data.folder_seg import FolderSeg
from src.models.smp_wrapper import SMPWrapper

# torchmetrics (match your LitSeg)
from torchmetrics.segmentation import DiceScore
import torchmetrics as tm
import cv2
from PIL import Image  # for palette-preserving mask reads


# ------------------------------- helpers -------------------------------

def _flat_layout_exists(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "masks"))

def _has_voc_split_layout(root: str) -> bool:
    """Detect data/voc/{train,val[,test]}/{img,ann} layout."""
    return (
        os.path.isdir(os.path.join(root, "train", "img"))
        and os.path.isdir(os.path.join(root, "train", "ann"))
        and os.path.isdir(os.path.join(root, "val", "img"))
        and os.path.isdir(os.path.join(root, "val", "ann"))
    )

def _has_voc_slim_layout(root: str) -> bool:
    """
    Detect official VOC2012-like 'slim' layout without VOCdevkit/VOC2012 wrapper:
      root/{JPEGImages, SegmentationClass, ImageSets/Segmentation, Annotations}
    """
    return (
        os.path.isdir(os.path.join(root, "JPEGImages")) and
        os.path.isdir(os.path.join(root, "SegmentationClass")) and
        os.path.isdir(os.path.join(root, "ImageSets", "Segmentation"))
    )

def _parts_layout_mask_dir(root: str) -> Optional[str]:
    """Detect PASCAL-Person-Part-like layout. Return mask dir name if found."""
    if not os.path.isdir(os.path.join(root, "JPEGImages")):
        return None
    for d in ("pascal_person_part_gt", "pascal_person_parts_gt", "SegmentationPart", "PartMasks7"):
        if os.path.isdir(os.path.join(root, d)):
            return d
    return None

def _read_ids_or_manifest(path: str) -> Dict[str, List[str]] | List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {
            "train": list(data.get("train", [])),
            "val": list(data.get("val", [])),
            "test": list(data.get("test", [])),
        }
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unsupported split file format: {path}")

def _read_txt_ids(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _get_test_ids(cfg) -> List[str]:
    """Return explicit test IDs if present, else fall back to 'val', else [] (handled upstream)."""
    sp_manifest = cfg.get("split_manifest") or cfg.get("split_file")
    if sp_manifest:
        data = _read_ids_or_manifest(sp_manifest)
        if isinstance(data, dict):
            test_ids = list(data.get("test", []))
            if test_ids:
                return test_ids
            return list(data.get("val", []))
        return []
    return []


# --------- tiny in-file datasets for special layouts ---------

class _VOCDatasetSlim(Dataset):
    """
    Minimal VOC dataset for the 'slim' layout:
      root/JPEGImages/<id>.jpg
      root/SegmentationClass/<id>.png   (indexed palette, 255=ignore)
    """
    def __init__(self, root: str | Path, ids: Optional[List[str]] = None, transform=None):
        self.root = Path(root)
        self.img_dir = self.root / "JPEGImages"
        self.msk_dir = self.root / "SegmentationClass"
        self.transform = transform

        if ids is None:
            self.ids = []
            for p in sorted(self.msk_dir.iterdir()):
                if p.is_file() and p.suffix.lower() == ".png":
                    stem = p.stem
                    if (self.img_dir / f"{stem}.jpg").exists() or (self.img_dir / f"{stem}.png").exists():
                        self.ids.append(stem)
        else:
            self.ids = list(ids)
        if not self.ids:
            raise RuntimeError(f"No (image, mask) pairs found under {self.root}")

    def __len__(self): return len(self.ids)

    def _img_path(self, stem: str) -> Path:
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.img_dir / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found for id='{stem}' under {self.img_dir}")

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        img_p = self._img_path(stem)
        msk_p = self.msk_dir / f"{stem}.png"

        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image unreadable for id='{stem}' at {img_p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        msk = np.array(Image.open(msk_p), dtype=np.uint8)
        if msk.ndim == 3 and msk.shape[-1] == 3:
            msk = msk[..., 0]

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            if img.max() > 1.0:
                img = img / 255.0
        else:
            img = img.float()

        if isinstance(msk, np.ndarray):
            if msk.ndim == 3 and msk.shape[-1] == 1:
                msk = msk[..., 0]
            msk = torch.from_numpy(msk.astype(np.int64, copy=False))
        else:
            msk = msk.long().squeeze()
        return img, msk


class _PersonPartDataset(Dataset):
    """
    Minimal dataset for PASCAL-Person-Part-like layout:
      root/JPEGImages/<id>.jpg
      root/<mask_dir>/<id>.png  (indices: 0..C-1, 255=ignore)
    """
    def __init__(self, root: str | Path, mask_dir_name: str, ids: Optional[List[str]] = None, transform=None):
        self.root = Path(root)
        self.img_dir = self.root / "JPEGImages"
        self.msk_dir = self.root / mask_dir_name
        self.transform = transform

        if ids is None:
            # intersection of images that have masks
            img_stems = {p.stem for p in self.img_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")}
            mask_stems = {p.stem for p in self.msk_dir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")}
            self.ids = sorted(img_stems & mask_stems)
        else:
            self.ids = list(ids)
        if not self.ids:
            raise RuntimeError(f"No (image, mask) pairs found under {self.root} with masks in {self.msk_dir.name}")

    def __len__(self): return len(self.ids)

    def _img_path(self, stem: str) -> Path:
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.img_dir / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found for id='{stem}' under {self.img_dir}")

    def _msk_path(self, stem: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg"):
            p = self.msk_dir / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Mask not found for id='{stem}' under {self.msk_dir}")

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        img_p = self._img_path(stem)
        msk_p = self._msk_path(stem)

        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image unreadable for id='{stem}' at {img_p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        msk = np.array(Image.open(msk_p), dtype=np.uint8)
        if msk.ndim == 3 and msk.shape[-1] == 3:
            msk = msk[..., 0]

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            if img.max() > 1.0:
                img = img / 255.0
        else:
            img = img.float()

        if isinstance(msk, np.ndarray):
            if msk.ndim == 3 and msk.shape[-1] == 1:
                msk = msk[..., 0]
            msk = torch.from_numpy(msk.astype(np.int64, copy=False))
        else:
            msk = msk.long().squeeze()
        return img, msk


# -------- extra metric helpers (no external deps) --------

def _accumulate_cm(cm: torch.Tensor, preds: torch.Tensor, target: torch.Tensor,
                   num_classes: int, ignore_index: Optional[int]):
    """Accumulate confusion matrix for multiclass."""
    if ignore_index is not None:
        mask = (target != ignore_index)
    else:
        mask = torch.ones_like(target, dtype=torch.bool)
    if mask.sum() == 0:
        return cm
    p = preds[mask].view(-1)
    t = target[mask].view(-1)
    k = int(num_classes)
    idx = t * k + p
    cm = cm + torch.bincount(idx, minlength=k * k).reshape(k, k).to(cm.device)
    return cm

def _scores_from_cm(cm: torch.Tensor):
    """Return per-class IoU, mIoU (extra), PixelAcc, mAcc, FWIoU."""
    eps = 1e-7
    cm = cm.float()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou_c = tp / (tp + fp + fn + eps)
    miou  = iou_c.mean().item()
    pixel_acc = tp.sum().item() / (cm.sum().item() + eps)
    acc_c = tp / (tp + fn + eps)  # per-class recall
    macc  = acc_c.mean().item()
    freq  = cm.sum(1) / (cm.sum() + eps)
    fwiou = (freq * iou_c).sum().item()
    return {
        "per_class_iou": iou_c.cpu().tolist(),
        "mIoU_extra": miou,
        "PixelAcc": pixel_acc,
        "mAcc": macc,
        "FWIoU": fwiou,
    }

def _update_binary_counts(state, preds: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int]):
    """Accumulate TP/TN/FP/FN for binary after thresholding."""
    if ignore_index is not None:
        mask = (target != ignore_index)
        preds = preds[mask]
        target = target[mask]
    preds = preds.view(-1)
    target = target.view(-1)
    state["tp"] += int(((preds == 1) & (target == 1)).sum().item())
    state["tn"] += int(((preds == 0) & (target == 0)).sum().item())
    state["fp"] += int(((preds == 1) & (target == 0)).sum().item())
    state["fn"] += int(((preds == 0) & (target == 1)).sum().item())

def _binary_from_counts(state):
    """Compute Dice/IoU/Precision/Recall/Specificity from accumulated counts."""
    tp, tn, fp, fn = [float(state[k]) for k in ("tp", "tn", "fp", "fn")]
    eps = 1e-7
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou  = tp / (tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    return {"Dice": dice, "IoU": iou, "Precision": prec, "Recall": rec, "Specificity": spec}

# -------- per-image helpers (NEW) --------

def _get_sample_id(dataset, idx: int) -> str:
    """Return a stable filename stem for sample idx."""
    try:
        if hasattr(dataset, "ids"):
            return str(dataset.ids[idx])
    except Exception:
        pass
    return f"{idx:06d}"

def _per_image_binary_dice(pred: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = None) -> float:
    """pred/target: HxW (0/1)."""
    if ignore_index is not None:
        valid = (target != ignore_index)
        pred = pred[valid]
        target = target[valid]
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    return (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)

def _per_image_miou_multiclass(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: Optional[int]) -> float:
    """Mean IoU over classes present in GT (VOC/Parts)."""
    if ignore_index is not None:
        valid = (target != ignore_index)
        pred = pred[valid]
        target = target[valid]
    ious = []
    for c in range(int(num_classes)):
        gt_c = (target == c)
        if not gt_c.any():
            continue  # skip classes not present in GT
        pr_c = (pred == c)
        inter = torch.logical_and(gt_c, pr_c).sum().item()
        union = torch.logical_or(gt_c, pr_c).sum().item()
        ious.append(inter / max(union, 1))
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))

# ------------------------------- color helpers (NEW) -------------------------------

def voc_colormap(N: int = 256) -> np.ndarray:
    """Standard PASCAL VOC colormap. Returns [N,3] uint8 RGB."""
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap

def binary_colormap(bg=(0, 0, 0), fg=(220, 20, 60)) -> np.ndarray:
    """2-class palette (0=bg, 1=fg) padded to 256 entries."""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    cmap[0] = np.array(bg, dtype=np.uint8)
    cmap[1] = np.array(fg, dtype=np.uint8)
    return cmap

def colorize_mask(mask_idx: np.ndarray, cmap: np.ndarray, ignore_index: int | None) -> np.ndarray:
    """mask_idx: [H,W] int (0..K-1, maybe ignore) -> [H,W,3] uint8 RGB."""
    m = mask_idx.astype(np.int64, copy=False)
    m_clip = np.clip(m, 0, cmap.shape[0]-1)
    rgb = cmap[m_clip]
    if ignore_index is not None:
        rgb[m == ignore_index] = (0, 0, 0)  # keep ignore black (customize if needed)
    return rgb

def tensor_image_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """img_t: [3,H,W] -> [H,W,3] uint8 RGB. Assumes 0..1 or 0..255 range (no de-norm here)."""
    x = img_t.detach().cpu()
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Unexpected image tensor shape {tuple(x.shape)} (expect [3,H,W])")
    x = x.clamp(0, 1) if x.max() <= 1.0 + 1e-6 else (x / 255.0).clamp(0, 1)
    x = (x * 255.0 + 0.5).to(torch.uint8)
    return x.permute(1, 2, 0).numpy()

def overlay_rgb(image_rgb: np.ndarray, mask_rgb: np.ndarray,
                mask_idx: np.ndarray | None = None, ignore_index: int | None = None,
                alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend mask over image. Only blends non-ignore pixels if mask_idx provided."""
    img = image_rgb.astype(np.float32)
    msk = mask_rgb.astype(np.float32)
    if mask_idx is not None and ignore_index is not None:
        valid = (mask_idx != ignore_index)
    else:
        valid = np.ones(image_rgb.shape[:2], dtype=bool)
    out = img.copy()
    out[valid] = (1 - alpha) * img[valid] + alpha * msk[valid]
    return np.clip(out, 0, 255).astype(np.uint8)

# ------------------------------- original-file helpers (NEW) -------------------------------

IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
MSK_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _first_existing(base: Path, subdirs: Tuple[str, ...], stem: str, exts: Tuple[str, ...]) -> Optional[Path]:
    for sd in subdirs:
        d = base / sd
        for ext in exts:
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None

def _find_original_paths(ds, gid: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Best-effort resolution of ORIGINAL (pre-transform) image/mask files for the given sample id.
    Works for: VOC official/slim, flat images/masks folders, Person-Part-style, Kvasir-SEG and most FolderSeg.
    """
    # Try to infer a filesystem base close to the dataset's root
    base = None
    # Prefer an attribute named 'root' (KvasirSeg/VOCSeg/FolderSeg/PascalParts) or Path-likes in custom classes
    if hasattr(ds, "root"):
        try:
            base = Path(ds.root)
        except Exception:
            pass
    if base is None:
        # Some composite datasets may tuck the real dataset inside .dataset
        if hasattr(ds, "dataset") and hasattr(ds.dataset, "root"):
            try:
                base = Path(ds.dataset.root)
            except Exception:
                pass
    if base is None:
        base = Path(".")

    # If dataset exposes path getters, use them first
    img_p = None
    if hasattr(ds, "_img_path"):
        try:
            p = ds._img_path(gid)
            img_p = Path(p) if p is not None else None
        except Exception:
            img_p = None
    if img_p is None:
        img_p = _first_existing(base, ("JPEGImages", "images", "img"), gid, IMG_EXTS)

    msk_p = None
    if hasattr(ds, "_msk_path"):
        try:
            p = ds._msk_path(gid)
            msk_p = Path(p) if p is not None else None
        except Exception:
            msk_p = None
    if msk_p is None:
        msk_p = _first_existing(base, ("SegmentationClass", "masks", "ann"), gid, MSK_EXTS)

    return img_p, msk_p


def _make_test_dataset(cfg) -> Tuple[torch.utils.data.Dataset, str, int]:
    """
    Build the test dataset and return (dataset, task, ignore_index).
    Task: 'binary' for Kvasir (num_classes==1), else 'multiclass'
    """
    ds = cfg["dataset"].lower()
    root = cfg["root"]
    img_size = int(cfg["image_size"])

    if ds == "kvasir":
        _, val_tf = make_transforms("medical", img_size)
        task, ignore = "binary", -100
        ids = _get_test_ids(cfg)
        if ids:
            return KvasirSeg(root, ids, transform=val_tf), task, ignore
        names = sorted(os.listdir(os.path.join(root, "images")))
        ids_all = [os.path.splitext(i)[0] for i in names]
        n_val = max(1, int(0.1 * len(ids_all))) if len(ids_all) > 1 else len(ids_all)
        return KvasirSeg(root, ids_all[-n_val:], transform=val_tf), task, ignore

    elif ds == "voc":
        _, val_tf = make_transforms("scene", img_size)
        task, ignore = "multiclass", VOC_IGNORE

        if _has_voc_slim_layout(root):
            ids = _get_test_ids(cfg)
            if not ids:
                val_txt = os.path.join(root, "ImageSets", "Segmentation", "val.txt")
                if not os.path.isfile(val_txt):
                    raise FileNotFoundError(f"VOC slim layout detected but missing {val_txt}")
                ids = _read_txt_ids(val_txt)
            ds_slim = _VOCDatasetSlim(root, ids=ids, transform=val_tf)
            return ds_slim, task, ignore

        if _has_voc_split_layout(root):
            use_test = (
                os.path.isdir(os.path.join(root, "test", "img"))
                and os.path.isdir(os.path.join(root, "test", "ann"))
                and len(os.listdir(os.path.join(root, "test", "img"))) > 0
                and len(os.listdir(os.path.join(root, "test", "ann"))) > 0
            )
            sub = "test" if use_test else "val"
            return FolderSeg(os.path.join(root, sub), ids=None, transform=val_tf), task, ignore

        if _flat_layout_exists(root):
            ids = _get_test_ids(cfg)
            if ids:
                return FolderSeg(root, ids, transform=val_tf), task, ignore
            names = sorted(os.listdir(os.path.join(root, "images")))
            ids_all = [os.path.splitext(i)[0] for i in names]
            n_val = max(1, int(0.1 * len(ids_all))) if len(ids_all) > 1 else len(ids_all)
            return FolderSeg(root, ids_all[-n_val:], transform=val_tf), task, ignore

        return VOCSeg(root, "val", transform=val_tf), task, ignore

    elif ds == "parts":
        _, val_tf = make_transforms("parts", img_size)
        task, ignore = "multiclass", int(cfg.get("ignore_index", 255))

        # Prefer person-part layout if present (your case)
        mask_dir = _parts_layout_mask_dir(root)
        if mask_dir is not None:
            # Priority: manifest 'test'/'val' -> splits/val.txt -> derive from masks
            ids = _get_test_ids(cfg)
            if not ids:
                split_txt = os.path.join(root, "splits", "val.txt")
                if os.path.isfile(split_txt):
                    ids = _read_txt_ids(split_txt)
            if not ids:
                # derive from mask dir to guarantee GT
                ids = []
                for p in sorted((Path(root) / mask_dir).iterdir()):
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        stem = p.stem
                        if (Path(root) / "JPEGImages" / f"{stem}.jpg").exists() or (Path(root) / "JPEGImages" / f"{stem}.png").exists():
                            ids.append(stem)
            return _PersonPartDataset(root, mask_dir, ids=ids, transform=val_tf), task, ignore

        # Flat images/masks fallback (other datasets)
        if _flat_layout_exists(root):
            ids = _get_test_ids(cfg)
            if ids:
                return FolderSeg(root, ids, transform=val_tf), task, ignore
            names = sorted(os.listdir(os.path.join(root, "images")))
            ids_all = [os.path.splitext(i)[0] for i in names]
            n_val = max(1, int(0.1 * len(ids_all))) if len(ids_all) > 1 else len(ids_all)
            return FolderSeg(root, ids_all[-n_val:], transform=val_tf), task, ignore

        # Final fallback: legacy PascalParts class (expects ImageSets/Part/*)
        return PascalParts(root, "val", transform=val_tf, ignore_index=ignore), task, ignore

    else:
        raise ValueError(f"Unknown dataset: {ds}")

def _build_model(cfg) -> SMPWrapper:
    return SMPWrapper(
        cfg["family"], cfg["encoder"], cfg["num_classes"],
        pretrained=False,  # weights come from checkpoint
        **cfg.get("model_kwargs", {})
    )

# ------------------------------- checkpoint helpers -------------------------------

def _best_from_last_state(ckpt_dir: str) -> Optional[str]:
    """
    Read 'last.ckpt' and try to recover 'best_model_path' from the ModelCheckpoint callback state.
    Works even when metrics.csv doesn't log a 'step' for epoch-level metrics.
    """
    last = os.path.join(ckpt_dir, "last.ckpt")
    if not os.path.exists(last):
        return None
    try:
        state = torch.load(last, map_location="cpu")
    except Exception:
        return None
    cbs = state.get("callbacks", {})
    if isinstance(cbs, dict):
        for _, cb_state in cbs.items():
            if isinstance(cb_state, dict) and "best_model_path" in cb_state:
                p = cb_state.get("best_model_path")
                if isinstance(p, str) and os.path.exists(p) and p.endswith(".ckpt"):
                    return p
    return None

def _load_ckpt_path_from_logs(out_dir: str, monitor: str, ckpt_dir: str) -> Optional[str]:
    """
    Find best checkpoint by scanning the latest CSV logger's metrics.csv
    for the best 'monitor' value, then matching a ckpt file containing that step in its name.
    Falls back to newest ckpt if step mapping fails.
    """
    csv_root = Path(out_dir) / "csv"
    if not csv_root.exists():
        return None
    vers = sorted(csv_root.glob("version_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not vers:
        return None
    metrics_csv = vers[0] / "metrics.csv"
    if not metrics_csv.exists():
        return None

    best_row = None
    with open(metrics_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(monitor, "")
            if val in ("", "nan", "NaN", None):
                continue
            try:
                v = float(val)
            except Exception:
                continue
            if (best_row is None) or (v > float(best_row[monitor])):
                best_row = row

    if best_row is None:
        return None

    step_str = best_row.get("step") or best_row.get("global_step")
    if step_str:
        try:
            step = int(step_str)
            pat = os.path.join(ckpt_dir, f"*step{step:06d}.ckpt")
            matches = glob.glob(pat)
            if matches:
                return matches[0]
        except Exception:
            pass

    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
    return all_ckpts[0] if all_ckpts else None

def _find_checkpoint(out_dir: str, monitor: str, policy: str = "best") -> Optional[str]:
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    if policy == "best":
        p = _best_from_last_state(ckpt_dir)
        if p:
            return p
        p = _load_ckpt_path_from_logs(out_dir, monitor, ckpt_dir)
        if p:
            return p
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
    return all_ckpts[0] if all_ckpts else None


# ------------------------------- evaluation -------------------------------

@torch.no_grad()
def evaluate(cfg_path: str, ckpt: Optional[str] = None, ckpt_policy: str = "best",
             tta: bool = False, threshold: float = 0.5,
             dump_dir: Optional[str] = None, save_preds: bool = False, save_gts: bool = False,
             topk: int = 0, save_color: bool = False, save_overlay: bool = False,
             save_orig: bool = False) -> Dict[str, float]:
    cfg = yaml.safe_load(open(cfg_path))
    out_dir = cfg.get("out_dir", "runs")
    seed = int(cfg.get("seed", os.environ.get("SEED", 0)))
    pl.seed_everything(seed, workers=True)

    ds_test, task, ignore_index = _make_test_dataset(cfg)
    loader = DataLoader(ds_test, batch_size=int(cfg.get("batch_size", 8)),
                        shuffle=False, num_workers=int(cfg.get("workers", 4)), pin_memory=True)

    # metrics (primary)
    if task == "binary":
        metric = DiceScore(num_classes=2, include_background=False, average="macro")
        metric_name = "test/dice"
    else:
        metric = tm.JaccardIndex(task="multiclass",
                                 num_classes=int(cfg["num_classes"]),
                                 ignore_index=ignore_index,
                                 average="macro")
        metric_name = "test/miou"

    # model
    net = _build_model(cfg)

    # ckpt
    monitor = cfg.get("monitor") or ("val/dice" if cfg["num_classes"] == 1 else "val/miou")
    if ckpt is None:
        ckpt = _find_checkpoint(out_dir, monitor, policy=ckpt_policy)
    if ckpt is None or (not os.path.exists(ckpt)):
        raise FileNotFoundError(f"Could not find a checkpoint. Looked in: {out_dir}/checkpoints. "
                                f"Pass --ckpt explicitly or ensure training produced checkpoints.")
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("state_dict", state)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("net."):
            new_sd[k.replace("net.", "", 1)] = v
        elif k.startswith("model.") or k.startswith("module."):
            new_sd[k.split(".", 1)[1]] = v
        else:
            new_sd[k] = v
    missing, unexpected = net.load_state_dict(new_sd, strict=False)
    if unexpected:
        print(f"[WARN] Unexpected keys in state_dict (ignored): {unexpected}")
    if missing:
        print(f"[WARN] Missing keys when loading state_dict: {missing}")

    device = "cuda" if torch.cuda.is_available() and int(cfg.get("gpus", 0)) > 0 else "cpu"
    net = net.to(device)
    net.eval()
    metric = metric.to(device)

    # accumulators for extra metrics
    cm = None  # for multiclass
    bin_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}  # for binary

    # per-image accumulators (NEW)
    per_ids: List[str] = []
    per_scores: List[float] = []
    seen = 0

    # optional dump dirs (NEW)
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        pred_dir = os.path.join(dump_dir, "preds")
        gt_dir   = os.path.join(dump_dir, "gts")
        if save_preds: os.makedirs(pred_dir, exist_ok=True)
        if save_gts:   os.makedirs(gt_dir, exist_ok=True)

        # NEW: color + overlay directories
        pred_rgb_dir = os.path.join(dump_dir, "preds_rgb") if save_color else None
        gt_rgb_dir   = os.path.join(dump_dir, "gts_rgb")   if save_color else None
        ovl_dir      = os.path.join(dump_dir, "overlays")  if save_overlay else None
        if pred_rgb_dir: os.makedirs(pred_rgb_dir, exist_ok=True)
        if gt_rgb_dir:   os.makedirs(gt_rgb_dir, exist_ok=True)
        if ovl_dir:      os.makedirs(ovl_dir, exist_ok=True)

        # NEW: original copies
        orig_img_dir = os.path.join(dump_dir, "images_orig") if save_orig else None
        orig_gt_dir  = os.path.join(dump_dir, "gts_orig")    if save_orig else None
        orig_gt_rgb_dir = os.path.join(dump_dir, "gts_rgb_orig") if (save_orig and save_color) else None
        if orig_img_dir: os.makedirs(orig_img_dir, exist_ok=True)
        if orig_gt_dir:  os.makedirs(orig_gt_dir,  exist_ok=True)
        if orig_gt_rgb_dir: os.makedirs(orig_gt_rgb_dir, exist_ok=True)
    else:
        pred_dir = gt_dir = pred_rgb_dir = gt_rgb_dir = ovl_dir = None
        orig_img_dir = orig_gt_dir = orig_gt_rgb_dir = None

    # NEW: choose colormap for saving
    if task == "binary":
        cmap = binary_colormap()     # 0=bg (black), 1=fg (crimson)
        ignore_for_color = None
    else:
        cmap = voc_colormap(256)     # VOC/Parts palette
        ignore_for_color = ignore_index if isinstance(ignore_index, int) else None

    use_amp = (device == "cuda")

    # --- NEW: reset peak memory stats before timing ---
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    t0 = time.time()
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            if not tta:
                logits = net(x)
            else:
                logits0 = net(x)
                logits1 = torch.flip(net(torch.flip(x, dims=[-1])), dims=[-1])                   # h
                logits2 = torch.flip(net(torch.flip(x, dims=[-2])), dims=[-2])                   # v
                logits3 = torch.flip(net(torch.flip(torch.flip(x, [-1]), [-2])), dims=[-1, -2])  # hv
                logits = (logits0 + logits1 + logits2 + logits3) / 4.0

        if task == "binary":
            preds = (torch.sigmoid(logits) > threshold).long().squeeze(1)
            metric.update(preds, y)
            _update_binary_counts(bin_counts, preds, y, ignore_index=None)  # Kvasir has no ignore
        else:
            preds = logits.argmax(dim=1)
            metric.update(preds, y)
            if cm is None:
                cm = torch.zeros(int(cfg["num_classes"]), int(cfg["num_classes"]),
                                 device=preds.device, dtype=torch.long)
            cm = _accumulate_cm(cm, preds, y, int(cfg["num_classes"]), ignore_index)

        # --- per-image scores + optional dumps (NEW) ---
        bsz = x.shape[0]
        for j in range(bsz):
            gid = _get_sample_id(loader.dataset, seen + j)
            if task == "binary":
                pi = preds[j].detach().cpu().to(torch.int64)
                gi = y[j].detach().cpu().to(torch.int64)
                score_ij = _per_image_binary_dice(pi, gi, ignore_index=None)
            else:
                pi = preds[j].detach().cpu().to(torch.int64)
                gi = y[j].detach().cpu().to(torch.int64)
                score_ij = _per_image_miou_multiclass(pi, gi, int(cfg["num_classes"]), ignore_index)

            per_ids.append(gid)
            per_scores.append(float(score_ij))

            if dump_dir:
                # raw (indexed) masks (existing behavior)
                if save_preds and pred_dir:
                    Image.fromarray(pi.numpy().astype(np.uint8)).save(os.path.join(pred_dir, f"{gid}.png"))
                if save_gts and gt_dir:
                    Image.fromarray(gi.numpy().astype(np.uint8)).save(os.path.join(gt_dir, f"{gid}.png"))

                # colorized RGB masks (NEW)
                if save_color:
                    pi_np = pi.numpy()
                    gi_np = gi.numpy()
                    pred_rgb = colorize_mask(pi_np, cmap, ignore_for_color)
                    gt_rgb   = colorize_mask(gi_np, cmap, ignore_for_color)
                    if pred_rgb_dir:
                        Image.fromarray(pred_rgb).save(os.path.join(pred_rgb_dir, f"{gid}.png"))
                    if gt_rgb_dir:
                        Image.fromarray(gt_rgb).save(os.path.join(gt_rgb_dir, f"{gid}.png"))

                # overlay on input image using PRED mask (NEW)
                if save_overlay and ovl_dir:
                    # reconstruct uint8 RGB image from input tensor
                    img_u8 = tensor_image_to_uint8(x[j].detach().cpu())
                    # ensure we have pred_rgb available even if --save_color is off
                    if not save_color:
                        pred_rgb = colorize_mask(pi.numpy(), cmap, ignore_for_color)
                    ov = overlay_rgb(img_u8, pred_rgb, mask_idx=pi.numpy(),
                                     ignore_index=ignore_for_color, alpha=0.5)
                    Image.fromarray(ov).save(os.path.join(ovl_dir, f"{gid}.png"))

                # copy ORIGINAL (pre-transform) files + optional colorized original GT
                if save_orig:
                    img_p, msk_p = _find_original_paths(loader.dataset, gid)

                    # Copy original image (or fallback to saving the transformed tensor)
                    if img_p is not None and orig_img_dir:
                        try:
                            shutil.copy2(str(img_p), os.path.join(orig_img_dir, img_p.name))
                        except Exception:
                            img_u8 = tensor_image_to_uint8(x[j].detach().cpu())
                            Image.fromarray(img_u8).save(os.path.join(orig_img_dir, f"{gid}.png"))
                    elif orig_img_dir:
                        img_u8 = tensor_image_to_uint8(x[j].detach().cpu())
                        Image.fromarray(img_u8).save(os.path.join(orig_img_dir, f"{gid}.png"))

                    # Copy original GT (if resolvable)
                    if msk_p is not None and orig_gt_dir:
                        try:
                            shutil.copy2(str(msk_p), os.path.join(orig_gt_dir, msk_p.name))
                        except Exception:
                            pass

                        # Also colorize the ORIGINAL GT at native resolution (if requested)
                        if save_color and orig_gt_rgb_dir:
                            try:
                                gt_orig = np.array(Image.open(msk_p), dtype=np.uint8)
                                if gt_orig.ndim == 3 and gt_orig.shape[-1] == 3:
                                    gt_orig = gt_orig[..., 0]
                                if task == "binary":
                                    # robust: many medical GTs are {0,255}
                                    gt_idx = (gt_orig > 0).astype(np.uint8)
                                    gt_rgb_orig = colorize_mask(gt_idx, binary_colormap(), None)
                                else:
                                    gt_rgb_orig = colorize_mask(gt_orig, cmap, ignore_for_color)
                                Image.fromarray(gt_rgb_orig).save(os.path.join(orig_gt_rgb_dir, f"{gid}.png"))
                            except Exception:
                                pass

        seen += bsz

    elapsed = time.time() - t0

    # --- NEW: latency + throughput + peak memory stats ---
    n_test = len(loader.dataset)
    latency_per_image_s = elapsed / max(n_test, 1)
    latency_ms = latency_per_image_s * 1000.0
    throughput = n_test / elapsed if elapsed > 0 else 0.0
    peak_mem_gb = None
    if device == "cuda":
        try:
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mem_gb = peak_bytes / (1024 ** 3)
        except Exception:
            peak_mem_gb = None

    score = float(metric.compute().item())

    # ---- extra metrics (JSON + console only; CSV kept unchanged to avoid header drift) ----
    extra_out = {}
    per_class_iou = None
    if task == "binary":
        b = _binary_from_counts(bin_counts)
        extra_out = {
            "test/iou": b["IoU"],
            "test/precision": b["Precision"],
            "test/recall": b["Recall"],
            "test/specificity": b["Specificity"],
        }
    else:
        if cm is None:
            cm = torch.zeros(int(cfg["num_classes"]), int(cfg["num_classes"]))
        extras = _scores_from_cm(cm)
        extra_out = {
            "test/pixel_acc": extras["PixelAcc"],
            "test/macc": extras["mAcc"],
            "test/fwiou": extras["FWIoU"],
        }
        per_class_iou = extras["per_class_iou"]

    # write outputs
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "eval_test_metrics.json")
    out_csv  = os.path.join(out_dir, "metrics_test.csv")
    payload = {
        "config": cfg_path,
        "checkpoint": ckpt,
        "dataset": cfg["dataset"],
        "root": cfg["root"],
        "split": cfg.get("split_manifest") or cfg.get("split_file") or "builtin_val_as_test",
        "task": task,
        "metric": metric_name,
        "score": score,
        "batch_size": int(cfg.get("batch_size", 8)),
        "workers": int(cfg.get("workers", 4)),
        "num_classes": int(cfg["num_classes"]),
        "encoder": cfg["encoder"],
        "family": cfg["family"],
        "image_size": int(cfg["image_size"]),
        "seed": seed,
        "eval_wall_clock_s": elapsed,
        "tta": bool(tta),
        "threshold": float(threshold) if task == "binary" else None,
    }

    # JSON payload can include extras (and per-class IoU list)
    payload_json = dict(payload)
    payload_json.update(extra_out)
    if per_class_iou is not None:
        payload_json["per_class_iou"] = per_class_iou

    # NEW: add latency + throughput + peak mem to JSON only
    payload_json["n_test_samples"] = n_test
    payload_json["eval_wall_clock_s_per_image"] = latency_per_image_s
    payload_json["latency_ms_per_image"] = latency_ms
    payload_json["throughput_img_per_s"] = throughput
    if peak_mem_gb is not None:
        payload_json["peak_mem_gb"] = peak_mem_gb

    try:
        if os.path.exists(out_json):
            arr = json.load(open(out_json, "r"))
            if isinstance(arr, list):
                arr.append(payload_json)
            else:
                arr = [arr, payload_json]
        else:
            arr = [payload_json]
        with open(out_json, "w") as f:
            json.dump(arr, f, indent=2)
    except Exception:
        pass

    # CSV remains with original columns only (no extras) to avoid header mismatch
    write_header = (not os.path.exists(out_csv))
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(payload.keys()))
        if write_header:
            w.writeheader()
        w.writerow(payload)

    # per-image outputs (NEW; optional)
    if dump_dir and per_ids:
        per_csv = os.path.join(dump_dir, "per_image_metrics.csv")
        with open(per_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", metric_name])
            for _id, sc in zip(per_ids, per_scores):
                w.writerow([_id, f"{sc:.6f}"])

        if topk and topk > 0:
            order = np.argsort(per_scores)  # ascending
            bot_idx = order[:min(topk, len(order))]
            top_idx = order[::-1][:min(topk, len(order))]
            with open(os.path.join(dump_dir, "bottomk.txt"), "w") as f:
                for i in bot_idx:
                    f.write(f"{per_ids[i]}\t{per_scores[i]:.6f}\n")
            with open(os.path.join(dump_dir, "topk.txt"), "w") as f:
                for i in top_idx:
                    f.write(f"{per_ids[i]}\t{per_scores[i]:.6f}\n")

    # Console summary (keep original + append brief extras + new efficiency info)
    if task == "binary":
        extras_print = f" | iou={extra_out['test/iou']:.4f} prec={extra_out['test/precision']:.4f} rec={extra_out['test/recall']:.4f}"
    else:
        extras_print = f" | pixAcc={extra_out['test/pixel_acc']:.4f} mAcc={extra_out['test/macc']:.4f} FWIoU={extra_out['test/fwiou']:.4f}"

    if peak_mem_gb is not None:
        speed_str = f" | N_test={n_test} | time={elapsed:.2f}s (~{latency_ms:.2f} ms/img, {throughput:.1f} img/s) | peak_mem={peak_mem_gb:.2f} GB | tta={tta}"
    else:
        speed_str = f" | N_test={n_test} | time={elapsed:.2f}s (~{latency_ms:.2f} ms/img, {throughput:.1f} img/s) | tta={tta}"

    print(f"[OK] {metric_name} = {score:.4f}{extras_print}{speed_str}")

    return {metric_name: score}


def main():
    import argparse
    ap = argparse.ArgumentParser("Evaluate a trained model on the test set (or val if test is unavailable).")
    ap.add_argument("--cfg", required=True, help="Path to a training YAML used to produce the run.")
    ap.add_argument("--ckpt", default=None, help="Path to a checkpoint to evaluate (overrides policy).")
    ap.add_argument("--ckpt_policy", default="best", choices=["best", "last"],
                    help="How to choose checkpoint if --ckpt is not provided.")
    ap.add_argument("--tta", action="store_true", help="Enable simple flip TTA (h/v/hv).")
    ap.add_argument("--threshold", type=float, default=0.5, help="Binary threshold (default 0.5).")

    # NEW (optional; backward-compatible)
    ap.add_argument("--dump_dir", default=None, help="If set, save per-image metrics (CSV) and optional masks here.")
    ap.add_argument("--save_preds", action="store_true", help="With --dump_dir, save predicted masks as indexed PNGs.")
    ap.add_argument("--save_gts", action="store_true", help="With --dump_dir, also save GT masks.")
    ap.add_argument("--topk", type=int, default=0, help="If >0, also write topk.txt and bottomk.txt by per-image score.")
    ap.add_argument("--save_color", action="store_true",
                    help="With --dump_dir, also save colorized RGB masks (preds_rgb/, gts_rgb/).")
    ap.add_argument("--save_overlay", action="store_true",
                    help="With --dump_dir, also save image overlays with prediction mask (overlays/).")
    ap.add_argument("--save_orig", action="store_true",
                    help="With --dump_dir, copy ORIGINAL images to images_orig/ and ORIGINAL GTs to gts_orig/. If --save_color is also on, save colorized original GTs to gts_rgb_orig/.")

    args = ap.parse_args()
    evaluate(args.cfg, ckpt=args.ckpt, ckpt_policy=args.ckpt_policy,
             tta=args.tta, threshold=args.threshold,
             dump_dir=args.dump_dir, save_preds=args.save_preds, save_gts=args.save_gts,
             topk=args.topk, save_color=args.save_color, save_overlay=args.save_overlay,
             save_orig=args.save_orig)


if __name__ == "__main__":
    main()
