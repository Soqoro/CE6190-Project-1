#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse, random
from pathlib import Path
from typing import List, Dict, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png"}
MSK_EXTS = {".png", ".jpg", ".jpeg"}

# --------------------------- helpers ---------------------------

def _list_stems(dirpath: Path, exts: set[str]) -> List[str]:
    if not dirpath.exists():
        return []
    stems = []
    for p in sorted(dirpath.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            stems.append(p.stem)
    return stems

def _has_any_mask(msk_dir: Path, stem: str) -> bool:
    return any((msk_dir / f"{stem}{e}").exists() for e in MSK_EXTS)

def collect_ids_flat(root: Path) -> List[str]:
    """
    Flat layout:
      root/images/*.jpg|*.png
      root/masks/*.png|*.jpg
    """
    img_dir = root / "images"
    msk_dir = root / "masks"
    if not img_dir.exists() or not msk_dir.exists():
        print(f"[WARN] Skipping {root} — missing 'images/' or 'masks/'")
        return []
    ids = []
    for stem in _list_stems(img_dir, IMG_EXTS):
        if _has_any_mask(msk_dir, stem):
            ids.append(stem)
        else:
            print(f"[WARN] No mask for '{stem}' in {msk_dir}")
    if not ids:
        print(f"[WARN] No valid pairs in {root}")
    return ids

def split_80_10_10(ids: List[str], seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    ids_shuf = ids[:]
    rng.shuffle(ids_shuf)
    n = len(ids_shuf)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    n_train = int(round(0.8 * n))
    n_val   = int(round(0.1 * n))
    n_train = min(n_train, n)
    n_val   = min(n_val, n - n_train)
    train = sorted(ids_shuf[:n_train])
    val   = sorted(ids_shuf[n_train:n_train+n_val])
    test  = sorted(ids_shuf[n_train+n_val:])
    return {"train": train, "val": val, "test": test}

def _split_layout_exists(ds_root: Path) -> bool:
    # e.g., data/<ds>/{train,val}/{img,ann}
    return all((ds_root / s / "img").exists() and (ds_root / s / "ann").exists() for s in ("train", "val"))

def _read_lines(p: Path) -> List[str]:
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]

# --------------------------- VOC official helpers ---------------------------

def _voc_official_layout(ds_root: Path) -> bool:
    """Detect official VOC layout at ds_root: JPEGImages/, SegmentationClass/, ImageSets/Segmentation/"""
    return (
        (ds_root / "JPEGImages").exists() and
        (ds_root / "SegmentationClass").exists() and
        (ds_root / "ImageSets" / "Segmentation").exists()
    )

def _voc_filter_existing(ds_root: Path, ids: List[str]) -> List[str]:
    """Keep only ids that have an image and a segmentation mask on disk."""
    keep: List[str] = []
    img_dir = ds_root / "JPEGImages"
    msk_dir = ds_root / "SegmentationClass"
    for stem in ids:
        img_ok = any((img_dir / f"{stem}{e}").exists() for e in IMG_EXTS)
        msk_ok = (msk_dir / f"{stem}.png").exists()  # VOC masks are png
        if img_ok and msk_ok:
            keep.append(stem)
        else:
            if not img_ok:
                print(f"[WARN] VOC missing image for id '{stem}'")
            if not msk_ok:
                print(f"[WARN] VOC missing mask (SegmentationClass/{stem}.png)")
    return sorted(keep)

def collect_ids_voc_official(ds_root: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Use official ImageSets lists. Test masks are not publicly available;
    we will use val as test if test list is absent or unusable.
    """
    seg_sets = ds_root / "ImageSets" / "Segmentation"
    train = _read_lines(seg_sets / "train.txt")
    val   = _read_lines(seg_sets / "val.txt")
    test  = _read_lines(seg_sets / "test.txt")  # might not exist

    if not train or not val:
        print(f"[WARN] VOC ImageSets/Segmentation train.txt or val.txt missing under {seg_sets}")
        return [], [], []

    train = _voc_filter_existing(ds_root, train)
    val   = _voc_filter_existing(ds_root, val)
    if test:
        # keep only those that actually have masks (often they don't)
        test_filtered = _voc_filter_existing(ds_root, test)
        if not test_filtered:
            print("[INFO] VOC test.txt present but no GT masks found; using val as test.")
            test = val[:]
        else:
            test = test_filtered
    else:
        print("[INFO] VOC test.txt not found; using val as test.")
        test = val[:]

    return train, val, test

# --------------------------- split (train/val/img/ann) helpers ---------------------------

def collect_ids_split(ds_root: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Split layout:
      ds_root/train/{img,ann}
      ds_root/val/{img,ann}
      ds_root/test/{img,ann}     (optional; if missing we reuse val as test)
    """
    if not _split_layout_exists(ds_root):
        return [], [], []
    def _pair_ids(split: str) -> List[str]:
        img_dir = ds_root / split / "img"
        msk_dir = ds_root / split / "ann"
        ids = []
        for stem in _list_stems(img_dir, IMG_EXTS):
            if _has_any_mask(msk_dir, stem):
                ids.append(stem)
            else:
                print(f"[WARN] No mask for '{stem}' in {msk_dir} (split={split})")
        return sorted(ids)

    train_ids = _pair_ids("train")
    val_ids   = _pair_ids("val")
    test_ids  = _pair_ids("test") if (ds_root / "test").exists() else val_ids[:]  # fallback
    return train_ids, val_ids, test_ids

# --------------------------- person-part helpers (NEW) ---------------------------

def _find_pp_mask_dir(ds_root: Path) -> Path | None:
    for name in ("pascal_person_parts_gt", "pascal_person_part_gt", "PartMasks7"):
        p = ds_root / name
        if p.exists():
            return p
    return None

def collect_ids_person_part_from_lists(ds_root: Path, seed: int, val2test_pct: float) -> Dict[str, List[str]]:
    """
    Expect:
      ds_root/JPEGImages/
      ds_root/<mask_dir>/         (e.g., pascal_person_parts_gt)
      ds_root/splits/{train.txt, val.txt}   # stems
    We split val deterministically into (val, test).
    """
    img_dir = ds_root / "JPEGImages"
    mask_dir = _find_pp_mask_dir(ds_root)
    splits_dir = ds_root / "splits"
    if not (img_dir.exists() and mask_dir and (splits_dir / "train.txt").exists() and (splits_dir / "val.txt").exists()):
        return {}  # not a person-part layout

    def _filter_existing(stems: List[str]) -> List[str]:
        keep = []
        for s in stems:
            img_ok = any((img_dir / f"{s}{e}").exists() for e in IMG_EXTS)
            msk_ok = any((mask_dir / f"{s}{e}").exists() for e in MSK_EXTS)
            if img_ok and msk_ok:
                keep.append(s)
            else:
                if not img_ok: print(f"[WARN] parts missing image '{s}'")
                if not msk_ok: print(f"[WARN] parts missing mask  '{s}'")
        return sorted(keep)

    train_ids = _filter_existing(_read_lines(splits_dir / "train.txt"))
    val_ids   = _filter_existing(_read_lines(splits_dir / "val.txt"))

    # val -> (val, test)
    rng = random.Random(seed)
    perm = val_ids[:]
    rng.shuffle(perm)
    k_test = int(round(len(perm) * (val2test_pct / 100.0)))
    test_ids = sorted(perm[:k_test]) if k_test > 0 else []
    val_ids  = sorted(perm[k_test:])
    if not test_ids:  # if k_test=0, reuse val as test
        test_ids = val_ids[:]

    return {"train": train_ids, "val": val_ids, "test": test_ids}

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(
        "Make two-level (20% & 100%) manifests.\n"
        "- VOC (official layout): read ImageSets/Segmentation train/val[/test]. "
        "LOW reuses val/test and subsamples only train (default).\n"
        "- VOC (split layout): use existing train/val/test dirs. LOW subsamples train only (default).\n"
        "- Kvasir (flat): FULL 80/10/10; LOW reuses FULL val/test and subsamples train only (default).\n"
        "- parts (person-part): USE your splits/train.txt & val.txt; split val->(val,test)."
    )
    ap.add_argument("--data_root", default="data", help="Folder with voc/, kvasir/, parts/")
    ap.add_argument("--datasets", nargs="+", default=["voc", "kvasir", "parts"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--low_pct", type=float, default=20.0, help="LOW percent of FULL-TRAIN (or each split if --low_mode=train_val_test)")
    ap.add_argument("--out_root", default="splits")
    # NEW knobs (defaults preserve old behavior)
    ap.add_argument("--low_mode", choices=["train", "train_val_test"], default="train",
                    help="Subsample only train (default) or train+val+test")
    ap.add_argument("--parts_val2test_pct", type=float, default=50.0,
                    help="From parts/val.txt, % to move into test (default 50/50)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    processed_any = False

    for ds in args.datasets:
        ds_root = data_root / ds
        if not ds_root.exists():
            print(f"[WARN] Skipping '{ds}' — not found: {ds_root}")
            continue

        ds_out = out_root / ds
        ds_out.mkdir(parents=True, exist_ok=True)

        # ---------------- VOC official (unchanged) ----------------
        if ds.lower() == "voc":
            if _voc_official_layout(ds_root):
                tr_ids, va_ids, te_ids = collect_ids_voc_official(ds_root)
                if not (tr_ids and va_ids):
                    print(f"[WARN] VOC official layout lists are missing or empty under {ds_root}")
                    continue

                full_split = {"train": tr_ids, "val": va_ids, "test": te_ids}

                rng = random.Random(args.seed)
                k_low = max(1, int(round(len(tr_ids) * args.low_pct / 100.0)))
                low_train = sorted(tr_ids if k_low >= len(tr_ids) else rng.sample(tr_ids, k_low))
                low_split = {"train": low_train,
                             "val": va_ids if args.low_mode == "train" else sorted(rng.sample(va_ids, max(1,int(round(len(va_ids)*args.low_pct/100.0))))) if va_ids else [],
                             "test": te_ids if args.low_mode == "train" else sorted(rng.sample(te_ids, max(1,int(round(len(te_ids)*args.low_pct/100.0))))) if te_ids else []}

                low_path  = ds_out / f"seed{args.seed}_20.json"   # keep filename for compatibility
                full_path = ds_out / f"seed{args.seed}_100.json"
                json.dump(low_split,  open(low_path, "w"))
                json.dump(full_split, open(full_path, "w"))

                print(f"[voc] official layout detected at {ds_root}")
                print(f"  FULL -> {full_path}  (train={len(tr_ids)}, val={len(va_ids)}, test={len(te_ids)})")
                print(f"  LOW  ({args.low_pct:.1f}% {args.low_mode}) -> {low_path}  "
                      f"(train={len(low_split['train'])}, val={len(low_split['val'])}, test={len(low_split['test'])})")
                processed_any = True
                continue

            # Else, support split layout (unchanged)
            if _split_layout_exists(ds_root):
                tr_ids, va_ids, te_ids = collect_ids_split(ds_root)
                if not (tr_ids and va_ids):
                    print(f"[WARN] VOC split folders missing or empty under {ds_root}")
                    continue

                full_split = {"train": tr_ids, "val": va_ids, "test": te_ids}
                rng = random.Random(args.seed)
                k_low = max(1, int(round(len(tr_ids) * args.low_pct / 100.0)))
                low_train = sorted(tr_ids if k_low >= len(tr_ids) else rng.sample(tr_ids, k_low))
                low_split = {"train": low_train,
                             "val": va_ids if args.low_mode == "train" else sorted(rng.sample(va_ids, max(1,int(round(len(va_ids)*args.low_pct/100.0))))) if va_ids else [],
                             "test": te_ids if args.low_mode == "train" else sorted(rng.sample(te_ids, max(1,int(round(len(te_ids)*args.low_pct/100.0))))) if te_ids else []}

                low_path  = ds_out / f"seed{args.seed}_20.json"
                full_path = ds_out / f"seed{args.seed}_100.json"
                json.dump(low_split,  open(low_path, "w"))
                json.dump(full_split, open(full_path, "w"))

                print(f"[voc] split layout detected at {ds_root}")
                print(f"  FULL -> {full_path}  (train={len(tr_ids)}, val={len(va_ids)}, test={len(te_ids)})")
                print(f"  LOW  ({args.low_pct:.1f}% {args.low_mode}) -> {low_path}  "
                      f"(train={len(low_split['train'])}, val={len(low_split['val'])}, test={len(low_split['test'])})")
                processed_any = True
                continue

            print(f"[WARN] '{ds_root}' does not look like VOC official or split layout; skipping.")
            continue

        # ---------------- person-part (NEW branch; only when structure matches) ----------------
        pp_full = collect_ids_person_part_from_lists(ds_root, seed=args.seed, val2test_pct=args.parts_val2test_pct)
        if pp_full:
            rng = random.Random(args.seed)
            def _sub(ids: List[str]) -> List[str]:
                if args.low_mode == "train":
                    return ids  # keep full unless it's train (handled below)
                k = max(1, int(round(len(ids) * args.low_pct / 100.0))) if ids else 0
                return sorted(ids if k >= len(ids) else rng.sample(ids, k))

            # full
            full_split = pp_full
            # low
            k_low = max(1, int(round(len(pp_full["train"]) * args.low_pct / 100.0)))
            low_train = sorted(pp_full["train"] if k_low >= len(pp_full["train"]) else rng.sample(pp_full["train"], k_low))
            low_split = {
                "train": low_train,
                "val":   pp_full["val"]  if args.low_mode == "train" else _sub(pp_full["val"]),
                "test":  pp_full["test"] if args.low_mode == "train" else _sub(pp_full["test"]),
            }

            low_path  = ds_out / f"seed{args.seed}_20.json"
            full_path = ds_out / f"seed{args.seed}_100.json"
            json.dump(low_split,  open(low_path, "w"))
            json.dump(full_split, open(full_path, "w"))

            print(f"[parts] lists detected at {ds_root}/splits (val→test = {args.parts_val2test_pct:.1f}%)")
            print(f"  FULL -> {full_path}  (train={len(full_split['train'])}, val={len(full_split['val'])}, test={len(full_split['test'])})")
            print(f"  LOW  ({args.low_pct:.1f}% {args.low_mode}) -> {low_path}  "
                  f"(train={len(low_split['train'])}, val={len(low_split['val'])}, test={len(low_split['test'])})")
            processed_any = True
            continue

        # ---------------- flat fallback (unchanged; e.g., kvasir) ----------------
        ids_all = collect_ids_flat(ds_root)
        if not ids_all:
            continue
        full_split = split_80_10_10(ids_all, seed=args.seed)
        rng = random.Random(args.seed)
        k_low = max(1, int(round(len(full_split["train"]) * args.low_pct / 100.0)))
        low_train = sorted(full_split["train"] if k_low >= len(full_split["train"])
                           else rng.sample(full_split["train"], k_low))
        low_split = {"train": low_train,
                     "val":   full_split["val"]  if args.low_mode == "train" else sorted(rng.sample(full_split["val"],  max(1,int(round(len(full_split['val'] )*args.low_pct/100.0))))) if full_split["val"]  else [],
                     "test":  full_split["test"] if args.low_mode == "train" else sorted(rng.sample(full_split["test"], max(1,int(round(len(full_split['test'])*args.low_pct/100.0))))) if full_split["test"] else []}

        low_path  = ds_out / f"seed{args.seed}_20.json"
        full_path = ds_out / f"seed{args.seed}_100.json"
        json.dump(low_split,  open(low_path, "w"))
        json.dump(full_split, open(full_path, "w"))

        print(f"[{ds}] flat layout at {ds_root}")
        print(f"  FULL (80/10/10) -> {full_path}  (train={len(full_split['train'])}, "
              f"val={len(full_split['val'])}, test={len(full_split['test'])})")
        print(f"  LOW  ({args.low_pct:.1f}% {args.low_mode}) -> {low_path}  "
              f"(train={len(low_split['train'])}, val={len(low_split['val'])}, test={len(low_split['test'])})")
        processed_any = True

    if not processed_any:
        print("[INFO] No datasets processed. Ensure at least one dataset exists with the expected structure.")

if __name__ == "__main__":
    main()
