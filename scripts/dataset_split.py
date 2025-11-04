#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse, random
from pathlib import Path
from typing import List, Dict

IMG_EXTS = {".jpg", ".jpeg", ".png"}
MSK_EXTS = {".png", ".jpg", ".jpeg"}

def collect_ids(root: Path) -> List[str]:
    """
    Collect image IDs (filestems) under a flat folder layout:
        root/
          images/*.jpg|*.png
          masks/*.png|*.jpg
    Returns [] if images/ or masks/ is missing (non-fatal).
    Skips pairs with a missing counterpart.
    """
    img_dir = root / "images"
    msk_dir = root / "masks"
    if not img_dir.exists() or not msk_dir.exists():
        print(f"[WARN] Skipping {root} — missing 'images/' or 'masks/'")
        return []

    stems: List[str] = []
    for p in sorted(img_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        stem = p.stem
        has_mask = any((msk_dir / f"{stem}{ext}").exists() for ext in MSK_EXTS)
        if has_mask:
            stems.append(stem)
        else:
            print(f"[WARN] No mask found for '{stem}' in {msk_dir}, skipping")
    if not stems:
        print(f"[WARN] No valid (image, mask) pairs found under {root}")
    return stems


def split_80_10_10(ids: List[str], seed: int) -> Dict[str, List[str]]:
    """
    Deterministic 80/10/10 split using a local RNG (no global state).
    Keeps ordering stable in the JSON by sorting each split.
    """
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
    train = ids_shuf[:n_train]
    val   = ids_shuf[n_train:n_train + n_val]
    test  = ids_shuf[n_train + n_val:]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def main():
    ap = argparse.ArgumentParser(
        "Make two-level (20% & 100%) splits for voc/kvasir/parts from flat folders: data/<ds>/{images,masks}. "
        "The 100% level uses an 80/10/10 split. The 20% level reuses the SAME val/test as the 100% level and "
        "subsamples only the train set to low_pct of the FULL-TRAIN pool."
    )
    ap.add_argument("--data_root", default="data",
                    help="Folder that contains voc/, kvasir/, parts/ subfolders")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--low_pct", type=float, default=20.0,
                    help="LOW budget percent of the FULL-TRAIN pool (default 20%%)")
    ap.add_argument("--datasets", nargs="+", default=["voc", "kvasir", "parts"],
                    help="Which dataset subfolders to process")
    ap.add_argument("--out_root", default="splits",
                    help="Where to write split manifests")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    processed_any = False

    for ds in args.datasets:
        ds_root = data_root / ds
        if not ds_root.exists():
            print(f"[WARN] Skipping '{ds}' — folder not found: {ds_root}")
            continue

        ids_all = collect_ids(ds_root)
        if not ids_all:
            # Either missing images/masks or no valid pairs; skip gracefully
            continue

        # ---- FULL 80/10/10 from ALL pairs (shared val/test) ----
        full_split = split_80_10_10(sorted(ids_all), seed=args.seed)

        # ---- LOW budget: sample ONLY from FULL-TRAIN, reuse FULL val/test ----
        full_train = full_split["train"]
        rng = random.Random(args.seed)
        k_low_train = max(1, int(round(len(full_train) * args.low_pct / 100.0)))
        if k_low_train >= len(full_train):
            low_train = sorted(full_train)
        else:
            low_train = sorted(rng.sample(full_train, k_low_train))

        low_split = {
            "train": low_train,
            "val":   full_split["val"],   # same validation as FULL
            "test":  full_split["test"],  # same test as FULL
        }

        # ---- Save ----
        ds_out = out_root / ds
        ds_out.mkdir(parents=True, exist_ok=True)

        low_path  = ds_out / f"seed{args.seed}_20.json"
        full_path = ds_out / f"seed{args.seed}_100.json"
        with open(low_path, "w") as f:
            json.dump(low_split, f)
        with open(full_path, "w") as f:
            json.dump(full_split, f)

        # ---- Report ----
        n_all = len(ids_all)
        print(f"[{ds}] total pairs: {n_all}")
        print(f"  FULL (100%)  -> {full_path}")
        print(f"     train={len(full_split['train'])}, val={len(full_split['val'])}, test={len(full_split['test'])}")
        print(f"  LOW  ({args.low_pct:.1f}% of FULL-TRAIN) -> {low_path}")
        print(f"     train={len(low_split['train'])} (subset of FULL-TRAIN), "
              f"val={len(low_split['val'])} (shared), test={len(low_split['test'])} (shared)")
        processed_any = True

    if not processed_any:
        print("[INFO] No datasets processed. Ensure at least one of "
              f"{args.datasets} exists under {data_root}/<ds>/{{images,masks}}.")


if __name__ == "__main__":
    main()
