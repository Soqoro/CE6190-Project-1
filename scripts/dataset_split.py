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
        # find any mask with the same stem and allowed extension
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
        "Make two-level (20% & 100%) 80/10/10 splits for voc/kvasir/parts "
        "from flat folders: data/<ds>/{images,masks}"
    )
    ap.add_argument("--data_root", default="data",
                    help="Folder that contains voc/, kvasir/, parts/ subfolders")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--low_pct", type=float, default=20.0,
                    help="LOW budget percent (default 20%% of available pairs)")
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

        n_all = len(ids_all)
        k_low = max(1, int(round(n_all * args.low_pct / 100.0)))
        rng = random.Random(args.seed)
        low_ids = sorted(rng.sample(ids_all, k_low)) if k_low < n_all else sorted(ids_all)
        full_ids = sorted(ids_all)

        low_split  = split_80_10_10(low_ids, seed=args.seed)
        full_split = split_80_10_10(full_ids, seed=args.seed)

        ds_out = out_root / ds
        ds_out.mkdir(parents=True, exist_ok=True)

        low_path  = ds_out / f"seed{args.seed}_20.json"
        full_path = ds_out / f"seed{args.seed}_100.json"
        with open(low_path, "w") as f:
            json.dump(low_split, f)
        with open(full_path, "w") as f:
            json.dump(full_split, f)

        print(f"[{ds}] total pairs: {n_all}")
        print(f"  -> {low_path}  (LOW {args.low_pct:.1f}%: {len(low_ids)} ids) "
              f"[train {len(low_split['train'])}, val {len(low_split['val'])}, test {len(low_split['test'])}]")
        print(f"  -> {full_path} (FULL 100%: {len(full_ids)} ids) "
              f"[train {len(full_split['train'])}, val {len(full_split['val'])}, test {len(full_split['test'])}]")
        processed_any = True

    if not processed_any:
        print("[INFO] No datasets processed. Ensure at least one of "
              f"{args.datasets} exists under {data_root}/<ds>/{{images,masks}}.")


if __name__ == "__main__":
    main()
