#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, argparse, random, shutil
from pathlib import Path
from typing import List, Tuple

def find_pairs(img_dir: Path, msk_dir: Path) -> List[Tuple[Path, Path, str]]:
    """Return (img_path, mask_path, id) for files that exist in both folders."""
    imgs = {}
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file():
            imgs[p.stem] = p
    pairs = []
    for stem, img_p in imgs.items():
        # masks are typically PNG; try png then jpg as a fallback
        cand = msk_dir / f"{stem}.png"
        if not cand.exists():
            alt = msk_dir / f"{stem}.jpg"
            cand = alt if alt.exists() else cand
        if cand.exists():
            pairs.append((img_p, cand, stem))
        else:
            print(f"[WARN] Missing mask for {stem}, skipping.", file=sys.stderr)
    return pairs

def split_ids(ids: List[str], train: float, val: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    random.seed(seed)
    ids = ids[:]
    random.shuffle(ids)
    n = len(ids)
    n_train = int(round(train * n))
    n_val   = int(round(val * n))
    n_train = min(n_train, n)  # guard
    n_val   = min(max(n_val, 0), n - n_train)
    n_test  = n - n_train - n_val
    train_ids = ids[:n_train]
    val_ids   = ids[n_train:n_train+n_val]
    test_ids  = ids[n_train+n_val:]
    return train_ids, val_ids, test_ids

def place(pairs, out_root: Path, split_name: str, ids_set, mode: str):
    img_out = out_root / split_name / "images"
    msk_out = out_root / split_name / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_p, msk_p, stem in pairs:
        if stem not in ids_set:
            continue
        dst_img = img_out / img_p.name
        dst_msk = msk_out / msk_p.name

        if dst_img.exists() and dst_msk.exists():
            continue

        if mode == "symlink":
            try:
                os.symlink(os.path.relpath(img_p, img_out), dst_img)
                os.symlink(os.path.relpath(msk_p, msk_out), dst_msk)
            except OSError:
                # fallback to copy if symlink not permitted
                shutil.copy2(img_p, dst_img)
                shutil.copy2(msk_p, dst_msk)
        elif mode == "copy":
            shutil.copy2(img_p, dst_img)
            shutil.copy2(msk_p, dst_msk)
        elif mode == "move":
            shutil.move(str(img_p), str(dst_img))
            shutil.move(str(msk_p), str(dst_msk))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        copied += 1
    return copied

def main():
    ap = argparse.ArgumentParser(description="Split Kvasir images/masks into train/val/test folder structure.")
    ap.add_argument("--src", default="data/kvasir", help="Source folder with images/ and masks/")
    ap.add_argument("--out", default="data/kvasir_split", help="Output root folder")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["symlink","copy","move"], default="symlink",
                    help="How to materialize files into split folders (default: symlink)")
    ap.add_argument("--write_json", action="store_true",
                    help="Also write splits/kvasir/{train,val,test}.json with ID lists")
    args = ap.parse_args()

    if round(args.train + args.val + args.test, 6) != 1.0:
        ap.error("train + val + test must sum to 1.0")

    src = Path(args.src)
    img_dir = src / "images"
    msk_dir = src / "masks"
    if not img_dir.exists() or not msk_dir.exists():
        sys.exit(f"[ERR] Expected {img_dir} and {msk_dir} to exist.")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(img_dir, msk_dir)
    if not pairs:
        sys.exit("[ERR] No (image, mask) pairs found.")

    stems = [s for _, _, s in pairs]
    tr_ids, va_ids, te_ids = split_ids(stems, args.train, args.val, args.seed)

    # Place files
    n_tr = place(pairs, out_root, "train", set(tr_ids), args.mode)
    n_va = place(pairs, out_root, "val",   set(va_ids), args.mode)
    n_te = place(pairs, out_root, "test",  set(te_ids), args.mode)

    # Optionally write JSON id lists compatible with your training pipeline
    if args.write_json:
        split_dir = Path("splits/kvasir")
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / "train.json", "w") as f: json.dump(tr_ids, f)
        with open(split_dir / "val.json",   "w") as f: json.dump(va_ids, f)
        with open(split_dir / "test.json",  "w") as f: json.dump(te_ids, f)

    total = len(pairs)
    print(f"Done. Total pairs: {total}")
    print(f"  train: {n_tr}  val: {n_va}  test: {n_te}")
    print(f"Output root: {out_root.resolve()}")
    if args.write_json:
        print("ID lists written to splits/kvasir/{train,val,test}.json")

if __name__ == "__main__":
    main()
