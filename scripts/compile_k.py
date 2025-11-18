#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, os, sys, shutil
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm")
MSK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _read_rank_list(txt_path: Path) -> List[Tuple[str, float]]:
    out = []
    with open(txt_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = ln.split()
            gid = parts[0]
            score = float(parts[1]) if len(parts) > 1 else float("nan")
            out.append((gid, score))
    return out

def _read_per_image_csv(csv_path: Path) -> List[Tuple[str, float]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # prefer explicit columns if present
        metric_cols = [c for c in reader.fieldnames if c and c.startswith("test/")]
        for row in reader:
            gid = (row.get("id") or "").strip()
            if not gid: continue
            score = float("nan")
            for c in ("test/miou", "test/dice", *metric_cols):
                v = row.get(c, "")
                if v not in ("", "nan", "NaN", None):
                    try:
                        score = float(v)
                        break
                    except: pass
            rows.append((gid, score))
    return rows

def _gather_ids(dump_dir: Path, which: str, k: int) -> List[Tuple[str, float]]:
    txt = dump_dir / (f"{which}.txt")
    per = dump_dir / "per_image_metrics.csv"
    if txt.exists():
        pairs = _read_rank_list(txt)
        if which == "topk":
            return pairs[:k] if k > 0 else pairs
        else:
            return pairs[:k] if k > 0 else pairs
    elif per.exists():
        pairs = _read_per_image_csv(per)
        reverse = (which == "topk")
        pairs.sort(key=lambda t: (t[1], t[0]), reverse=reverse)
        return pairs[:k] if k > 0 else pairs
    return []

def _find_by_stem(dirpath: Path, stem: str, exts) -> Optional[Path]:
    if not dirpath.exists(): return None
    for ext in exts:
        p = dirpath / f"{stem}{ext}"
        if p.exists(): return p
    # fall back: strict stem match
    for p in dirpath.iterdir():
        if p.is_file() and p.stem == stem: return p
    return None

def _choose_orig_img(dump_dir: Path, gid: str) -> Optional[Path]:
    return _find_by_stem(dump_dir / "images_orig", gid, IMG_EXTS)

def _choose_orig_gt(dump_dir: Path, gid: str) -> Optional[Path]:
    return _find_by_stem(dump_dir / "gts_orig", gid, MSK_EXTS)

def _choose_gt_rgb(dump_dir: Path, gid: str) -> Optional[Path]:
    p = dump_dir / "gts_rgb_orig" / f"{gid}.png"
    if p.exists(): return p
    p2 = dump_dir / "gts_rgb" / f"{gid}.png"
    return p2 if p2.exists() else None

def _choose_pred_rgb(dump_dir: Path, gid: str) -> Optional[Path]:
    p = dump_dir / "preds_rgb" / f"{gid}.png"
    return p if p.exists() else None

def _validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()  # quick integrity check
        # re-open to load fully
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False

def _copy(src: Path, dst: Path):
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def _reencode_img_to_png(src: Path, dst_png: Path):
    """Re-encode *image* (RGB-like) to PNG RGB."""
    _ensure_dir(dst_png.parent)
    with Image.open(src) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.save(dst_png, format="PNG", compress_level=3)

def _reencode_mask_preserving_index(src: Path, dst_png: Path):
    """
    Re-encode *mask* to PNG while preserving indices (P or L).
    If palette exists, keep it. Otherwise save as L (8-bit).
    """
    _ensure_dir(dst_png.parent)
    with Image.open(src) as im:
        if im.mode == "P":
            # keep original palette
            im.save(dst_png, format="PNG", compress_level=3)
        elif im.mode == "L":
            im.save(dst_png, format="PNG", compress_level=3)
        else:
            # Try to reduce to single-channel if possible
            im = im.convert("L")
            im.save(dst_png, format="PNG", compress_level=3)

def _place_file(kind: str, src: Optional[Path], dst_folder: Path,
                reencode: bool, is_mask: bool) -> Optional[Path]:
    """
    Place file under dst_folder. If reencode=True -> write standardized PNG names:
      image.png / gt.png / gt_rgb.png / pred_rgb.png
    Otherwise copy original filename.
    Returns final path or None.
    """
    if src is None: return None
    name_map = {"image": "image.png", "gt": "gt.png",
                "gt_rgb": "gt_rgb.png", "pred_rgb": "pred_rgb.png"}
    if reencode:
        dst = dst_folder / name_map[kind]
        try:
            if is_mask:
                _reencode_mask_preserving_index(src, dst)
            else:
                _reencode_img_to_png(src, dst)
        except Exception as e:
            print(f"[WARN] Re-encode failed for {src} -> {e}; falling back to copy.", file=sys.stderr)
            dst = dst_folder / (src.name if src.suffix else f"{name_map[kind]}")
            _copy(src, dst)
    else:
        dst = dst_folder / src.name
        _copy(src, dst)

    if not _validate_image(dst):
        # final attempt: force re-encode to a fresh PNG
        try:
            if is_mask:
                _reencode_mask_preserving_index(src, dst_folder / name_map[kind])
                dst = dst_folder / name_map[kind]
            else:
                _reencode_img_to_png(src, dst_folder / name_map[kind])
                dst = dst_folder / name_map[kind]
        except Exception as e:
            print(f"[ERROR] Could not produce a valid file for {kind} ({src}). {e}", file=sys.stderr)
            return None
    return dst

def _compile_set(dump_dir: Path, out_dir: Path, which: str, k: int, reencode: bool) -> int:
    pairs = _gather_ids(dump_dir, which, k)
    if not pairs:
        print(f"[WARN] No {which} items found in {dump_dir}.", file=sys.stderr)
        return 0

    rows = []
    root = out_dir / which
    _ensure_dir(root)

    n_ok = 0
    for gid, score in pairs:
        dst = root / gid
        _ensure_dir(dst)

        img_src = _choose_orig_img(dump_dir, gid)
        gt_src = _choose_orig_gt(dump_dir, gid)
        gt_rgb_src = _choose_gt_rgb(dump_dir, gid)
        pred_rgb_src = _choose_pred_rgb(dump_dir, gid)

        img_dst = _place_file("image", img_src, dst, reencode, is_mask=False)
        gt_dst = _place_file("gt", gt_src, dst, reencode, is_mask=True)
        gt_rgb_dst = _place_file("gt_rgb", gt_rgb_src, dst, reencode, is_mask=False)
        pred_rgb_dst = _place_file("pred_rgb", pred_rgb_src, dst, reencode, is_mask=False)

        rows.append({
            "id": gid,
            "score": f"{score:.6f}" if score == score else "",
            "image": str(img_dst.relative_to(out_dir)) if img_dst else "",
            "gt": str(gt_dst.relative_to(out_dir)) if gt_dst else "",
            "gt_rgb": str(gt_rgb_dst.relative_to(out_dir)) if gt_rgb_dst else "",
            "pred_rgb": str(pred_rgb_dst.relative_to(out_dir)) if pred_rgb_dst else "",
        })
        n_ok += 1

    # write summary CSV
    csv_path = out_dir / f"summary_{which}.csv"
    _ensure_dir(csv_path.parent)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "score", "image", "gt", "gt_rgb", "pred_rgb"])
        w.writeheader()
        for r in rows: w.writerow(r)

    print(f"[OK] Compiled {n_ok} items to {root}")
    return n_ok

def main():
    ap = argparse.ArgumentParser("Compile Top-K/Bottom-K examples with robust copying and optional re-encode.")
    ap.add_argument("--dump_dir", required=True, help="Directory from eval_test.py (has per_image_metrics.csv, images_orig/, gts_orig/, preds_rgb/, ...)")
    ap.add_argument("--out_dir", required=True, help="Destination folder (e.g., runs/<exp>/compiled_k_fixed)")
    ap.add_argument("--k", type=int, default=0, help="How many from each set (0=all)")
    ap.add_argument("--which", choices=["both", "topk", "bottomk"], default="both")
    ap.add_argument("--reencode", action="store_true",
                    help="Re-encode outputs to PNG (image.png, gt.png, gt_rgb.png, pred_rgb.png). Preserves mask indexing.")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)

    total = 0
    if args.which in ("both", "topk"):
        total += _compile_set(dump_dir, out_dir, "topk", args.k, args.reencode)
    if args.which in ("both", "bottomk"):
        total += _compile_set(dump_dir, out_dir, "bottomk", args.k, args.reencode)

    if total == 0:
        print("[NOTE] Nothing compiled. Ensure your eval dump_dir contains files (use --save_orig/--save_color in eval).")

if __name__ == "__main__":
    main()
