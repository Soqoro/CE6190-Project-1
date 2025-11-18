#!/usr/bin/env python3
from __future__ import annotations
import os, json, base64, argparse, zlib, logging
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

IGNORE = 255
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# -------------------- logging helpers --------------------

def setup_logging(verbose: bool, quiet: bool):
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s"
    )


# -------------------- class discovery --------------------

def load_classes(meta_path: Path) -> dict:
    try:
        m = json.load(open(meta_path, 'r'))
        items = m.get('classes', [])
        if isinstance(items, list) and items:
            classes = {c['title']: i+1 for i, c in enumerate(items) if c.get('title')}
            logging.info("Loaded %d classes from %s", len(classes), meta_path)
            return classes
    except Exception as e:
        logging.debug("Could not load classes from %s: %s", meta_path, e)
    return {}

def infer_classes(ann_files: list[str], neutral_names=('neutral','background','void','ignore')) -> dict:
    classes = set()
    for p in ann_files:
        try:
            data = json.load(open(p, 'r'))
            for obj in data.get('objects', []):
                t = (obj.get('classTitle') or '').strip()
                if t:
                    classes.add(t)
        except Exception as e:
            logging.debug("Class scan skip %s: %s", p, e)
            continue
    classes = [c for c in sorted(classes) if c.lower() not in neutral_names]
    cm = {c: i+1 for i, c in enumerate(classes)}  # 0 reserved for background
    logging.info("Inferred %d classes: %s", len(cm), ", ".join(cm.keys()) or "(none)")
    return cm


# -------------------- bitmap decoding --------------------

def _open_image_bytes(b: bytes):
    try:
        return Image.open(BytesIO(b)).convert('L')
    except Exception:
        return None

def decode_bitmap(b64_blob: str) -> tuple[np.ndarray, str]:
    """
    Robust decode:
    - strip data URL header if present
    - try base64->PNG
    - else try base64->zlib.decompress->PNG
    Returns (boolean mask, decode_mode_str).
    """
    if not b64_blob:
        raise ValueError("empty bitmap.data")
    if b64_blob.startswith("data:"):
        b64_blob = b64_blob.split(",", 1)[1]

    raw = base64.b64decode(b64_blob)

    # 1) try as PNG
    im = _open_image_bytes(raw)
    if im is not None:
        arr = np.array(im)
        return (arr > 0), "png"

    # 2) try zlib-compressed payload -> PNG
    raw2 = None
    try:
        raw2 = zlib.decompress(raw)
        im2 = _open_image_bytes(raw2)
        if im2 is not None:
            arr = np.array(im2)
            return (arr > 0), "zlib+png"
    except Exception:
        pass

    raise ValueError("unsupported bitmap.data encoding (not PNG; zlib+PNG also failed)")


# -------------------- optional polygon support --------------------

def rasterize_polygon(poly_pts, H, W):
    """Handle polygon objects if present."""
    mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask)
    if isinstance(poly_pts, dict) and 'exterior' in poly_pts:
        pts = [(float(x), float(y)) for x, y in poly_pts['exterior']]
    else:
        pts = [(float(x), float(y)) for x, y in poly_pts]
    if len(pts) >= 3:
        draw.polygon(pts, outline=1, fill=1)
    return np.array(mask, dtype=bool)


# -------------------- conversion core --------------------

def convert_split(split_dir: Path, class_map: dict, neutral_to_ignore=True, progress_every=100, verbose=False):
    img_dir = split_dir / 'img'
    ann_dir = split_dir / 'ann'
    if not ann_dir.exists():
        logging.warning("Skip split '%s' — missing %s", split_dir.name, ann_dir)
        return

    json_files = sorted([p for p in ann_dir.glob('*.json')])
    logging.info("[%s] Found %d annotation JSONs under %s", split_dir.name, len(json_files), ann_dir)
    if not json_files:
        return

    n_ok, n_fail = 0, 0
    n_objs_total, n_bitmap, n_poly, n_ignored = 0, 0, 0, 0
    decode_modes = {"png": 0, "zlib+png": 0}

    for idx, jpath in enumerate(json_files, 1):
        try:
            data = json.load(open(jpath, 'r'))
            H = int(data['size']['height'])
            W = int(data['size']['width'])
            mask = np.zeros((H, W), dtype=np.uint8)

            obj_count = 0
            for obj in data.get('objects', []):
                obj_count += 1
                n_objs_total += 1
                title = (obj.get('classTitle') or '').strip()
                gt = obj.get('geometryType')

                is_neutral = title.lower() in ('neutral','void','ignore','background')
                cls_id = IGNORE if (neutral_to_ignore and is_neutral) else np.uint8(class_map.get(title, 0))
                if is_neutral:
                    n_ignored += 1

                if gt == 'bitmap':
                    b = obj.get('bitmap', {})
                    blob = b.get('data') or ''
                    if not blob and b.get('url'):
                        logging.debug("%s: bitmap.url present but no inline data; skipping object", jpath.name)
                        continue
                    try:
                        m, mode = decode_bitmap(blob)  # boolean
                        decode_modes[mode] = decode_modes.get(mode, 0) + 1
                    except Exception as e:
                        logging.debug("%s: bitmap decode failed: %s", jpath.name, e)
                        continue

                    ox, oy = map(int, b.get('origin', [0, 0]))
                    h, w = m.shape
                    y1, y2 = max(0, oy), min(H, oy + h)
                    x1, x2 = max(0, ox), min(W, ox + w)
                    if y2 > y1 and x2 > x1:
                        patch = m[: (y2 - y1), : (x2 - x1)]
                        if cls_id == IGNORE:
                            mask[y1:y2, x1:x2][patch] = IGNORE
                        else:
                            mask[y1:y2, x1:x2][patch] = cls_id
                    n_bitmap += 1
                    if verbose:
                        logging.debug("  %s: obj #%d '%s' bitmap origin=(%d,%d) %dx%d mode=%s",
                                      jpath.name, obj_count, title, ox, oy, w, h, mode)

                elif gt in ('polygon', 'polyline'):
                    pts = obj.get('points')
                    if pts:
                        pmask = rasterize_polygon(pts, H, W)
                        if cls_id == IGNORE:
                            mask[pmask] = IGNORE
                        else:
                            mask[pmask] = cls_id
                        n_poly += 1
                        if verbose:
                            logging.debug("  %s: obj #%d '%s' polygon", jpath.name, obj_count, title)
                else:
                    # unsupported geometry types can be added here
                    if verbose:
                        logging.debug("  %s: obj #%d '%s' geometryType=%s (ignored)", jpath.name, obj_count, title, gt)

            out_png = ann_dir / (jpath.stem.replace('.jpg', '') + '.png')
            Image.fromarray(mask).save(out_png)
            n_ok += 1

        except Exception as e:
            n_fail += 1
            logging.warning("failed %s: %s", jpath, e)

        if progress_every > 0 and idx % progress_every == 0:
            logging.info("[%s] progress %d/%d | ok=%d fail=%d | objs=%d (bitmap=%d, poly=%d, neutral/ignored=%d)",
                         split_dir.name, idx, len(json_files), n_ok, n_fail, n_objs_total, n_bitmap, n_poly, n_ignored)

    # Final summary
    logging.info("[%s] wrote %d PNG masks to %s (failed %d)", split_dir.name, n_ok, ann_dir, n_fail)
    logging.info("[%s] objects: total=%d | bitmap=%d | polygon=%d | neutral/ignored=%d | decode_modes=%s",
                 split_dir.name, n_objs_total, n_bitmap, n_poly, n_ignored, decode_modes)


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser("Convert Supervisely-style .jpg.json bitmaps to semantic PNG masks (with logs)")
    ap.add_argument('--root', required=True, help="dataset root, e.g., data/voc")
    ap.add_argument('--neutral_as_ignore', action='store_true',
                    help="map 'neutral'/'void'/'ignore' to 255 (ignore); default is False")
    ap.add_argument('--progress_every', type=int, default=100,
                    help="log progress every N files (default 100; set 0 to disable)")
    ap.add_argument('--verbose', action='store_true', help="verbose (DEBUG) logging")
    ap.add_argument('--quiet', action='store_true', help="warnings only")
    args = ap.parse_args()

    setup_logging(args.verbose, args.quiet)

    root = Path(args.root)
    splits = [s for s in ('train', 'val', 'test') if (root / s / 'ann').exists()]
    if not splits:
        raise SystemExit(f"No split folders with ann/ under {root}")

    meta = root / 'meta.json'
    if meta.exists():
        class_map = load_classes(meta)
        if not class_map:
            logging.warning("meta.json present but no classes parsed; will infer from data")
    else:
        all_json = []
        for sp in splits:
            all_json += [str(p) for p in (root / sp / 'ann').glob('*.json')]
        class_map = infer_classes(all_json)

    logging.info("Class map: %s", class_map or "(empty; all foreground will map to 0/background)")
    for sp in splits:
        logging.info("=== Converting split: %s ===", sp)
        convert_split(root / sp, class_map,
                      neutral_to_ignore=args.neutral_as_ignore,
                      progress_every=args.progress_every,
                      verbose=args.verbose)

    logging.info("Done.")

if __name__ == "__main__":
    main()
