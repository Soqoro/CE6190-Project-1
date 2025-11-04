from __future__ import annotations
import os, json, yaml, csv, time, glob
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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


# ------------------------------- helpers -------------------------------

def _flat_layout_exists(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "masks"))

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

def _get_test_ids(cfg) -> List[str]:
    """Return explicit test IDs if present, else fall back to 'val', else [] (handled upstream)."""
    sp_manifest = cfg.get("split_manifest") or cfg.get("split_file")
    if sp_manifest:
        data = _read_ids_or_manifest(sp_manifest)
        if isinstance(data, dict):
            test_ids = list(data.get("test", []))
            if test_ids:
                return test_ids
            # Fallback if no test present
            return list(data.get("val", []))
        # If it's a list, we have no notion of test; fallback to [] (caller will handle)
        return []
    # No split file provided -> VOC/Parts official or Kvasir folder; we'll fallback to 'val' split loaders.
    return []

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
        # fallback to 10% val split from folder if no manifest provided
        names = sorted(os.listdir(os.path.join(root, "images")))
        ids_all = [os.path.splitext(i)[0] for i in names]
        n_val = max(1, int(0.1 * len(ids_all))) if len(ids_all) > 1 else len(ids_all)
        return KvasirSeg(root, ids_all[-n_val:], transform=val_tf), task, ignore

    elif ds == "voc":
        _, val_tf = make_transforms("scene", img_size)
        task, ignore = "multiclass", VOC_IGNORE
        if _flat_layout_exists(root):
            ids = _get_test_ids(cfg)
            if ids:
                return FolderSeg(root, ids, transform=val_tf), task, ignore
            # fallback: last 10% as "test"
            names = sorted(os.listdir(os.path.join(root, "images")))
            ids_all = [os.path.splitext(i)[0] for i in names]
            n_val = max(1, int(0.1 * len(ids_all))) if len(ids_all) > 1 else len(ids_all)
            return FolderSeg(root, ids_all[-n_val:], transform=val_tf), task, ignore
        else:
            # Official VOC has no public test masks; we evaluate on the official val set.
            return VOCSeg(root, "val", transform=val_tf), task, ignore

    elif ds == "parts":
        _, val_tf = make_transforms("parts", img_size)
        task, ignore = "multiclass", int(cfg.get("ignore_index", 255))
        if _flat_layout_exists(root):
            ids = _get_test_ids(cfg)
            if ids:
                return FolderSeg(root, ids, transform=val_tf), task, ignore
            names = sorted(os.listdir(os.path.join(root, "images")))
            ids_all = [os.path.splitext(i)[0] for i in names]
            n_val = max(1, int(0.1 * len(ids_all))) if len(ids_all) > 1 else len(ids_all)
            return FolderSeg(root, ids_all[-n_val:], transform=val_tf), task, ignore
        else:
            # Pascal-Parts: use 'val' as evaluation set
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
    # pick newest version_* dir
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

    # Prefer a 'step' if present; else try to infer by newest ckpt
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

    # Fallback: newest ckpt in directory (often best if save_top_k tracks best)
    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
    return all_ckpts[0] if all_ckpts else None

def _find_checkpoint(out_dir: str, monitor: str, policy: str = "best") -> Optional[str]:
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    if policy == "best":
        # 1) Try reading 'best_model_path' embedded in last.ckpt callback state
        p = _best_from_last_state(ckpt_dir)
        if p:
            return p
        # 2) Try CSV logs → best step → filename match
        p = _load_ckpt_path_from_logs(out_dir, monitor, ckpt_dir)
        if p:
            return p
    # 3) Fall back to last or newest
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
    return all_ckpts[0] if all_ckpts else None


# ------------------------------- evaluation -------------------------------

@torch.no_grad()
def evaluate(cfg_path: str, ckpt: Optional[str] = None, ckpt_policy: str = "best",
             tta: bool = False, threshold: float = 0.5) -> Dict[str, float]:
    cfg = yaml.safe_load(open(cfg_path))
    out_dir = cfg.get("out_dir", "runs")
    seed = int(cfg.get("seed", os.environ.get("SEED", 0)))
    pl.seed_everything(seed, workers=True)

    ds_test, task, ignore_index = _make_test_dataset(cfg)
    loader = DataLoader(ds_test, batch_size=int(cfg.get("batch_size", 8)),
                        shuffle=False, num_workers=int(cfg.get("workers", 4)), pin_memory=True)

    # metrics
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
    # The Lightning ckpt stores the LightningModule's state_dict under 'state_dict'
    sd = state.get("state_dict", state)
    # Strip possible 'net.' prefix or Lit module prefixes
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("net."):
            new_sd[k.replace("net.", "", 1)] = v
        elif k.startswith("model.") or k.startswith("module."):
            new_sd[k.split(".", 1)[1]] = v
        else:
            new_sd[k] = v
    # Load into SMPWrapper (keys should now match)
    missing, unexpected = net.load_state_dict(new_sd, strict=False)
    if unexpected:
        print(f"[WARN] Unexpected keys in state_dict (ignored): {unexpected}")
    if missing:
        print(f"[WARN] Missing keys when loading state_dict: {missing}")

    device = "cuda" if torch.cuda.is_available() and int(cfg.get("gpus", 0)) > 0 else "cpu"
    net = net.to(device)
    net.eval()

    # inference (AMP if GPU)
    use_amp = (device == "cuda")
    t0 = time.time()
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            if not tta:
                logits = net(x)
            else:
                # simple flip TTA: original + hflip + vflip + hvflip (avg)
                logits0 = net(x)
                logits1 = torch.flip(net(torch.flip(x, dims=[-1])), dims=[-1])                # h
                logits2 = torch.flip(net(torch.flip(x, dims=[-2])), dims=[-2])                # v
                logits3 = torch.flip(net(torch.flip(torch.flip(x, [-1]), [-2])), dims=[-1, -2])  # hv
                logits = (logits0 + logits1 + logits2 + logits3) / 4.0

        if task == "binary":
            preds = (torch.sigmoid(logits) > threshold).long().squeeze(1)  # [B,H,W]
            metric.update(preds, y)
        else:
            preds = logits.argmax(dim=1)  # [B,H,W]
            metric.update(preds, y)
    elapsed = time.time() - t0
    score = float(metric.compute().item())

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
    # JSON (append as list)
    try:
        if os.path.exists(out_json):
            arr = json.load(open(out_json, "r"))
            if isinstance(arr, list):
                arr.append(payload)
            else:
                arr = [arr, payload]
        else:
            arr = [payload]
        with open(out_json, "w") as f:
            json.dump(arr, f, indent=2)
    except Exception:
        # never fail evaluation due to JSON write issues
        pass

    # CSV row
    write_header = (not os.path.exists(out_csv))
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(payload.keys()))
        if write_header:
            w.writeheader()
        w.writerow(payload)

    print(f"[OK] {metric_name} = {score:.4f} | ckpt={ckpt} | N_test={len(loader.dataset)} | time={elapsed:.2f}s | tta={tta}")
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
    args = ap.parse_args()
    evaluate(args.cfg, ckpt=args.ckpt, ckpt_policy=args.ckpt_policy,
             tta=args.tta, threshold=args.threshold)


if __name__ == "__main__":
    main()
