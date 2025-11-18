from __future__ import annotations
import os, yaml, time, json, tempfile
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import torch

# -------- Matmul / TF32 settings (use legacy unified API only) --------
# Valid values: "highest" (default), "high", "medium"
# "high" enables TF32 on Ampere+ while staying compatible with Lightning checks.
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from src.data.transforms import make_transforms
from src.data.kvasir import KvasirSeg
from src.data.voc import VOCSeg, IGNORE_INDEX as VOC_IGNORE
from src.data.pascal_parts import PascalParts
from src.data.splits import load_ids
from src.models.smp_wrapper import SMPWrapper
from src.engine.lit_module import LitSeg
from src.data.folder_seg import FolderSeg  # generic flat images/masks loader

# ---- Checkpoint IO that avoids cross-device atomic moves (works on Drive) ----
try:
    from lightning_fabric.plugins import CheckpointIO as _CheckpointIOBase
except Exception:
    # very old PL versions
    from pytorch_lightning.plugins import CheckpointIO as _CheckpointIOBase  # type: ignore

class LocalCheckpointIO(_CheckpointIOBase):
    """Save to a tmp file inside the destination directory, then atomic replace."""
    def save_checkpoint(self, checkpoint, path: str, storage_options=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_dir = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, prefix=".tmp_ckpt_", suffix=".ckpt")
        os.close(fd)
        try:
            torch.save(checkpoint, tmp_path)
            # atomic within same filesystem; avoids cross-device link errors
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def load_checkpoint(self, path: str, storage_options=None):
        return torch.load(path, map_location="cpu")

    def remove_checkpoint(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _set_seed(seed: int):
    """One-seed-per-run policy with deterministic flags."""
    pl.seed_everything(int(seed), workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _dataloader(ds, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True
    )


class _RunStatsCallback(pl.callbacks.Callback):
    """
    Logs run-level stats for efficiency analysis.
    """
    def __init__(self, monitor: str, mode: str, img_size: int, batch_size: int):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.img_size = int(img_size)
        self.batch_size = int(batch_size)
        self._t0: Optional[float] = None
        self._best = None
        self._best_step = None
        self._best_epoch = None
        self._best_wall_s = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._t0 = time.time()
        # params
        params = sum(p.numel() for p in pl_module.parameters())
        trainer.logger.log_metrics({"params": float(params)}, step=trainer.global_step)
        # latency (rough)
        try:
            dev = pl_module.device
            x = torch.randn(1, 3, self.img_size, self.img_size, device=dev)
            iters, warmup = 30, 10
            if dev.type == "cuda":
                torch.cuda.synchronize()
            with torch.no_grad():
                for _ in range(warmup):
                    _ = pl_module.net(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                for _ in range(iters):
                    _ = pl_module.net(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t2 = time.time()
            latency_ms = (t2 - t1) / iters * 1000.0
        except Exception:
            latency_ms = float("nan")
        trainer.logger.log_metrics({"latency_ms": float(latency_ms)}, step=trainer.global_step)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.monitor in trainer.callback_metrics:
            cur = trainer.callback_metrics[self.monitor].item()
            step = trainer.global_step
            if self._best is None:
                self._best, self._best_step = cur, step
                self._best_epoch = trainer.current_epoch
                self._best_wall_s = (time.time() - (self._t0 or time.time()))
            else:
                better = (cur > self._best) if self.mode == "max" else (cur < self._best)
                if better:
                    self._best, self._best_step = cur, step
                    self._best_epoch = trainer.current_epoch
                    self._best_wall_s = (time.time() - (self._t0 or time.time()))

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        elapsed = time.time() - (self._t0 or time.time())
        steps_to_best = int(self._best_step or -1)
        epochs_to_best = float(self._best_epoch if self._best_epoch is not None else -1)
        wall_to_best = float(self._best_wall_s if self._best_wall_s is not None else float("nan"))

        best_metric = float(self._best) if self._best is not None else float("nan")
        denom_steps = max(steps_to_best, 1)
        denom_epochs = max(epochs_to_best, 1e-6)
        denom_secs = max(wall_to_best, 1e-6)

        metric_per_1k_steps = best_metric / (denom_steps / 1000.0)
        metric_per_epoch = best_metric / denom_epochs
        metric_per_sec = best_metric / denom_secs

        images_to_best = steps_to_best * max(self.batch_size, 1)

        trainer.logger.log_metrics(
            {
                "wall_clock_s": float(elapsed),
                "steps_to_best": steps_to_best,
                "epochs_to_best": epochs_to_best,
                "wall_to_best_s": wall_to_best,
                "images_to_best": int(images_to_best),
                "metric_per_1k_steps": float(metric_per_1k_steps),
                "metric_per_epoch": float(metric_per_epoch),
                "metric_per_sec": float(metric_per_sec),
            },
            step=trainer.global_step,
        )


class _ValDebugOnce(pl.callbacks.Callback):
    """Log target/pred histograms and background ratio on the first val batch."""
    def __init__(self, num_classes: int, ignore_index: int = 255):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self._done = False

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self._done:
            return
        x, y = batch
        x = x.to(pl_module.device, non_blocking=True)
        with torch.no_grad():
            logits = pl_module.net(x)
            preds = logits.argmax(dim=1).cpu()

        y = y.cpu()
        valid = (y != self.ignore_index)
        if valid.any():
            y_valid = y[valid]
            p_valid = preds[valid]
            t_hist = torch.bincount(y_valid.flatten(), minlength=self.num_classes)
            p_hist = torch.bincount(p_valid.flatten(), minlength=self.num_classes)
        else:
            t_hist = torch.zeros(self.num_classes, dtype=torch.long)
            p_hist = torch.zeros(self.num_classes, dtype=torch.long)

        bg_ratio = (preds == 0).float().mean().item()

        print(f"[DEBUG] target_hist (first {min(10,self.num_classes)}): {t_hist[:10].tolist()} ... sum={int(t_hist.sum())}")
        print(f"[DEBUG] pred_hist   (first {min(10,self.num_classes)}): {p_hist[:10].tolist()} ... sum={int(p_hist.sum())}")
        print(f"[DEBUG] pred background ratio: {bg_ratio:.3f}")

        try:
            trainer.logger.log_metrics({
                "debug/bg_ratio": bg_ratio,
                **{f"debug/t_hist/{i}": float(t_hist[i]) for i in range(min(self.num_classes, 5))},
                **{f"debug/p_hist/{i}": float(p_hist[i]) for i in range(min(self.num_classes, 5))},
            }, step=trainer.global_step)
        except Exception:
            pass

        self._done = True


def _flat_layout_exists(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "masks"))


def _voc_split_layout_exists(root: str) -> bool:
    # e.g., data/voc/{train,val}/{img,ann}
    return (
        os.path.isdir(os.path.join(root, "train", "img"))
        and os.path.isdir(os.path.join(root, "train", "ann"))
        and os.path.isdir(os.path.join(root, "val", "img"))
        and os.path.isdir(os.path.join(root, "val", "ann"))
    )


def _voc_slim_layout_exists(root: str) -> bool:
    # e.g., data/voc/{JPEGImages, SegmentationClass, ImageSets, Annotations}
    return (
        os.path.isdir(os.path.join(root, "JPEGImages"))
        and os.path.isdir(os.path.join(root, "SegmentationClass"))
    )

# ---- Person-Part layout (JPEGImages + person-part masks)
def _person_part_layout_exists(root: str) -> bool:
    if not os.path.isdir(os.path.join(root, "JPEGImages")):
        return False
    mask_dirs = (
        "pascal_person_parts_gt",
        "pascal_person_part_gt",
        "PartMasks7",
        "SegmentationPart",
    )
    return any(os.path.isdir(os.path.join(root, d)) for d in mask_dirs)


def _read_ids_or_manifest(path: str) -> Dict[str, List[str]] | List[str]:
    """
    Try to read a manifest (dict with train/val/test) first; if it's a plain list, return list.
    Falls back to load_ids() for legacy JSON arrays.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # normalize keys
        return {
            "train": list(data.get("train", [])),
            "val": list(data.get("val", [])),
            "test": list(data.get("test", [])),
        }
    if isinstance(data, list):
        return list(data)
    # last resort
    return load_ids(path)


def make_datasets(cfg):
    """
    Modes:
      - split_manifest: use explicit train/val/test ids (preferred).
      - split_file: can be EITHER a plain list (legacy) OR a manifest dict; both handled.
      - else: defaults (Kvasir 90/10; VOC/Parts official lists or inferred).
    """
    ds = cfg["dataset"].lower()
    root = cfg["root"]
    img_size = int(cfg["image_size"])
    seed = int(cfg.get("seed", 0))

    # Pick transforms
    if ds == "kvasir":
        train_tf, val_tf = make_transforms("medical", img_size)
    elif ds == "voc":
        train_tf, val_tf = make_transforms("scene", img_size)
    elif ds == "parts":
        train_tf, val_tf = make_transforms("parts", img_size)
    else:
        raise ValueError(f"Unknown dataset: {ds}")

    is_flat = _flat_layout_exists(root)
    voc_split_layout = (ds == "voc") and _voc_split_layout_exists(root)
    voc_slim_layout  = (ds == "voc") and _voc_slim_layout_exists(root)
    pp_layout        = (ds == "parts") and _person_part_layout_exists(root)

    if ds == "kvasir":
        loader_cls = KvasirSeg
        task, ignore = "binary", None
    elif ds == "voc":
        loader_cls = FolderSeg if (is_flat or voc_split_layout or voc_slim_layout) else VOCSeg
        task, ignore = "multiclass", VOC_IGNORE
    else:  # parts
        loader_cls = FolderSeg if (is_flat or pp_layout) else PascalParts
        task, ignore = "multiclass", int(cfg.get("ignore_index", 255))

    # ---------- Preferred: split_manifest ----------
    if cfg.get("split_manifest"):
        man = _read_ids_or_manifest(cfg["split_manifest"])
        if isinstance(man, list):
            ids = man
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(ids), generator=g).tolist()
            n_val = max(1, int(0.1 * len(ids)))
            train_ids = [ids[i] for i in perm[:-n_val]] if len(ids) > 1 else ids
            val_ids   = [ids[i] for i in perm[-n_val:]] if len(ids) > 1 else ids
        else:
            train_ids, val_ids = man["train"], man["val"]

        if ds == "voc" and voc_split_layout:
            tr = FolderSeg(os.path.join(root, "train"), ids=train_ids, transform=train_tf)
            va = FolderSeg(os.path.join(root, "val"),   ids=val_ids,   transform=val_tf)
        elif ds == "voc" and voc_slim_layout:
            print("[INFO] Using VOC slim layout with FolderSeg.")
            tr = FolderSeg(root, ids=train_ids, transform=train_tf)
            va = FolderSeg(root, ids=val_ids,   transform=val_tf)
        elif loader_cls in (FolderSeg, KvasirSeg):
            tr = loader_cls(root, train_ids, transform=train_tf)
            va = loader_cls(root, val_ids,   transform=val_tf)
        else:
            if ds == "voc":
                tr = VOCSeg(root, "train", ids_subset=train_ids, transform=train_tf)
                va = VOCSeg(root, "val", transform=val_tf)
            else:
                tr = PascalParts(root, "train", transform=train_tf, ignore_index=ignore)
                tr.ids = [i for i in tr.ids if i in set(train_ids)]
                va = PascalParts(root, "val", transform=val_tf, ignore_index=ignore)
        return tr, va, task, ignore

    # ---------- Legacy / flexible: split_file ----------
    if cfg.get("split_file"):
        sp = _read_ids_or_manifest(cfg["split_file"])

        if ds == "kvasir":
            if isinstance(sp, dict):
                train_ids, val_ids = sp.get("train", []), sp.get("val", [])
                if loader_cls is KvasirSeg:
                    tr = KvasirSeg(root, train_ids, transform=train_tf)
                    va = KvasirSeg(root, val_ids,   transform=val_tf)
                else:
                    tr = loader_cls(root, train_ids, transform=train_tf)
                    va = loader_cls(root, val_ids,   transform=val_tf)
                return tr, va, task, ignore
            else:
                ids = sp
                g = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(ids), generator=g).tolist()
                n_val = max(1, int(0.1 * len(ids)))
                train_ids = [ids[i] for i in perm[:-n_val]] if len(ids) > 1 else ids
                val_ids   = [ids[i] for i in perm[-n_val:]] if len(ids) > 1 else ids
                tr = KvasirSeg(root, train_ids, transform=train_tf)
                va = KvasirSeg(root, val_ids,   transform=val_tf)
                return tr, va, task, ignore

        elif ds == "voc":
            if isinstance(sp, dict):
                train_ids, val_ids = sp.get("train", []), sp.get("val", [])
            else:
                train_ids, val_ids = sp, None

            if voc_split_layout:
                tr = FolderSeg(os.path.join(root, "train"), ids=train_ids, transform=train_tf)
                va = FolderSeg(os.path.join(root, "val"),   ids=val_ids,   transform=val_tf)
            elif is_flat or voc_slim_layout:
                if voc_slim_layout:
                    print("[INFO] Using VOC slim layout with FolderSeg.")
                tr = FolderSeg(root, train_ids, transform=train_tf)
                if val_ids is None:
                    n_val = max(1, int(0.1 * len(train_ids))) if len(train_ids) > 1 else len(train_ids)
                    val_ids = train_ids[-n_val:]
                va = FolderSeg(root, val_ids, transform=val_tf)
            else:
                tr = VOCSeg(root, "train", ids_subset=train_ids, transform=train_tf)
                va = VOCSeg(root, "val", transform=val_tf)
            return tr, va, task, ignore

        elif ds == "parts":
            if isinstance(sp, dict):
                train_ids, val_ids = sp.get("train", []), sp.get("val", [])
            else:
                train_ids, val_ids = sp, None

            use_folder = is_flat or pp_layout
            if use_folder:
                tr = FolderSeg(root, train_ids, transform=train_tf)
                if val_ids is None:
                    n_val = max(1, int(0.1 * len(train_ids))) if len(train_ids) > 1 else len(train_ids)
                    val_ids = train_ids[-n_val:]
                va = FolderSeg(root, val_ids, transform=val_tf)
            else:
                tr = PascalParts(root, "train", transform=train_tf, ignore_index=ignore)
                tr.ids = [i for i in tr.ids if i in set(train_ids)]
                va = PascalParts(root, "val", transform=val_tf, ignore_index=ignore)
            return tr, va, task, ignore

    # ---------- Defaults ----------
    if ds == "kvasir":
        names = sorted(os.listdir(os.path.join(root, "images")))
        ids = [os.path.splitext(i)[0] for i in names]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(ids), generator=g).tolist()
        n_val = max(1, int(0.1 * len(ids)))
        train_ids = [ids[i] for i in perm[:-n_val]] if len(ids) > 1 else ids
        val_ids   = [ids[i] for i in perm[-n_val:]] if len(ids) > 1 else ids
        tr = KvasirSeg(root, train_ids, transform=train_tf)
        va = KvasirSeg(root, val_ids,   transform=val_tf)
        return tr, va, task, ignore

    elif ds == "voc":
        if voc_split_layout:
            tr = FolderSeg(os.path.join(root, "train"), transform=train_tf)
            va = FolderSeg(os.path.join(root, "val"),   transform=val_tf)
            return tr, va, task, ignore
        if is_flat:
            names = sorted(os.listdir(os.path.join(root, "images")))
            ids = [os.path.splitext(i)[0] for i in names]
            n_val = max(1, int(0.1 * len(ids))) if len(ids) > 1 else len(ids)
            tr = FolderSeg(root, ids[:-n_val], transform=train_tf)
            va = FolderSeg(root, ids[-n_val:], transform=val_tf)
            return tr, va, task, ignore
        if voc_slim_layout:
            # Build split from JPEGImages ∩ SegmentationClass
            img_dir = os.path.join(root, "JPEGImages")
            msk_dir = os.path.join(root, "SegmentationClass")
            names = sorted(os.listdir(img_dir))
            ids = []
            for n in names:
                stem, _ = os.path.splitext(n)
                if any(os.path.exists(os.path.join(msk_dir, stem + e)) for e in (".png", ".jpg", ".jpeg")):
                    ids.append(stem)
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(ids), generator=g).tolist()
            n_val = max(1, int(0.1 * len(ids))) if len(ids) > 1 else len(ids)
            train_ids = [ids[i] for i in perm[:-n_val]] if len(ids) > 1 else ids
            val_ids   = [ids[i] for i in perm[-n_val:]] if len(ids) > 1 else ids
            print(f"[INFO] VOC slim default split -> train={len(train_ids)}, val={len(val_ids)}")
            tr = FolderSeg(root, train_ids, transform=train_tf)
            va = FolderSeg(root, val_ids,   transform=val_tf)
            return tr, va, task, ignore

        # Final fallback: classic VOCdevkit layout (expects ImageSets lists)
        tr = VOCSeg(root, "train", transform=train_tf)
        va = VOCSeg(root, "val", transform=val_tf)
        return tr, va, task, ignore

    else:  # parts
        use_folder = is_flat or pp_layout
        if use_folder:
            names = sorted(os.listdir(os.path.join(root, "JPEGImages"))) if pp_layout else sorted(os.listdir(os.path.join(root, "images")))
            ids = [os.path.splitext(i)[0] for i in names]
            n_val = max(1, int(0.1 * len(ids))) if len(ids) > 1 else len(ids)
            tr = FolderSeg(root, ids[:-n_val], transform=train_tf)
            va = FolderSeg(root, ids[-n_val:], transform=val_tf)
        else:
            tr = PascalParts(root, "train", transform=train_tf, ignore_index=int(cfg.get("ignore_index", 255)))
            va = PascalParts(root, "val", transform=val_tf, ignore_index=int(cfg.get("ignore_index", 255)))
        return tr, va, task, int(cfg.get("ignore_index", 255))


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    # seed policy: exactly one seed per run (CLI can override by writing to cfg or env)
    seed = int(cfg.get("seed", os.environ.get("SEED", 0)))
    cfg["seed"] = seed
    _set_seed(seed)

    ds_train, ds_val, task, ignore_index = make_datasets(cfg)
    train_loader = _dataloader(ds_train, int(cfg["batch_size"]), int(cfg.get("workers", 4)), shuffle=True)
    val_loader   = _dataloader(ds_val,   int(cfg["batch_size"]), int(cfg.get("workers", 4)), shuffle=False)

    # quick sanity on batch shapes/dtypes
    xb, yb = next(iter(train_loader))
    assert xb.ndim == 4 and xb.shape[1] == 3 and xb.dtype == torch.float32, \
        f"Images must be [B,3,H,W] float32; got {xb.shape} {xb.dtype}"
    assert yb.ndim == 3 and yb.dtype in (torch.int64, torch.long), \
        f"Masks must be [B,H,W] int64; got {yb.shape} {yb.dtype}"

    # ---- [SANITY LOGS] input ranges to catch double-scaling / wrong normalization
    try:
        xb_min, xb_max, xb_mean = float(xb.min()), float(xb.max()), float(xb.mean())
        yb_min, yb_max = int(yb.min()), int(yb.max())
        print(
            f"[SANITY] xb dtype={xb.dtype} shape={tuple(xb.shape)} "
            f"min={xb_min:.3f} max={xb_max:.3f} mean={xb_mean:.3f}"
        )
        print(
            f"[SANITY] yb dtype={yb.dtype} shape={tuple(yb.shape)} "
            f"min={yb_min} max={yb_max}"
        )
    except Exception as e:
        print(f"[SANITY] failed to compute xb/yb stats: {e}")

    # Dataset sizes & steps (visibility)
    print(f"[INFO] train samples={len(ds_train)}  val samples={len(ds_val)}")
    print(f"[INFO] train steps/epoch={len(train_loader)}  val steps/epoch={len(val_loader)}")

    # Infer task once and reuse
    inferred_task = "binary" if cfg["num_classes"] == 1 else "multiclass"

    # Multiclass-only: labels must be within [0, C-1] or == ignore_index
    if inferred_task == "multiclass":
        ign = ignore_index if (ignore_index is not None) else 255
        valid_mask = (yb != ign)
        if valid_mask.any():
            yb_valid_max = int(yb[valid_mask].max())
            assert yb_valid_max < cfg["num_classes"], \
                f"num_classes={cfg['num_classes']} too small for labels up to {yb_valid_max}"
        else:
            print("[WARN] No valid labels in first batch (all ignore). Check your split/IDs.")

    net = SMPWrapper(cfg["family"], cfg["encoder"], cfg["num_classes"],
                     pretrained=cfg.get("pretrained", True),
                     **cfg.get("model_kwargs", {}))

    lit = LitSeg(net, task=inferred_task, num_classes=cfg["num_classes"],
                 lr=cfg["lr"], wd=cfg["wd"], ignore_index=ignore_index or 255)

    # Logging (TensorBoard + CSV)
    out_dir = cfg.get("out_dir", "runs")
    tb_logger  = pl.loggers.TensorBoardLogger(save_dir=out_dir, name="tb")
    csv_logger = pl.loggers.CSVLogger(save_dir=out_dir, name="csv")
    logger = [tb_logger, csv_logger]

    # Determine monitor if not specified in cfg
    monitor = cfg.get("monitor")
    if monitor is None:
        monitor = "val/dice" if cfg["num_classes"] == 1 else "val/miou"

    # Checkpoints (top-k + last)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:02d}-step{step:06d}",
            monitor=monitor,
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(monitor=monitor, mode="max",
                                   patience=int(cfg["patience"]), min_delta=float(cfg["min_delta"])),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        _RunStatsCallback(monitor=monitor, mode="max", img_size=int(cfg["image_size"]), batch_size=int(cfg["batch_size"])),
        _ValDebugOnce(num_classes=int(cfg["num_classes"]), ignore_index=ignore_index or 255),
    ]

    # Train-to-convergence toggle
    train_to_convergence = bool(cfg.get("train_to_convergence", False))
    cfg_max_steps = int(cfg.get("max_steps", 0))
    use_convergence = train_to_convergence or (cfg_max_steps <= 0)

    # ---- derive a safe validation interval ----
    n_train_batches = len(train_loader)
    eval_every_cfg = int(cfg.get("eval_every_steps", 0))
    if eval_every_cfg > 0 and n_train_batches > 0:
        val_check_interval_safe = min(eval_every_cfg, n_train_batches)
    else:
        val_check_interval_safe = None  # fall back to epoch-based validation

    # ---- adaptive logging interval so PL doesn't warn
    log_every = int(cfg.get("log_every_n_steps", 50))
    if n_train_batches > 0:
        log_every = max(1, min(log_every, n_train_batches))

    trainer_kwargs = dict(
        log_every_n_steps=log_every,
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        default_root_dir=out_dir,
        accelerator="gpu" if int(cfg.get("gpus", 0)) > 0 and torch.cuda.is_available() else "cpu",
        devices=int(cfg.get("gpus", 0)) or 1,
        logger=logger,
        plugins=[LocalCheckpointIO()],  # Drive-safe checkpoint IO
        # deterministic decided below
    )
    if use_convergence:
        trainer_kwargs["max_epochs"] = int(cfg.get("max_epochs", 1000))
        trainer_kwargs["max_steps"] = None
    else:
        trainer_kwargs["max_steps"] = int(cfg["max_steps"])

    # Apply validation cadence (either steps-capped or epoch-based)
    if val_check_interval_safe is not None:
        trainer_kwargs["val_check_interval"] = val_check_interval_safe
        if val_check_interval_safe != eval_every_cfg:
            print(f"[INFO] Capped val_check_interval from {eval_every_cfg} to {val_check_interval_safe} "
                  f"(train batches={n_train_batches}).")
    else:
        trainer_kwargs["check_val_every_n_epoch"] = 1  # validate once per epoch

    # ---- Determinism guard for CUDA + CrossEntropy (VOC/Parts: multiclass)
    deterministic_flag = True
    if ("multiclass" in (task or "")) and trainer_kwargs["accelerator"] == "gpu" and torch.cuda.is_available():
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        deterministic_flag = False
        print("[INFO] Disabled strict determinism for multiclass CE on CUDA to avoid nll_loss2d error.")
    trainer_kwargs["deterministic"] = deterministic_flag

    trainer = pl.Trainer(**trainer_kwargs)

    # Auto-resume from last.ckpt if present
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None
    trainer.fit(lit, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--seed", type=int, default=None, help="Override seed without editing YAML")
    args = p.parse_args()
    if args.seed is not None:
        cfg = yaml.safe_load(open(args.cfg))
        cfg["seed"] = int(args.seed)
        tmp = ".run_override_seed.yaml"
        with open(tmp, "w") as f:
            yaml.safe_dump(cfg, f)
        try:
            main(tmp)
        finally:
            os.remove(tmp)
    else:
        main(args.cfg)
