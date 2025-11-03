from __future__ import annotations
import os, yaml, time, json
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import torch

from src.data.transforms import make_transforms
from src.data.kvasir import KvasirSeg
from src.data.voc import VOCSeg, IGNORE_INDEX as VOC_IGNORE
from src.data.pascal_parts import PascalParts
from src.data.splits import load_ids
from src.models.smp_wrapper import SMPWrapper
from src.engine.lit_module import LitSeg
from src.data.folder_seg import FolderSeg  # generic flat images/masks loader


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


def _flat_layout_exists(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "masks"))


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
      - else: defaults (Kvasir 90/10; VOC/Parts official lists).
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

    if ds == "kvasir":
        loader_cls = KvasirSeg
        task, ignore = "binary", None
    elif ds == "voc":
        loader_cls = FolderSeg if is_flat else VOCSeg
        task, ignore = "multiclass", VOC_IGNORE
    else:  # parts
        loader_cls = FolderSeg if is_flat else PascalParts
        task, ignore = "multiclass", int(cfg.get("ignore_index", 255))

    # ---------- Preferred: split_manifest ----------
    if cfg.get("split_manifest"):
        man = _read_ids_or_manifest(cfg["split_manifest"])
        if isinstance(man, list):
            # If someone accidentally points a list here, make a 90/10 split for train/val.
            ids = man
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(ids), generator=g).tolist()
            n_val = max(1, int(0.1 * len(ids)))
            train_ids = [ids[i] for i in perm[:-n_val]] if len(ids) > 1 else ids
            val_ids   = [ids[i] for i in perm[-n_val:]] if len(ids) > 1 else ids
        else:
            train_ids, val_ids = man["train"], man["val"]

        if loader_cls in (FolderSeg, KvasirSeg):
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
            # If it's a manifest dict, use it directly; else do seeded 90/10 on the list.
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
                ids = sp  # list
                g = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(ids), generator=g).tolist()
                n_val = max(1, int(0.1 * len(ids)))
                train_ids = [ids[i] for i in perm[:-n_val]] if len(ids) > 1 else ids
                val_ids   = [ids[i] for i in perm[-n_val:]] if len(ids) > 1 else ids
                tr = KvasirSeg(root, train_ids, transform=train_tf)
                va = KvasirSeg(root, val_ids,   transform=val_tf)
                return tr, va, task, ignore

        elif ds in ("voc", "parts"):
            # Restrict train to ids in list or manifest; val depends on layout
            if isinstance(sp, dict):
                train_ids, val_ids = sp.get("train", []), sp.get("val", [])
            else:
                train_ids, val_ids = sp, None

            if is_flat:
                tr = FolderSeg(root, train_ids, transform=train_tf)
                if val_ids is None:
                    # simple fallback val: take 10% of the provided train ids
                    n_val = max(1, int(0.1 * len(train_ids))) if len(train_ids) > 1 else len(train_ids)
                    val_ids = train_ids[-n_val:]
                va = FolderSeg(root, val_ids, transform=val_tf)
            else:
                if ds == "voc":
                    tr = VOCSeg(root, "train", ids_subset=train_ids, transform=train_tf)
                    va = VOCSeg(root, "val", transform=val_tf)
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
        if is_flat:
            names = sorted(os.listdir(os.path.join(root, "images")))
            ids = [os.path.splitext(i)[0] for i in names]
            n_val = max(1, int(0.1 * len(ids))) if len(ids) > 1 else len(ids)
            tr = FolderSeg(root, ids[:-n_val], transform=train_tf)
            va = FolderSeg(root, ids[-n_val:], transform=val_tf)
        else:
            tr = VOCSeg(root, "train", transform=train_tf)
            va = VOCSeg(root, "val", transform=val_tf)
        return tr, va, task, ignore

    else:  # parts
        if is_flat:
            names = sorted(os.listdir(os.path.join(root, "images")))
            ids = [os.path.splitext(i)[0] for i in names]
            n_val = max(1, int(0.1 * len(ids))) if len(ids) > 1 else len(ids)
            tr = FolderSeg(root, ids[:-n_val], transform=train_tf)
            va = FolderSeg(root, ids[-n_val:], transform=val_tf)
        else:
            tr = PascalParts(root, "train", transform=train_tf, ignore_index=ignore)
            va = PascalParts(root, "val", transform=val_tf, ignore_index=ignore)
        return tr, va, task, ignore


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

    net = SMPWrapper(cfg["family"], cfg["encoder"], cfg["num_classes"],
                     pretrained=cfg.get("pretrained", True),
                     **cfg.get("model_kwargs", {}))

    inferred_task = "binary" if cfg["num_classes"] == 1 else "multiclass"
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
    ]

    # Train-to-convergence toggle
    train_to_convergence = bool(cfg.get("train_to_convergence", False))
    cfg_max_steps = int(cfg.get("max_steps", 0))
    use_convergence = train_to_convergence or (cfg_max_steps <= 0)

    trainer_kwargs = dict(
        val_check_interval=int(cfg["eval_every_steps"]),
        log_every_n_steps=50,
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        default_root_dir=out_dir,
        accelerator="gpu" if int(cfg.get("gpus", 0)) > 0 and torch.cuda.is_available() else "cpu",
        devices=int(cfg.get("gpus", 0)) or 1,
        deterministic=True,
        logger=logger,
    )
    if use_convergence:
        trainer_kwargs["max_epochs"] = int(cfg.get("max_epochs", 1000))
        trainer_kwargs["max_steps"] = None
    else:
        trainer_kwargs["max_steps"] = int(cfg["max_steps"])

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
