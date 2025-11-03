from __future__ import annotations
import os, yaml, time
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import torch

from src.data.transforms import make_transforms
from src.data.kvasir import KvasirSeg
from src.data.voc import VOCSeg, IGNORE_INDEX as VOC_IGNORE
from src.data.pascal_parts import PascalParts
from src.data.splits import load_ids
from src.models.smp_wrapper import SMPWrapper
from src.engine.lit_module import LitSeg


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
    Logs run-level stats:
      - params
      - simple latency_ms for a single forward (approx)
      - steps_to_best for the monitored metric
      - wall_clock_s
    Works with CSV/TensorBoard loggers.
    """
    def __init__(self, monitor: str, mode: str, img_size: int):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.img_size = int(img_size)
        self._t0: Optional[float] = None
        self._best = None
        self._best_step = None

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
            else:
                better = (cur > self._best) if self.mode == "max" else (cur < self._best)
                if better:
                    self._best, self._best_step = cur, step

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        elapsed = time.time() - (self._t0 or time.time())
        trainer.logger.log_metrics(
            {"wall_clock_s": float(elapsed), "steps_to_best": int(self._best_step or -1)},
            step=trainer.global_step,
        )


def make_datasets(cfg):
    ds = cfg["dataset"].lower()

    if ds == "kvasir":
        train_tf, val_tf = make_transforms("medical", int(cfg["image_size"]))
        # Expect split_file like splits/kvasir/seed0_5.json (list of ids)
        if cfg.get("split_file"):
            ids = load_ids(cfg["split_file"])
        else:
            names = sorted(os.listdir(os.path.join(cfg["root"], "images")))
            ids = [os.path.splitext(i)[0] for i in names]
        # split into 90/10 using seed for reproducibility
        seed = int(cfg.get("seed", 0))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(ids), generator=g).tolist()
        n_val = max(1, int(0.1 * len(ids)))
        train_ids = [ids[i] for i in perm[:-n_val]] if len(ids) > 1 else ids
        val_ids   = [ids[i] for i in perm[-n_val:]] if len(ids) > 1 else ids
        tr = KvasirSeg(cfg["root"], train_ids, transform=train_tf)
        va = KvasirSeg(cfg["root"], val_ids,   transform=val_tf)
        task = "binary"
        ignore = None

    elif ds == "voc":
        train_tf, val_tf = make_transforms("scene", int(cfg["image_size"]))
        tr = VOCSeg(cfg["root"], "train",
                    ids_subset=load_ids(cfg["split_file"]) if cfg.get("split_file") else None,
                    transform=train_tf)
        va = VOCSeg(cfg["root"], "val", transform=val_tf)
        task = "multiclass"
        ignore = VOC_IGNORE

    elif ds == "parts":
        train_tf, val_tf = make_transforms("parts", int(cfg["image_size"]))
        tr = PascalParts(cfg["root"], "train", transform=train_tf,
                         ignore_index=cfg.get("ignore_index", 255))
        va = PascalParts(cfg["root"], "val", transform=val_tf,
                         ignore_index=cfg.get("ignore_index", 255))
        task = "multiclass"
        ignore = cfg.get("ignore_index", 255)

    else:
        raise ValueError(f"Unknown dataset: {ds}")

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

    # Checkpoints (top-k + last) to a stable directory
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
        _RunStatsCallback(monitor=monitor, mode="max", img_size=int(cfg["image_size"])),
    ]

    trainer = pl.Trainer(
        max_steps=int(cfg["max_steps"]),
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
        # Lightweight override of seed for this invocation
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
