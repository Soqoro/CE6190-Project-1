from __future__ import annotations
import os, time, shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from src.utils.count_flops import count_flops_params
from src.utils.latency import measure_latency

class RunStatsCallback(Callback):
    """Log run-level stats: wall-clock, params, FLOPs (approx), latency, steps_to_best."""

    def __init__(self, monitor: str, mode: str, image_size: int, out_dir: str):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.image_size = image_size
        self.best = None
        self.best_step = None
        self.t0 = None
        self.out_dir = out_dir

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.t0 = time.time()
        # Params
        params = sum(p.numel() for p in pl_module.parameters())
        trainer.logger.log_metrics({"params": float(params)}, step=trainer.global_step)
        # FLOPs (approx)
        try:
            flops, _ = count_flops_params(pl_module.net, input_res=(3, self.image_size, self.image_size))
            trainer.logger.log_metrics({"flops": float(flops)}, step=trainer.global_step)
        except Exception as e:
            trainer.logger.log_metrics({"flops": float('nan')}, step=trainer.global_step)
        # Latency (single forward)
        try:
            device = pl_module.device
            lat_ms = measure_latency(pl_module.net, device=device, input_size=(1,3,self.image_size,self.image_size), warmup=10, iters=30)
            trainer.logger.log_metrics({"latency_ms": float(lat_ms)}, step=trainer.global_step)
        except Exception:
            trainer.logger.log_metrics({"latency_ms": float('nan')}, step=trainer.global_step)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.monitor in trainer.callback_metrics:
            cur = trainer.callback_metrics[self.monitor].item()
            step = trainer.global_step
            if self.best is None:
                self.best, self.best_step = cur, step
            else:
                improved = (cur > self.best) if self.mode == "max" else (cur < self.best)
                if improved:
                    self.best, self.best_step = cur, step

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        elapsed = time.time() - (self.t0 or time.time())
        trainer.logger.log_metrics({"wall_clock_s": float(elapsed), "steps_to_best": int(self.best_step or -1)}, step=trainer.global_step)
        # Copy latest metrics.csv to out_dir/metrics.csv for easy aggregation
        try:
            if isinstance(trainer.logger, pl.loggers.LoggerCollection):
                loggers = trainer.logger
                for lg in loggers:
                    if isinstance(lg, pl.loggers.csv_logs.CSVLogger):
                        src = os.path.join(lg.log_dir, "metrics.csv")
                        if os.path.exists(src):
                            shutil.copyfile(src, os.path.join(self.out_dir, "metrics.csv"))
            elif isinstance(trainer.logger, pl.loggers.csv_logs.CSVLogger):
                src = os.path.join(trainer.logger.log_dir, "metrics.csv")
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(self.out_dir, "metrics.csv"))
        except Exception:
            pass
