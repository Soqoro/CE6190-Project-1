from __future__ import annotations
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary segmentation given logits and 0/1 targets [B,1,H,W]."""
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(1, 2, 3)) + eps
    den = (probs.pow(2) + targets.pow(2)).sum(dim=(1, 2, 3)) + eps
    return 1.0 - (num / den).mean()


class LitSeg(pl.LightningModule):
    """Lightning training/eval for segmentation with SMP backbone.

    - Binary: BCEWithLogits + Dice (monitor `val/dice`)
    - Multiclass: CrossEntropy + mIoU (monitor `val/miou`)
    - Fixed `max_steps` is set at Trainer-level for fair compute.
    """
    def __init__(
        self,
        net: nn.Module,
        task: str,
        num_classes: int,
        lr: float = 3e-4,
        wd: float = 5e-4,
        ignore_index: int | None = 255,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.task = task
        # torchmetrics wants -100 for "no ignore" in CE; keep None -> -100 internally
        self.ignore_index = -100 if ignore_index is None else int(ignore_index)

        if task == "binary":
            self.loss_bce = nn.BCEWithLogitsLoss()
            # Dice metric for binary segmentation; thresholds preds internally at 0.5
            self.metric = tm.Dice(task="binary")
        else:
            self.loss_ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            self.metric = tm.JaccardIndex(
                task="multiclass",
                num_classes=num_classes,
                ignore_index=self.ignore_index,
            )

    def forward(self, x):  # type: ignore
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch  # y: [B,H,W] (int64)
        logits = self(x)
        if self.task == "binary":
            y_f = y.float().unsqueeze(1)  # [B,1,H,W] in {0,1}
            loss = 0.5 * self.loss_bce(logits, y_f) + 0.5 * dice_loss_with_logits(logits, y_f)
        else:
            loss = self.loss_ce(logits, y)

        # Step-wise logging for plots
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        # Log LR from first optimizer group
        opt = self.optimizers()
        if opt is not None:
            opt0 = opt[0] if isinstance(opt, (list, tuple)) else opt
            lr = opt0.param_groups[0].get("lr", None)
            if lr is not None:
                self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch  # y: [B,H,W] (int64)
        logits = self(x)
        if self.task == "binary":
            # Provide probabilities to the Dice metric; it applies threshold=0.5
            preds = torch.sigmoid(logits).squeeze(1)  # [B,H,W]
            val = self.metric(preds, y.int())
            self.log("val/dice", val, prog_bar=True, sync_dist=True)
        else:
            # JaccardIndex handles logits directly for multiclass
            val = self.metric(logits, y)
            self.log("val/miou", val, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        # Trainer may not be attached at construction time; use safe fallback
        tmax = getattr(self.trainer, "max_steps", 1000) or 1000
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
