from __future__ import annotations
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm
from torchmetrics.segmentation import DiceScore

class LitSeg(pl.LightningModule):
    def __init__(self, net: torch.nn.Module, task: str, num_classes: int,
                 lr: float, wd: float, ignore_index: int = 255):
        super().__init__()
        self.net = net
        self.task = task  # "binary" or "multiclass"
        self.num_classes = int(num_classes)
        self.lr = float(lr)
        self.wd = float(wd)
        self.ignore_index = int(ignore_index)

        if self.task == "multiclass":
            # IMPORTANT: respect VOC ignore label
            self.loss_ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            self.val_metric = tm.JaccardIndex(
                task="multiclass",
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                average="macro",
            )
        else:
            self.loss_bce = nn.BCEWithLogitsLoss()
            self.val_metric = DiceScore(num_classes=2, include_background=False, average="macro")

    def forward(self, x):
        return self.net(x)

    def _sanitize_targets(self, y: torch.Tensor) -> torch.Tensor:
        """
        Ensure labels are valid for CE; any label outside [0, C-1] is set to ignore_index.
        This prevents CUDA device-side asserts.
        """
        if self.task == "multiclass":
            if (y.min() < 0) or (y.max() >= self.num_classes):
                y = y.clone()
                y[(y < 0) | (y >= self.num_classes)] = self.ignore_index
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)

        if self.task == "multiclass":
            y = self._sanitize_targets(y)
            loss = self.loss_ce(logits, y)
        else:
            # y: [B,H,W] -> [B,1,H,W] float
            loss = self.loss_bce(logits.squeeze(1), y.float())

        # sync_dist=False avoids extra device juggling during the assert path
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)

        if self.task == "multiclass":
            y = self._sanitize_targets(y)
            preds = logits.argmax(dim=1)
        else:
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)

        self.val_metric.update(preds, y)

    def on_validation_epoch_end(self):
        val_score = self.val_metric.compute()
        # metric name matches your configs (val/miou for multiclass, val/dice for binary)
        name = "val/miou" if self.task == "multiclass" else "val/dice"
        self.log(name, val_score, prog_bar=True, sync_dist=False)
        self.val_metric.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return opt
