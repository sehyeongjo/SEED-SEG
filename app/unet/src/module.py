from typing import Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


def dice_score_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    # logits: [B,1,H,W], targets: [B,1,H,W] (0/1)
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        # soft dice
        eps = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + eps) / (union + eps)
        dice_loss = 1 - dice.mean()

        return self.bce_w * bce + self.dice_w * dice_loss


class SegLitModule(pl.LightningModule):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.save_hyperparameters(cfg)

        mcfg = cfg["model"]
        self.model = smp.Unet(
            encoder_name=mcfg["encoder_name"],
            encoder_weights=mcfg["encoder_weights"],
            in_channels=mcfg["in_channels"],
            classes=mcfg["num_classes"],
        )

        lcfg = cfg["loss"]
        self.criterion = BCEDiceLoss(lcfg["bce_weight"], lcfg["dice_weight"])

        tcfg = cfg["train"]
        self.lr = float(tcfg["lr"])
        self.lr_unfreeze = float(tcfg["lr_unfreeze"])
        self.weight_decay = float(tcfg["weight_decay"])
        self.freeze_encoder_epochs = int(tcfg["freeze_encoder_epochs"])
        self.threshold = float(tcfg["threshold"])

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        # freeze encoder for first N epochs
        if self.current_epoch < self.freeze_encoder_epochs:
            self._set_encoder_trainable(False)
        elif self.current_epoch == self.freeze_encoder_epochs:
            self._set_encoder_trainable(True)

    def _set_encoder_trainable(self, trainable: bool):
        for p in self.model.encoder.parameters():
            p.requires_grad = trainable

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        logits = self(x)
        loss = self.criterion(logits, y)
        dice = dice_score_from_logits(logits, y, threshold=self.threshold)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_dice", dice, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        logits = self(x)
        loss = self.criterion(logits, y)
        dice = dice_score_from_logits(logits, y, threshold=self.threshold)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        self.log("val_dice", dice, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        # Use one LR while encoder is frozen, then switch after unfreeze.
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Use cosine scheduler by default.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # Switch learning rate based on epoch.
        if epoch < self.freeze_encoder_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr
        else:
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr_unfreeze

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

