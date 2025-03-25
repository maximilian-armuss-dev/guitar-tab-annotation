import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from dataclasses import dataclass
from torch.optim import Adam


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, reduce_res=False):
        super().__init__()
        self.reduce_res = reduce_res
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stack(x)
        if self.reduce_res:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x


class Lambda(nn.Module):
    def __init__(self, f: callable):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


@dataclass
class ModelConfig:
    num_classes: int
    feature_num: int
    im_res: int
    in_channels: int
    num_heads: int
    avg_pool_dim: int
    learning_rate: float
    dropout_p: float = 0.2
    decay_rate: float = 1e-5

    def __post_init__(self):
        assert math.log2(self.im_res).is_integer(), f"Image Resolution must be power of 2, but was {self.im_res}"


class NotationModel(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.train_losses = []
        self.val_losses = []
        self.encoder = nn.Sequential(
            ConvBlock2D(1, config.in_channels, True),
            ConvBlock2D(config.in_channels, config.in_channels * 2),
            ConvBlock2D(config.in_channels * 2, config.in_channels * (2 ** 2), True),
            ConvBlock2D(config.in_channels * (2 ** 2), config.in_channels * (2 ** 3)),
            ConvBlock2D(config.in_channels * (2 ** 3), config.in_channels * (2 ** 4), True),
            ConvBlock2D(config.in_channels * (2 ** 4), config.feature_num),
            nn.AdaptiveAvgPool2d(output_size=(config.avg_pool_dim, config.avg_pool_dim)),
            nn.Flatten(start_dim=1),
            Lambda(lambda x: F.dropout(x, p=config.dropout_p, training=self.training))
        )
        self.classification_heads = nn.ModuleList(
            [nn.Linear((config.avg_pool_dim ** 2) * config.feature_num, config.num_classes)
             for _ in range(config.num_heads)]
        )

    def forward(self, x):
        assert len(x.shape) == 4, "Input shape must be (B, C, H, W)"
        x = self.encoder(x)
        x = torch.stack([head(x) for head in self.classification_heads], dim=-1)
        return x

    def predict(self, x):
        logits = self(x)
        pred_classes = torch.argmax(logits, dim=1)
        return pred_classes

    def _get_loss(self, batch, batch_idx):
        images, labels = batch
        class_pred = self(images)
        ce_loss = sum(F.cross_entropy(class_pred[:, :, i], labels[:, i])
                      for i in range(class_pred.shape[-1]))
        normalizing_constant = 10_000 / (self.config.im_res**2)
        loss = ce_loss * normalizing_constant
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics["train_loss"].item()
        self.train_losses.append(train_loss)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.decay_rate)
