from typing import Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics.functional as F
from pytorch_lightning import LightningModule

from .models import RosaModel


class RosaLightningModule(LightningModule):
    def __init__(
        self,
        in_dim,
        out_dim,
        model_cfg,
        learning_rate,
    ):
        super().__init__()
        self.model = RosaModel(
            in_dim=in_dim,
            out_dim=out_dim,
            model_cfg=model_cfg,
        )
        self.learning_rate = learning_rate
        self.criterion = F.mean_squared_error
        # self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, _):
        x, _ = batch
        y_hat = self(x)
        # _, y_hat = torch.max(y_hat, dim=1)
        return y_hat

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.learning_rate)
