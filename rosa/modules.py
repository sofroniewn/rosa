from typing import Union

import torch
import torch.optim as optim
import torchmetrics.functional as F
from pytorch_lightning import LightningModule

from .models import RosaJointModel, RosaSingleModel


def mean_log_prob_criterion(output: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    return -output.log_prob(target).sum(-1).mean()


class RosaLightningModule(LightningModule):
    def __init__(
        self,
        in_dim,
        out_dim,
        model_config,
        learning_rate,
    ):
        super().__init__()
        if isinstance(in_dim, tuple):
            RosaModel = RosaJointModel  # type: Union[RosaSingleModel, RosaJointModel]
        else:
            RosaModel = RosaSingleModel
        self.model = RosaModel(
            in_dim=in_dim,
            out_dim=out_dim,
            config=model_config,
        )
        self.learning_rate = learning_rate
        self.criterion = None

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        if isinstance(y_hat, torch.distributions.Distribution) and self.criterion is None:
            loss = mean_log_prob_criterion(y_hat, y)
        else:
            loss = self.model.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        if isinstance(y_hat, torch.distributions.Distribution) and self.criterion is None:
            loss = mean_log_prob_criterion(y_hat, y)
        else:
            loss = self.model.loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, _):
        x, _ = batch
        y_hat = self(x)
        if isinstance(y_hat, torch.distributions.Distribution):
            return y_hat.mean
        return y_hat

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.learning_rate)
