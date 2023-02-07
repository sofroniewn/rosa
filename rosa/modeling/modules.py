from typing import Union

import torch.optim as optim
from pytorch_lightning import LightningModule

from ..utils.config import ModuleConfig
from .components import criterion_factory
from .models import RosaJointModel, RosaSingleModel


class RosaLightningModule(LightningModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: ModuleConfig,
    ):
        super(RosaLightningModule, self).__init__()
        if isinstance(in_dim, tuple):
            self.model = RosaJointModel(
                in_dim=in_dim,
                out_dim=out_dim,
                config=config.model,
            )  # type: Union[RosaSingleModel, RosaJointModel]
        else:
            self.model = RosaSingleModel(
                in_dim=in_dim,
                out_dim=out_dim,
                config=config.model,
            )
        self.learning_rate = config.learning_rate
        self.criterion = criterion_factory(config.criterion)

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
        return self.model.sample(y_hat)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.learning_rate)
