from typing import Optional, Union

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule

from ..utils.config import ModuleConfig
from .models import RosaJointModel, RosaSingleModel, RosaFormerModel, criterion_factory


class RosaLightningModule(LightningModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: ModuleConfig,
        var_input: Optional[torch.Tensor] = None,
    ):
        super(RosaLightningModule, self).__init__()
        if isinstance(in_dim, tuple):
            if out_dim == 1:
                self.model = RosaJointModel(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    config=config.model,
                )  # type: Union[RosaSingleModel, RosaJointModel, RosaFormerModel]
            else:
                self.model = RosaFormerModel(
                    in_dim=in_dim,
                    config=config.model,
                )
        else:
            self.model = RosaSingleModel(
                in_dim=in_dim,
                out_dim=out_dim,
                config=config.model,
            )
        if var_input is not None:
            self.register_buffer("var_input", var_input)
        else:
            self.var_input = None
        self.learning_rate = config.learning_rate
        self.criterion = criterion_factory(config.criterion)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, _):
        x, y = batch
        if self.var_input is not None:
            x = (x[0], self.var_input[x[1]])
        y_hat = self(x)
        if isinstance(self.model, RosaFormerModel):
            mask = x[0][1]
            y_hat = y_hat[mask]
            y = y[mask]
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        if self.var_input is not None:
            x = (x[0], self.var_input[x[1]])
        y_hat = self(x)
        if isinstance(self.model, RosaFormerModel):
            mask = x[0][1]
            y_hat = y_hat[mask]
            y = y[mask]
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, _):
        x, y = batch
        if self.var_input is not None:
            x = (x[0], self.var_input[x[1]])
        y_hat = self(x)
        return (y_hat, y)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def explain_iter(self, dataloader, explainer, indices=None):
        for x, y in iter(dataloader):
            if isinstance(x, list):
                x = tuple(
                    x_ind.reshape(-1, x_ind.shape[-1]).requires_grad_() for x_ind in x
                )
                attribution = explainer.attribute(x)
                yield tuple(a.reshape(y.shape[0], y.shape[1], -1) for a in attribution)
            else:
                attribution = []
                if indices is None:
                    indices = range(y.shape[-1])
                for target in indices:
                    x.requires_grad_()
                    attribution.append(explainer.attribute(x, target=int(target)))
                yield torch.stack(attribution, dim=1)
