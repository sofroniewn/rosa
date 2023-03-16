from typing import Optional, Union

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule

from ..utils.config import ModuleConfig
from .models import RosaTransformer, criterion_factory


class RosaLightningModule(LightningModule):
    def __init__(
        self,
        var_input: torch.Tensor,
        config: ModuleConfig,
    ):
        super(RosaLightningModule, self).__init__()
        self.model = RosaTransformer(
            in_dim=var_input.shape[1],
            config=config.model,
        )
        if var_input is not None:
            self.register_buffer("var_input", var_input)
        else:
            self.var_input = None
        self.learning_rate = config.learning_rate
        self.criterion = criterion_factory(config.criterion)

    def forward(self, batch):
        return self.model.forward(batch)

    def training_step(self, batch, _):
        expression = batch['expression'].clone().detach()
        batch['var_input'] = self.var_input[batch['indices']]
        expression_predicted = self(batch)
        expression_predicted = expression_predicted[batch['mask']]
        expression = expression[batch['mask']]
        loss = self.criterion(expression_predicted, expression)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        expression = batch['expression'].clone().detach()
        batch['var_input'] = self.var_input[batch['indices']]
        expression_predicted = self(batch)
        expression_predicted = expression_predicted[batch['mask']]
        expression = expression[batch['mask']]
        loss = self.criterion(expression_predicted, expression)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        expression = batch['expression'].clone().detach()
        batch['var_input'] = self.var_input[batch['indices']]
        expression_predicted = self(batch)
        batch_size = batch['mask'].shape[0]
        nbins = expression_predicted.shape[-1]
        results = {}
        results['expression_predicted'] = expression_predicted[batch['mask']].reshape(batch_size, -1, nbins)
        results['expression'] = expression[batch['mask']].reshape(batch_size, -1)
        results['batch_idx'] = batch_idx
        results['dataloader_idx'] = dataloader_idx
        return results

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    # def explain_iter(self, dataloader, explainer, indices=None):
    #     for x, y in iter(dataloader):
    #         if isinstance(x, list):
    #             x = tuple(
    #                 x_ind.reshape(-1, x_ind.shape[-1]).requires_grad_() for x_ind in x
    #             )
    #             attribution = explainer.attribute(x)
    #             yield tuple(a.reshape(y.shape[0], y.shape[1], -1) for a in attribution)
    #         else:
    #             attribution = []
    #             if indices is None:
    #                 indices = range(y.shape[-1])
    #             for target in indices:
    #                 x.requires_grad_()
    #                 attribution.append(explainer.attribute(x, target=int(target)))
    #             yield torch.stack(attribution, dim=1)
