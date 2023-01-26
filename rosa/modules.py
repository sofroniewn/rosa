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
        return y_hat

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.learning_rate)


# # Define the KS loss function
# def ks_loss(output, target):
#     output_sorted, _ = torch.sort(output)
#     target_sorted, _ = torch.sort(target)
#     output_cdf = torch.cumsum(output_sorted, dim=0)
#     target_cdf = torch.cumsum(target_sorted, dim=0)
#     output_cdf = output_cdf / output_cdf[-1]
#     target_cdf = target_cdf / target_cdf[-1]
#     # ks = torch.max(torch.abs(output_cdf - target_cdf))
#     ks = F.mean_squared_error(output_cdf, target_cdf)
#     return ks


#         # self.criterion = nn.CrossEntropyLoss(reduction='mean')
#         # self.criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)
#         # self.loss_sm = nn.LogSoftmax(dim=0)


#     def reconstruction_loss(self, y_hat, y):
#         # library_size_hat = torch.expm1(y_hat).sum(axis=0) * y_hat.shape[0]
#         # library_size_hat = y_hat.sum(axis=0) * y_hat.shape[0]
#         # library_size = self.model.library_size * torch.ones_like(library_size_hat)
#         # y_hat = self.loss_sm(y_hat)
#         # y_hat = nn.LogSoftmax(y, dim=0)
#         # loss = (1 - F.concordance_corrcoef(y_hat, y).mean() + F.mean_squared_error(y_hat, y)) / 2
#         # loss = F.mean_absolute_error(y_hat, y)
#         # loss = 1 - F.pearson_corrcoef(y_hat, y).mean()
#         # loss = 1 - F.concordance_corrcoef(y_hat, y).mean()
#         # return ks_loss(y_hat, y)
#         # return (1 - F.pearson_corrcoef(y_hat, y).mean() + F.mean_squared_error(y_hat, y)) / 2
#         # return self.criterion(y_hat, y.long())
#         return F.mean_squared_error(y_hat, y)
#         # return F.mean_squared_error(y_hat, y) + F.mean_squared_error(library_size_hat, library_size)
#         # return F.mean_squared_error(y_hat, y) + torch.log1p(F.mean_squared_error(torch.expm1(y_hat).sum(axis=0), torch.expm1(y).sum(axis=0)))
#         # return F.mean_squared_error(y_hat, y) + F.mean_squared_error(y_hat.sum(axis=0), y.sum(axis=0)) / y.shape[0]
#         # return F.mean_absolute_error(y_hat, y)
#         # return F.kl_divergence(y_hat, y, log_prob=True, reduction='mean')

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.reconstruction_loss(y_hat, y)
#         self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.reconstruction_loss(y_hat, y)
#         self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
#         return loss

#     def predict_step(self, batch, batch_idx):
#         x, _ = batch
#         y_hat = self(x)
#         # _, y_hat = torch.max(y_hat, dim=1)
#         return y_hat

#     def configure_optimizers(self):
#         # return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
#         return optim.AdamW(self.model.parameters(), lr=0.001)
