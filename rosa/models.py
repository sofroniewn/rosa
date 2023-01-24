from typing import Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics.functional as F
from pytorch_lightning import LightningModule
from scvi.distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
    ZeroInflatedNegativeBinomial,
)

from .modules import BilinearHead, ConcatHead, SingleHead


class JointEmbedding2ExpressionModel(LightningModule):
    def __init__(
        self,
        in_dim_1,
        in_dim_2,
        rank,
        head_1="Linear",
        head_2="Linear",
        method="bilinear",
    ):
        super().__init__()
        if method == "bilinear":
            self.model = BilinearHead(
                in_dim_1=in_dim_1,
                in_dim_2=in_dim_2,
                rank=rank,
                head_1=head_1,
                head_2=head_2,
            )
        elif method == "concat":
            self.model = ConcatHead(
                in_dim_1=in_dim_1,
                in_dim_2=in_dim_2,
                head=head_1,
            )
        else:
            raise ValueError(f"Item {method} not recognized")

    def forward(self, x1, x2):
        return self.model.forward(x1, x2)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat = self(x1, x2)
        loss = F.mean_squared_error(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat = self(x1, x2)
        loss = F.mean_squared_error(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x1, x2, _ = batch
        y_hat = self(x1, x2)
        return y_hat

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)


# Define the KS loss function
def ks_loss(output, target):
    output_sorted, _ = torch.sort(output)
    target_sorted, _ = torch.sort(target)
    output_cdf = torch.cumsum(output_sorted, dim=0)
    target_cdf = torch.cumsum(target_sorted, dim=0)
    output_cdf = output_cdf / output_cdf[-1]
    target_cdf = target_cdf / target_cdf[-1]
    # ks = torch.max(torch.abs(output_cdf - target_cdf))
    ks = F.mean_squared_error(output_cdf, target_cdf)
    return ks


class SingleEmbedding2ExpressionModel(LightningModule):
    def __init__(
        self,
        in_dim,
        out_dim,
        head="Linear",
    ):
        super().__init__()
        self.model = SingleHead(in_dim, out_dim, head=head)
        # self.criterion = nn.CrossEntropyLoss(reduction='mean')
        # self.criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)
        # self.loss_sm = nn.LogSoftmax(dim=0)

    def forward(self, x):
        return self.model.forward(x)

    def reconstruction_loss(self, y_hat, y):
        # library_size_hat = torch.expm1(y_hat).sum(axis=0) * y_hat.shape[0]
        # library_size_hat = y_hat.sum(axis=0) * y_hat.shape[0]
        # library_size = self.model.library_size * torch.ones_like(library_size_hat)
        # y_hat = self.loss_sm(y_hat)
        # y_hat = nn.LogSoftmax(y, dim=0)
        # loss = (1 - F.concordance_corrcoef(y_hat, y).mean() + F.mean_squared_error(y_hat, y)) / 2
        # loss = F.mean_absolute_error(y_hat, y)
        # loss = 1 - F.pearson_corrcoef(y_hat, y).mean()
        # loss = 1 - F.concordance_corrcoef(y_hat, y).mean()
        # return ks_loss(y_hat, y)
        # return (1 - F.pearson_corrcoef(y_hat, y).mean() + F.mean_squared_error(y_hat, y)) / 2
        # return self.criterion(y_hat, y.long())
        return F.mean_squared_error(y_hat, y)
        # return F.mean_squared_error(y_hat, y) + F.mean_squared_error(library_size_hat, library_size)
        # return F.mean_squared_error(y_hat, y) + torch.log1p(F.mean_squared_error(torch.expm1(y_hat).sum(axis=0), torch.expm1(y).sum(axis=0)))
        # return F.mean_squared_error(y_hat, y) + F.mean_squared_error(y_hat.sum(axis=0), y.sum(axis=0)) / y.shape[0]
        # return F.mean_absolute_error(y_hat, y)
        # return F.kl_divergence(y_hat, y, log_prob=True, reduction='mean')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.reconstruction_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.reconstruction_loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        # _, y_hat = torch.max(y_hat, dim=1)
        return y_hat

    def configure_optimizers(self):
        # return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        return optim.AdamW(self.model.parameters(), lr=0.001)


class SingleSCVIDecoderEmbedding2ExpressionModel(LightningModule):
    def __init__(
        self,
        in_dim,
        out_dim,
        library_size=1,
        gene_likelihood="nbm",
    ):
        super().__init__()
        self.library_size = torch.scalar_tensor(library_size)
        self.gene_likelihood = gene_likelihood
        self.model = nn.ModuleDict(
            {
                "px_scale_decoder": nn.Sequential(
                    nn.Linear(in_features=in_dim, out_features=out_dim), nn.Softplus()
                ),
                "px_r_decoder": nn.Linear(in_features=in_dim, out_features=out_dim),
                "px_scale2_decoder": nn.Sequential(
                    nn.Linear(in_features=in_dim, out_features=out_dim), nn.Softplus()
                ),
                "px_r2_decoder": nn.Linear(in_features=in_dim, out_features=out_dim),
                "px_dropout_decoder": nn.Linear(
                    in_features=in_dim, out_features=out_dim
                ),
            }
        )

    def forward(self, x):
        px_scale = self.model["px_scale_decoder"].forward(x)
        px_rate = px_scale * self.library_size / 2

        px_r = self.model["px_r_decoder"].forward(x)
        px_r = torch.exp(px_r)

        px_scale2 = self.model["px_scale2_decoder"].forward(x)
        px_rate2 = px_scale2 * self.library_size / 2

        px_r2 = self.model["px_r2_decoder"].forward(x)
        px_r2 = torch.exp(px_r2)

        px_dropout = self.model["px_dropout_decoder"].forward(x)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "nbm":
            px = NegativeBinomialMixture(
                mu1=px_rate,
                mu2=px_rate2,
                theta1=px_r,
                theta2=px_r2,
                mixture_logits=px_dropout,
            )
            px.theta2 = px_r2
        # elif self.gene_likelihood == "poisson":
        #     px = Poisson(px_rate, scale=px_scale)
        else:
            raise ValueError(f"Gene-likelihood {self.gene_likelihood} not recognized")
        return px

    def reconstruction_loss(self, y_hat, y):
        # # Undo log1p
        # y = torch.expm1(y)
        return F.mean_squared_error(y_hat.mean, y)  # -y_hat.log_prob(y).sum(-1) # +

    def sample(self, y_hat):
        y_hat = y_hat.mean
        # y_hat = y_hat.scale * self.library_size
        # y_hat = torch.log1p(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.reconstruction_loss(y_hat, y).mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.reconstruction_loss(y_hat, y).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return self.sample(y_hat)

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
