from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics.functional as F
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


class SingleEmbedding2ExpressionModel(LightningModule):
    def __init__(
        self,
        in_dim,
        out_dim,
        head="Linear",
    ):
        super().__init__()
        self.model = SingleHead(in_dim, out_dim, head=head)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = (1 - F.pearson_corrcoef(y_hat, y).mean() + F.mean_squared_error(y_hat, y)) / 2
        loss = F.mean_squared_error(y_hat, y)
        # loss = F.mean_absolute_error(y_hat, y)
        # loss = 1 - F.pearson_corrcoef(y_hat, y).mean()
        # loss = 1 - F.concordance_corrcoef(y_hat, y).mean()
        # loss = (1 - F.concordance_corrcoef(y_hat, y).mean() + F.mean_squared_error(y_hat, y)) / 2
        # loss = F.kl_divergence(y_hat, y, log_prob=False, reduction='mean')
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mean_squared_error(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)


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
