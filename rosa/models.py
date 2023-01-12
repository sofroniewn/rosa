from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
import torchmetrics.functional as F

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
                head_1=head_1,
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


class GeneEmbedding2ExpressionModel(LightningModule):
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
        loss = F.mean_squared_error(y_hat, y)
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
