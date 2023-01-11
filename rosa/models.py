from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
import torchmetrics.functional as F

from .modules import BilinearHead, ConcatHead, SingleHead


class Embedding2ExpressionModel(LightningModule):
    def __init__(self, in_dim_1, in_dim_2, n_dim_1, n_dim_2, rank, bias=False, head_1='Linear', head_2='Linear', item='joint'):
        super().__init__()
        self.alpha = None
        if item == 'joint':
            self.model = BilinearHead(in_dim_1=in_dim_1, in_dim_2=in_dim_2, rank=rank, bias=bias,
                                      n_dim_1=n_dim_1, n_dim_2=n_dim_2, head_1=head_1, head_2=head_2)
        elif item == 'joint-concat':
            self.model = ConcatHead(in_dim_1=in_dim_1, in_dim_2=in_dim_2, rank=rank, bias=bias,
                                      n_dim_1=n_dim_1, n_dim_2=n_dim_2, head_1=head_1, head_2=head_2)
        elif item == 'cell':
            self.model = SingleHead(in_dim_1, n_dim_2, head=head_1)
        elif item == 'gene':
            self.model = SingleHead(in_dim_2, n_dim_1, head=head_2)
        else:
            raise ValueError(f'Item {item} not recognized')
    
    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mean_squared_error(y_hat, y)
        if self.alpha is not None:
             loss += self.alpha * torch.linalg.norm(self.model.model.weight)**2

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mean_squared_error(y_hat, y)
        if self.alpha is not None:
             loss += self.alpha * (self.model.model.weight @ self.model.model.weight).sum
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # return optim.Adam(self.model.parameters(), lr=0.001)