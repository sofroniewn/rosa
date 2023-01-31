from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn as nn

from .config import (
    ExpressionHeadActivations,
    ExpressionHeadConfig,
    FeedForwardConfig,
    InputEmbedConfig,
    JoinEmbedsConfig,
    JoinEmbedsMethods,
)

# from scvi.distributions import (NegativeBinomial, NegativeBinomialMixture,
#                                 ZeroInflatedNegativeBinomial)


class ExpressionHead(nn.Module):
    """
    Go from a latent space to expression
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        config: ExpressionHeadConfig,
        n_bins: int = 1,
    ):
        super(ExpressionHead, self).__init__()
        if n_bins > 1 and config.activation is not None:
            raise ValueError(f"An activation should not be used for classification")

        projection_nn = nn.Linear(input_dim, output_dim * n_bins)

        if config.activation is None:
            activation_nn = nn.Identity()  # type: nn.Module
        elif config.activation == ExpressionHeadActivations.SOFTPLUS.name.lower():
            activation_nn = nn.Softplus()
        elif config.activation == ExpressionHeadActivations.SOFTMAX.name.lower():
            activation_nn = nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Activation {config.activation} not recognized")

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("projection", projection_nn),
                    ("activation", activation_nn),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class InputEmbed(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: InputEmbedConfig) -> None:
        super(InputEmbed, self).__init__()

        if config.layer_norm:
            layer_norm_nn = nn.LayerNorm(out_dim)  # type: nn.Module
        else:
            layer_norm_nn = nn.Identity()

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("projection", nn.Linear(in_dim, out_dim)),
                    ("layer_norm", layer_norm_nn),
                    ("dropout", nn.Dropout(config.dropout_prob)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: FeedForwardConfig) -> None:
        super(FeedForward, self).__init__()
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("linear_1", nn.Linear(in_dim, config.hidden_dim)),
                    ("gelu", nn.GELU()),
                    ("linear_2", nn.Linear(config.hidden_dim, out_dim)),
                    ("dropout", nn.Dropout(config.dropout_prob)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ParallelEmbed(nn.Module):
    def __init__(self, models: nn.ModuleList) -> None:
        super(ParallelEmbed, self).__init__()
        self.models = models

    def forward(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple([f(x_) for f, x_ in zip(self.models, x)])


class AddEmbeds(nn.Module):
    def __init__(self, in_dim: Tuple[int, int]) -> None:
        super(AddEmbeds, self).__init__()
        if in_dim[0] != in_dim[1]:
            raise ValueError(f'Embeddings must have same dimensions for add method, got {in_dim}')
        self.out_dim = in_dim[0]

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return torch.add(*x)


class CatEmbeds(nn.Module):
    def __init__(self, in_dim: Tuple[int, int]) -> None:
        super(CatEmbeds, self).__init__()
        self.out_dim = in_dim[0] + in_dim[1]

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, dim=-1)


class BilinearEmbeds(nn.Module):
    def __init__(self, in_dim: Tuple[int, int], out_dim) -> None:
        super(BilinearEmbeds, self).__init__()
        self.model = nn.Bilinear(in_dim[0], in_dim[1], out_dim)
        self.out_dim = out_dim

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.model(*x)


class AttentionEmbeds(nn.Module):
    def __init__(self, in_dim: Tuple[int, int], out_dim) -> None:
        super(AttentionEmbeds, self).__init__()
        if in_dim[0] != in_dim[1]:
            raise ValueError(f'Embeddings must have same dimensions for attention method, got {in_dim}')

        self.value = nn.Parameter(torch.randn(out_dim))
        self.activation = nn.GELU()
        self.out_dim = out_dim

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        atten = self.activation(torch.einsum('...i, ...i ->...', *x))
        return torch.einsum('..., i -> ...i', atten, self.value)


def join_embeds_factory(in_dim: Tuple[int, int], config: JoinEmbedsConfig) -> Union[AddEmbeds, AttentionEmbeds, BilinearEmbeds, CatEmbeds]:
    if config.method == JoinEmbedsMethods.ADD.name.lower():
        return AddEmbeds(in_dim)
    if config.method == JoinEmbedsMethods.CAT.name.lower():
        return CatEmbeds(in_dim)
    if config.method == JoinEmbedsMethods.BILINEAR.name.lower():
        return BilinearEmbeds(in_dim, config.out_dim)
    if config.method == JoinEmbedsMethods.ATTENTION.name.lower():
        return AttentionEmbeds(in_dim, config.out_dim)
    raise ValueError(f"Activation {config.method} not recognized")



# class BilinearHead(nn.Module):
#     def __init__(
#         self,
#         in_dim_1,
#         in_dim_2,
#         rank,
#         head_1="linear",
#         head_2="linear",
#     ):
#         super(BilinearHead, self).__init__()
#         self.fc1 = SingleHead(in_dim_1, rank, head=head_1)
#         self.fc2 = SingleHead(in_dim_2, rank, head=head_2)
#         self.act = nn.Softplus()

#     def forward(self, x):
#         x1, x2 = x
#         x1 = self.fc1(x1)
#         x2 = self.fc2(x2)
#         x = (x1 * x2).sum(-1)
#         # x = self.act(x)
#         return x


# class ConcatHead(nn.Module):
#     def __init__(
#         self,
#         in_dim_1,
#         in_dim_2,
#         head="linear",
#     ):
#         super(ConcatHead, self).__init__()
#         self.fc = SingleHead(in_dim_1 + in_dim_2, 1, head=head)
#         self.act = nn.Softplus()

#     def forward(self, x):
#         x1, x2 = x
#         shape = list(x1.shape)
#         shape[-1] += x2.shape[-1]
#         x = torch.concat((x1, x2), -1)
#         x = self.fc(x).reshape(shape[:-1])
#         # x = self.act(x)
#         return x


# class SingleHead(nn.Module):
#     def __init__(self, in_dim, out_dim, head="linear"):
#         super(SingleHead, self).__init__()
#         self.head = head
#         self.out_dim = out_dim
#         if self.head == "linear":
#             # self.conv = nn.Conv2d(1, 10, (896, 1))
#             self.fc = nn.Linear(in_features=in_dim, out_features=out_dim)
#         elif self.head == "MLP":
#             self.fc = MLP(in_dim=in_dim, out_dim=out_dim, dropout=0.5)
#         elif self.head == "OneHot":
#             self.fc = nn.Embedding(in_dim, out_dim)
#         elif self.head == "Identity":
#             self.fc = nn.Identity()
#         else:
#             raise ValueError(f"Model type {self.head} not recognized")

#         # self.act1 = nn.ReLU()
#         # self.fc2 = nn.Linear(out_dim, 128 * out_dim)
#         # self.norm = nn.BatchNorm1d(out_dim)
#         # self.act = nn.Softplus()
#         # self.act = nn.Sigmoid()
#         # self.library_size = nn.Parameter(torch.tensor(0.5))
#         # self.act = nn.Softmax(dim=-1)
#         # self.mult = nn.Parameter(torch.tensor(0.7*1e5))
#         # self.mult = nn.Parameter(torch.tensor(1.0))

#     def forward(self, x):
#         # x = x.unsqueeze(1)
#         # x = self.conv(x)
#         # x = x.view(x.shape[0], -1)
#         x = self.fc(x)
#         # x = self.act1(x)
#         # # x = self.norm(x)
#         # x = self.fc2(x)
#         # x = x.view(-1, 128, self.out_dim)
#         # x = torch.exp(x)
#         # x = self.act(x)
#         # # x = x * x.shape[0] * self.mult
#         # x = x * self.mult
#         # x = torch.log1p(x)
#         return x

# class SingleSCVIDecoderEmbedding2ExpressionModel(LightningModule):
#     def __init__(
#         self,
#         in_dim,
#         out_dim,
#         library_size=1,
#         gene_likelihood="nbm",
#     ):
#         super().__init__()
#         self.library_size = torch.scalar_tensor(library_size)
#         self.gene_likelihood = gene_likelihood
#         self.model = nn.ModuleDict(
#             {
#                 "px_scale_decoder": nn.Sequential(
#                     nn.Linear(in_features=in_dim, out_features=out_dim), nn.Softplus()
#                 ),
#                 "px_r_decoder": nn.Linear(in_features=in_dim, out_features=out_dim),
#                 "px_scale2_decoder": nn.Sequential(
#                     nn.Linear(in_features=in_dim, out_features=out_dim), nn.Softplus()
#                 ),
#                 "px_r2_decoder": nn.Linear(in_features=in_dim, out_features=out_dim),
#                 "px_dropout_decoder": nn.Linear(
#                     in_features=in_dim, out_features=out_dim
#                 ),
#             }
#         )

#     def forward(self, x):
#         px_scale = self.model["px_scale_decoder"].forward(x)
#         px_rate = px_scale * self.library_size / 2

#         px_r = self.model["px_r_decoder"].forward(x)
#         px_r = torch.exp(px_r)

#         px_scale2 = self.model["px_scale2_decoder"].forward(x)
#         px_rate2 = px_scale2 * self.library_size / 2

#         px_r2 = self.model["px_r2_decoder"].forward(x)
#         px_r2 = torch.exp(px_r2)

#         px_dropout = self.model["px_dropout_decoder"].forward(x)

#         if self.gene_likelihood == "zinb":
#             px = ZeroInflatedNegativeBinomial(
#                 mu=px_rate,
#                 theta=px_r,
#                 zi_logits=px_dropout,
#                 scale=px_scale,
#             )
#         elif self.gene_likelihood == "nb":
#             px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
#         elif self.gene_likelihood == "nbm":
#             px = NegativeBinomialMixture(
#                 mu1=px_rate,
#                 mu2=px_rate2,
#                 theta1=px_r,
#                 theta2=px_r2,
#                 mixture_logits=px_dropout,
#             )
#             px.theta2 = px_r2
#         # elif self.gene_likelihood == "poisson":
#         #     px = Poisson(px_rate, scale=px_scale)
#         else:
#             raise ValueError(f"Gene-likelihood {self.gene_likelihood} not recognized")
#         return px

#     def reconstruction_loss(self, y_hat, y):
#         # # Undo log1p
#         # y = torch.expm1(y)
#         return F.mean_squared_error(y_hat.mean, y)  # -y_hat.log_prob(y).sum(-1) # +

#     def sample(self, y_hat):
#         y_hat = y_hat.mean
#         # y_hat = y_hat.scale * self.library_size
#         # y_hat = torch.log1p(y_hat)
#         return y_hat

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.reconstruction_loss(y_hat, y).mean()
#         self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.reconstruction_loss(y_hat, y).mean()
#         self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
#         return loss

#     def predict_step(self, batch, batch_idx):
#         x, _ = batch
#         y_hat = self(x)
#         return self.sample(y_hat)

#     def configure_optimizers(self):
#         return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
