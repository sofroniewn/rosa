from collections import OrderedDict
from typing import Union

import torch
import torch.nn as nn
from scvi.distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
    ZeroInflatedNegativeBinomial,
)

from ...utils.config import (
    ExpressionHeadActivations,
    ExpressionHeadConfig,
    ExpressionHeadLikelihood,
)


class ProjectionExpressionHead(nn.Module):
    """
    Go from a latent space to expression
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        config: ExpressionHeadConfig,
        n_bins: int = 1,
    ):
        super(ProjectionExpressionHead, self).__init__()
        if n_bins > 1 and config.activation is not None:
            raise ValueError(f"An activation should not be used for classification")

        if config.projection:
            projection_nn = nn.Linear(in_dim, out_dim * n_bins)  # type: nn.Module
        else:
            if in_dim != out_dim * n_bins:
                raise ValueError(
                    f"If no projection is used input dim {in_dim} must match output dim {out_dim * n_bins}"
                )
            projection_nn = nn.Identity()

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

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ZeroInflatedNegativeBinomialExpressionHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        config: ExpressionHeadConfig,
    ):
        super().__init__()
        if config.library_size is None:
            raise ValueError(
                f"Expression head library size must be provided for likelihood model"
            )
        self.library_size = torch.scalar_tensor(config.library_size)

        self.model = nn.ModuleDict(
            {
                "px_scale": nn.Sequential(nn.Linear(in_dim, out_dim), nn.Softplus()),
                "px_r": nn.Linear(in_dim, out_dim),
                "px_dropout": nn.Linear(in_dim, out_dim),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        px_scale = self.model["px_scale"].forward(x)
        px_rate = px_scale * self.library_size
        px_r = torch.exp(self.model["px_r"].forward(x))
        px_dropout = self.model["px_dropout"].forward(x)

        return ZeroInflatedNegativeBinomial(
            mu=px_rate,
            theta=px_r,
            zi_logits=px_dropout,
            scale=px_scale,
        )

    def sample(self, x: torch.distributions.Distribution) -> torch.Tensor:
        return x.mean


class NegativeBinomialExpressionHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        config: ExpressionHeadConfig,
    ):
        super().__init__()
        if config.library_size is None:
            raise ValueError(
                f"Expression head library size must be provided for likelihood model"
            )
        self.library_size = torch.scalar_tensor(config.library_size)

        self.model = nn.ModuleDict(
            {
                "px_scale": nn.Sequential(nn.Linear(in_dim, out_dim), nn.Softplus()),
                "px_r": nn.Linear(in_dim, out_dim),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        px_scale = self.model["px_scale"].forward(x)
        px_rate = px_scale * self.library_size
        px_r = torch.exp(self.model["px_r"].forward(x))

        return NegativeBinomial(
            mu=px_rate,
            theta=px_r,
            scale=px_scale,
        )

    def sample(self, x: torch.distributions.Distribution) -> torch.Tensor:
        return x.mean


class NegativeBinomialMixtureExpressionHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        config: ExpressionHeadConfig,
    ):
        super().__init__()
        if config.library_size is None:
            raise ValueError(
                f"Expression head library size must be provided for likelihood model"
            )
        self.library_size = torch.scalar_tensor(config.library_size)

        self.model = nn.ModuleDict(
            {
                "px_scale_1": nn.Sequential(nn.Linear(in_dim, out_dim), nn.Softplus()),
                "px_r_1": nn.Linear(in_dim, out_dim),
                "px_scale_2": nn.Sequential(nn.Linear(in_dim, out_dim), nn.Softplus()),
                "px_r_2": nn.Linear(in_dim, out_dim),
                "px_mixture": nn.Linear(in_dim, out_dim),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        px_scale_1 = self.model["px_scale_1"].forward(x)
        px_rate_1 = px_scale_1 * self.library_size
        px_r_1 = torch.exp(self.model["px_r_1"].forward(x))

        px_scale_2 = self.model["px_scale_2"].forward(x)
        px_rate_2 = px_scale_2 * self.library_size
        px_r_2 = torch.exp(self.model["px_r_2"].forward(x))

        px_mixture = self.model["px_mixture"].forward(x)

        nbm = NegativeBinomialMixture(
            mu1=px_rate_1,
            mu2=px_rate_2,
            theta1=px_r_1,
            theta2=px_r_2,
            mixture_logits=px_mixture,
        )
        nbm.theta2 = px_r_2  # hack to fix scvi-tools bug
        return nbm

    def sample(self, x: torch.distributions.Distribution) -> torch.Tensor:
        return x.mean


ExpressionHead = Union[
    ProjectionExpressionHead,
    NegativeBinomialExpressionHead,
    ZeroInflatedNegativeBinomialExpressionHead,
    NegativeBinomialMixtureExpressionHead,
]


def expression_head_factory(
    in_dim: int, out_dim: int, config: ExpressionHeadConfig
) -> ExpressionHead:
    if config.likelihood is None:
        return ProjectionExpressionHead(in_dim, out_dim, config)
    if config.likelihood == ExpressionHeadLikelihood.NB.name.lower():
        return NegativeBinomialExpressionHead(in_dim, out_dim, config)
    if config.likelihood == ExpressionHeadLikelihood.ZINB.name.lower():
        return ZeroInflatedNegativeBinomialExpressionHead(in_dim, out_dim, config)
    if config.likelihood == ExpressionHeadLikelihood.NBM.name.lower():
        return NegativeBinomialMixtureExpressionHead(in_dim, out_dim, config)
    raise ValueError(f"Unrecongnized expression head likelihood {config.likelihood}")
