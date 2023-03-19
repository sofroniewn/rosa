from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn as nn

from ....utils.config import (
    ExpressionHeadActivations,  # type: ignore
    ExpressionHeadConfig,
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
    ):
        super(ProjectionExpressionHead, self).__init__()
        if config.n_bins is None:
            self.n_bins = 1
        else:
            self.n_bins = config.n_bins

        if self.n_bins > 1 and config.activation is not None:
            raise ValueError(f"An activation should not be used for classification")

        if config.projection:
            projection_nn = nn.Linear(in_dim, out_dim * self.n_bins)  # type: nn.Module
        else:
            if in_dim != out_dim * self.n_bins:
                raise ValueError(
                    f"If no projection is used input dim {in_dim} must match output dim {out_dim * self.n_bins}"
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
        output = self.model(x)
        return output.squeeze(-1)
