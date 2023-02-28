from collections import OrderedDict
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn

from ....utils.config import InputEmbedConfig  # type: ignore


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


class MaskedEmbed(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: InputEmbedConfig) -> None:
        super(MaskedEmbed, self).__init__()

        if config.layer_norm:
            layer_norm_nn = nn.LayerNorm(out_dim)  # type: nn.Module
        else:
            layer_norm_nn = nn.Identity()

        self.in_dim = in_dim
        if in_dim == 1:
            embedding = nn.Linear(1, out_dim)  # type: nn.Module
            self.mask = nn.Linear(1, out_dim)  # type: Optional[nn.Module]
        else:
            embedding = nn.Embedding(self.in_dim + 1, out_dim)
            self.mask = None

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("embedding", embedding),
                    ("layer_norm", layer_norm_nn),
                    ("dropout", nn.Dropout(config.dropout_prob)),
                ]
            )
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x0, mask = x
        if self.in_dim == 1 and self.mask is not None:
            output = self.model(x0.unsqueeze_(-1)).squeeze(-2)
            output[mask] = self.mask.weight.T
        else:
            x0[mask] = self.in_dim
            output = self.model(x0)
        return output
