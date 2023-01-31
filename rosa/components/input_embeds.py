from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn as nn

from ..config import InputEmbedConfig


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