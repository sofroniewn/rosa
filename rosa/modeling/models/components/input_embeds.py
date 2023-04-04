from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ....utils.config import InputEmbedConfig  # type: ignore


class LinearEmbed(nn.Module):
    def __init__(self, in_dim: int, config: InputEmbedConfig) -> None:
        super(LinearEmbed, self).__init__()

        out_dim = config.dim

        if config.pre_layer_norm:
            pre_layer_norm_nn = nn.LayerNorm(in_dim)  # type: nn.Module
        else:
            pre_layer_norm_nn = nn.Identity()

        if config.layer_norm:
            layer_norm_nn = nn.LayerNorm(out_dim)  # type: nn.Module
        else:
            layer_norm_nn = nn.Identity()

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("pre_layer_norm", pre_layer_norm_nn),
                    ("projection", nn.Linear(in_dim, out_dim)),
                    ("layer_norm", layer_norm_nn),
                    ("dropout", nn.Dropout(config.dropout_prob)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BinnedEmbed(nn.Module):
    def __init__(self, in_dim: int, config: InputEmbedConfig) -> None:
        super(BinnedEmbed, self).__init__()

        out_dim = config.dim

        if config.pre_layer_norm is not False:
            raise ValueError("Pre layer norm must be false for binned embedding")

        if config.layer_norm:
            layer_norm_nn = nn.LayerNorm(out_dim)  # type: nn.Module
        else:
            layer_norm_nn = nn.Identity()

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("embedding", nn.Embedding(in_dim, out_dim)),
                    ("layer_norm", layer_norm_nn),
                    ("dropout", nn.Dropout(config.dropout_prob)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
