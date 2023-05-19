from collections import OrderedDict
from typing import Optional, Tuple, Union

import math
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


def sinusoidal_embedding(values, dim, nbins):
    half = dim // 2
    freqs = torch.exp(
        -math.log(nbins) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=values.device)
    args = values[..., None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding


def numerical_embedding(values, dim, nbins):
    indices = torch.arange(start=0, end=dim, dtype=torch.float32).to(device=values.device)
    freqs = (-1) ** indices / (indices + 1)
    embedding = values[..., None].float() * freqs[None]
    return embedding