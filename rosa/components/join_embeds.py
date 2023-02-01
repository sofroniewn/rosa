from typing import Tuple, Union

import torch
import torch.nn as nn

from ..config import (
    JoinEmbedsConfig,
    JoinEmbedsMethods,
)


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


JoinEmbeds = Union[AddEmbeds, AttentionEmbeds, BilinearEmbeds, CatEmbeds]


def join_embeds_factory(in_dim: Tuple[int, int], config: JoinEmbedsConfig) -> JoinEmbeds:
    if config.method == JoinEmbedsMethods.ADD.name.lower():
        return AddEmbeds(in_dim)
    if config.method == JoinEmbedsMethods.CAT.name.lower():
        return CatEmbeds(in_dim)
    if config.method == JoinEmbedsMethods.BILINEAR.name.lower():
        return BilinearEmbeds(in_dim, config.out_dim)
    if config.method == JoinEmbedsMethods.ATTENTION.name.lower():
        return AttentionEmbeds(in_dim, config.out_dim)
    raise ValueError(f"Activation {config.method} not recognized")