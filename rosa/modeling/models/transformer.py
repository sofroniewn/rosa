from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from performer_pytorch import Performer

from ...utils.config import ModelConfig
from .components import (
    InputEmbed,
    MaskedEmbed,
    ParallelEmbed,
    expression_head_factory,
    join_embeds_factory,
)


class RosaTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        config: ModelConfig,
    ):
        super(RosaTransformer, self).__init__()
        # No layer config provided for transformer like models
        assert config.layer_norm is None

        if config.layer_norm_1:
            layer_norm_nn_1 = nn.LayerNorm(in_dim)  # type: nn.Module
        else:
            layer_norm_nn_1 = nn.Identity()

        # Determine embedding dimension if embedding provided for first input
        if config.input_embed is None:
            raise ValueError(
                "An input embedding config must be provided for transformer like models"
            )

        embedding_dim_0 = config.input_embed.embedding_dim
        if config.n_bins is None:
            n_bins = 1
        else:
            n_bins = config.n_bins
        input_embed_0 = MaskedEmbed(n_bins, embedding_dim_0, config=config.input_embed)

        # Determine embedding dimension if embedding provided for second input
        if config.input_embed_1 is None:
            embedding_dim_1 = in_dim
            input_embed_1 = nn.Identity()  # type: nn.Module
        else:
            embedding_dim_1 = config.input_embed_1.embedding_dim
            input_embed_1 = InputEmbed(
                in_dim, embedding_dim_1, config=config.input_embed_1
            )

        input_embeds = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ("input_embed_0", input_embed_0),
                        ]
                    )
                ),
                nn.Sequential(
                    OrderedDict(
                        [
                            ("layer_norm_1", layer_norm_nn_1),
                            ("input_embed_1", input_embed_1),
                        ]
                    )
                ),
            ]
        )
        dual_embed = ParallelEmbed(input_embeds)
        if config.join_embeds is None:
            raise ValueError(
                f"A join embedding method must be specified for a joint model"
            )
        join_embeds = join_embeds_factory(
            (embedding_dim_0, embedding_dim_1), config.join_embeds
        )
        embedding_dim = join_embeds.out_dim

        # Add transformer if provided
        if config.transformer is None:
            transformer = nn.Identity()  # type: nn.Module
        else:
            transformer = Performer(
                dim=config.transformer.dim,
                depth=config.transformer.depth,
                heads=config.transformer.heads,
                dim_head=config.transformer.dim_head,
                ff_dropout=config.transformer.dropout,
                attn_dropout=config.transformer.dropout,
            )

        out_dim = 1
        head = expression_head_factory(
            embedding_dim,
            out_dim,
            config=config.expression_head,
        )
        self.sample = head.sample

        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("dual_embed", dual_embed),
                    ("join_embeds", join_embeds),
                ]
            )
        )
        self.transformer = transformer
        self.dropout = nn.Dropout(config.dropout_prob)
        self.expression_head = head

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, torch.distributions.Distribution]:
        mask = batch['mask']
        x = ((batch['expression_input'], batch['mask']), batch['var_input'])
        x = self.main(x)  # type: ignore
        # attention mask is true for values where attention can look,
        # false for values that should be ignored
        x = self.transformer(x, mask=~mask)   # type: ignore
        x = self.dropout(x)   # type: ignore
        return self.expression_head(x)   # type: ignore
