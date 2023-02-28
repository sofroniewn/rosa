from collections import OrderedDict
from typing import Union

import torch
import torch.nn as nn

from ...utils.config import ModelConfig
from .components import (
    FeedForward,
    InputEmbed,
    expression_head_factory,
)


class RosaSingleModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: ModelConfig,
    ):
        super(RosaSingleModel, self).__init__()

        if config.layer_norm:
            layer_norm_nn = nn.LayerNorm(in_dim)  # type: nn.Module
        else:
            layer_norm_nn = nn.Identity()

        # Determine embedding dimension if embedding provided
        if config.input_embed is None:
            embedding_dim = in_dim
            input = nn.Identity()  # type: nn.Module
        else:
            embedding_dim = config.input_embed.embedding_dim
            input = InputEmbed(in_dim, embedding_dim, config=config.input_embed)

        # Add feed forward layer if provided
        if config.feed_forward is None:
            feed_forward = nn.Identity()  # type: nn.Module
        else:
            feed_forward = FeedForward(
                embedding_dim, embedding_dim, config=config.feed_forward
            )

        # model_config.loss is cross_entropy then figure out n_bins .....
        head = expression_head_factory(
            embedding_dim,
            out_dim,
            config=config.expression_head,
        )
        self.sample = head.sample

        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("layer_norm", layer_norm_nn),
                    ("input_embed", input),
                    ("feed_forward", feed_forward),
                    ("dropout", nn.Dropout(config.dropout_prob)),
                    ("expression_head", head),
                ]
            )
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, torch.distributions.Distribution]:
        return self.main(x)
