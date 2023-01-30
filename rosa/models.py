from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn

from .components import ExpressionHead, FeedForward, InputEmbed, ParallelEmbed, join_embeds_factory
from .config import ModelConfig


class RosaSingleModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: ModelConfig,
        n_bins: int = 1,
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
        head = ExpressionHead(
            embedding_dim,
            out_dim,
            config=config.expression_head,
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class RosaJointModel(nn.Module):
    def __init__(
        self,
        in_dim: Tuple[int, int],
        out_dim: int,
        config: ModelConfig,
        n_bins: int = 1,
    ):
        super(RosaJointModel, self).__init__()

        if config.layer_norm:
            layer_norm_nn_0 = nn.LayerNorm(in_dim[0])  # type: nn.Module
        else:
            layer_norm_nn_0 = nn.Identity()

        if config.layer_norm_1:
            layer_norm_nn_1 = nn.LayerNorm(in_dim[1])  # type: nn.Module
        else:
            layer_norm_nn_1 = nn.Identity()

        # Determine embedding dimension if embedding provided for first input
        if config.input_embed is None:
            embedding_dim_0 = in_dim[0]
            input_embed_0 = nn.Identity()  # type: nn.Module
        else:
            embedding_dim_0 = config.input_embed.embedding_dim
            input_embed_0 = InputEmbed(
                in_dim[0], embedding_dim_0, config=config.input_embed
            )

        # Determine embedding dimension if embedding provided for second input
        if config.input_embed_1 is None:
            embedding_dim_1 = in_dim[1]
            input_embed_1 = nn.Identity()  # type: nn.Module
        else:
            embedding_dim_1 = config.input_embed_1.embedding_dim
            input_embed_1 = InputEmbed(
                in_dim[1], embedding_dim_1, config=config.input_embed_1
            )

        input_embeds = (
            nn.Sequential(
                OrderedDict(
                    [
                        ("layer_norm_0", layer_norm_nn_0),
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
        )
        dual_embed = ParallelEmbed(input_embeds)
        if config.join_embeds is None:
            raise ValueError(f'A join embedding method must be specified for a joint model')
        join_embeds = join_embeds_factory((embedding_dim_0, embedding_dim_1), config.join_embeds)
        embedding_dim = join_embeds.out_dim

        # Add feed forward layer if provided
        if config.feed_forward is None:
            feed_forward = nn.Identity()  # type: nn.Module
        else:
            feed_forward = FeedForward(
                embedding_dim, embedding_dim, config=config.feed_forward
            )

        # model_config.loss is cross_entropy then figure out n_bins .....
        head = ExpressionHead(
            embedding_dim,
            out_dim,
            config=config.expression_head,
        )

        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("dual_embed", dual_embed),
                    ("join_embeds", join_embeds),
                    ("feed_forward", feed_forward),
                    ("dropout", nn.Dropout(config.dropout_prob)),
                    ("expression_head", head),
                ]
            )
        )

    def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.main(x)