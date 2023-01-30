from collections import OrderedDict

import torch.nn as nn

from .components import ExpressionHead, FeedForward, InputEmbed
from .config import ModelConfig


class RosaSingleModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        config: ModelConfig,
        n_bins: int = 1,
    ):
        super().__init__()
        # if type(in_dim) == tuple:
        #     in_dim_1, in_dim_2 = in_dim
        #     if model_cfg.method == "bilinear":
        #         self.model = BilinearHead(
        #             in_dim_1=in_dim_1,
        #             in_dim_2=in_dim_2,
        #             rank=model_cfg.rank,
        #             head_1=model_cfg.obs_head,
        #             head_2=model_cfg.var_head,
        #         )
        #     elif model_cfg.method == "concat":
        #         self.model = ConcatHead(
        #             in_dim_1=in_dim_1,
        #             in_dim_2=in_dim_2,
        #             head=model_cfg.head,
        #         )
        #     else:
        #         raise ValueError(f"Item {model_cfg.method} not recognized")
        # else:
        #     self.model = SingleHead(in_dim, out_dim, head=model_cfg.head)

        # BREAKUP MODEL CONFIG INTO EXPRESSIONHEAD CONFIG / INPUT EMBED CONFIG .....

        if config.layer_norm:
            layer_norm_nn = nn.LayerNorm(in_dim)  # type: nn.Module
        else:
            layer_norm_nn = nn.Identity()

        # Determine embedding dimension if embedding provided
        input_embed_config = config.input_embed
        if input_embed_config is None:
            embedding_dim = in_dim
            input = nn.Identity()  # type: nn.Module
        else:
            embedding_dim = input_embed_config.embedding_dim
            input = InputEmbed(in_dim, embedding_dim, config=input_embed_config)

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

        self.rosa = nn.Sequential(
            OrderedDict(
                [
                    (
                        "layer_norm",
                        layer_norm_nn,
                    ),  # Note present as enformer implemntation missing layer norm
                    ("input_embed", input),
                    ("feed_forward", feed_forward),
                    ("dropout", nn.Dropout(config.dropout_prob)),
                    ("expression_head", head),
                ]
            )
        )

    def forward(self, x):
        return self.rosa(x)


RosaJointModel = RosaSingleModel

# class EmbeddingJoin(nn.Module)
# SUM = A + B
# CAT = [A, B]
# PROD = A^T * W * B
