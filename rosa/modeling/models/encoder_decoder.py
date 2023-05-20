from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn

from ...utils.config import ModelConfig
from .components import BinnedEmbed, LinearEmbed, LinearHead, sinusoidal_embedding, numerical_embedding
from performer_pytorch import Performer, CrossAttention


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, activation=None, glu=False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class RosaTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        config: ModelConfig,
    ):
        super(RosaTransformer, self).__init__()

        self.expression_embedding = partial(sinusoidal_embedding, dim=config.expression_embed.dim, nbins=config.n_bins)
        self.var_embedding = nn.Linear(in_dim, config.var_embed.dim)

        self.embedding_layer_norm_encoder = nn.LayerNorm(config.var_embed.dim)
        self.embedding_dropout_encoder = nn.Dropout(config.var_embed.dropout_prob)
        self.embedding_layer_norm_decoder = nn.LayerNorm(config.var_embed.dim)
        self.embedding_dropout_decoder = nn.Dropout(config.var_embed.dropout_prob)

        self.encoder = Performer(
            dim=config.transformer.dim,
            depth=config.transformer.depth,
            heads=config.transformer.heads,
            dim_head=config.transformer.dim_head,
            ff_dropout=config.transformer.dropout,
            attn_dropout=config.transformer.dropout,
        )

        self.cross_attention_decoder = CrossAttention(
            config.transformer.dim,
            heads=config.transformer.heads,
            dim_head=config.transformer.dim_head,
            dropout=config.transformer.dropout,
        )
        self.cross_attention_layer_norm_decoder = nn.LayerNorm(config.var_embed.dim)
        self.feed_forward_decoder = FeedForward(
                config.transformer.dim,
                mult=4,
                dropout=config.transformer.dropout,
                glu=config.transformer.dropout,
            )

        self.output_norm = nn.LayerNorm(config.transformer.dim)
        self.expression_head = nn.Linear(config.transformer.dim, config.n_bins)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, torch.distributions.Distribution]:
        # Embedd expression and genes
        expression = self.expression_embedding(batch["expression_encoder"])
        gene_encoder = self.var_embedding(batch["var_input_encoder"])
        gene_decoder = self.var_embedding(batch["var_input_decoder"])

        # Encoder
        gene_encoder = self.embedding_layer_norm_encoder(gene_encoder + expression)
        gene_encoder = self.embedding_dropout_encoder(gene_encoder)
        context = self.encoder(gene_encoder)

        # Decoder
        gene_decoder = self.embedding_layer_norm_decoder(gene_decoder)
        gene_decoder = self.embedding_dropout_decoder(gene_decoder)
        output = self.cross_attention_decoder(gene_decoder, context=context)
        output = output + gene_decoder
        output = self.cross_attention_layer_norm_decoder(output)
        output = self.feed_forward_decoder(output)

        # Norm and to logits
        output = self.output_norm(output)
        output = self.expression_head(output)
        return output

    # def base_parameters(self):
    #     param = [
    #         {"params": self.expression_params},
    #     ]
    #     if self.transformer is not None:
    #         param.append({"params": self.transformer.parameters()})
    #     return param

    # def var_parameters(self):
    #     return {"params": self.var_embedding.parameters()}
