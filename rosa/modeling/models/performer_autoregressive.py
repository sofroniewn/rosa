from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn

from ...utils.config import ModelConfig
from .components import BinnedEmbed, LinearEmbed, LinearHead
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


class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class RosaTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        config: ModelConfig,
    ):
        super(RosaTransformer, self).__init__()

        # Create expression embedding
        # self.expression_params = nn.Parameter(
        #     torch.randn(config.n_bins + 1, config.transformer.dim)
        # )
        # self.expression_embedding = nn.Embedding.from_pretrained(
        #     self.expression_params, freeze=False
        # )

        self.expression_embedding = BinnedEmbed(
            config.n_bins, config=config.expression_embed
        )

        # Create var embedding
        if config.var_embed is None:
            self.var_embedding = nn.Identity(3072)
        else:
            self.var_embedding = LinearEmbed(in_dim, config=config.var_embed)

        # self.var_embedding = nn.Linear(in_dim, config.var_embed.dim)

        # self.var_embedding = BinnedEmbed(19431, config=config.var_embed)
        # self.var_embedding.requires_grad_(False)

        # Add transformer if using
        # if config.transformer.depth == 0:
        #     self.transformer = None  # type: Optional[nn.Module]
        # else:
        self.transformer = Performer(
            dim=config.transformer.dim,
            depth=config.transformer.depth,
            heads=config.transformer.heads,
            dim_head=config.transformer.dim_head,
            ff_dropout=config.transformer.dropout,
            attn_dropout=config.transformer.dropout,
            causal=config.transformer.causal,
        )

        wrapper_fn = partial(PreLayerNorm, config.transformer.dim)
        self.pre_layer_norm_cross_attention = nn.LayerNorm(config.transformer.dim)
        self.pre_layer_norm_cross_attention_context = nn.LayerNorm(
            config.transformer.dim
        )
        self.cross_attention = CrossAttention(
            config.transformer.dim,
            heads=config.transformer.heads,
            dim_head=config.transformer.dim_head,
            dropout=config.transformer.dropout,
            causal=config.transformer.causal,
        )
        self.feed_forward = wrapper_fn(
            FeedForward(
                config.transformer.dim,
                mult=4,
                dropout=config.transformer.dropout,
                glu=config.transformer.dropout,
            )
        )
        self.expression_head = nn.Linear(config.transformer.dim, config.n_bins)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, torch.distributions.Distribution]:
        # Embedd expression and var
        expression = self.expression_embedding(batch["expression_target"])
        var = self.var_embedding(batch["var_input"])

        # attention mask is true for values where attention can look,
        # false for values that should be ignored
        input = var + expression
        # input = torch.concat([var, expression], dim=-1)
        output = self.transformer(input)

        # Layer norms for cross attention
        output = self.pre_layer_norm_cross_attention_context(output)
        var_input = self.pre_layer_norm_cross_attention(var)
        # var_input = var

        # Shifted cross attention
        output = self.cross_attention(var_input[:, 1:], context=output[:, :-1])
        output = torch.cat((var_input[:, 0][:, None, :], output), dim=1)
        # output = self.cross_attention(var_input, context=output)

        # Feed forward after cross attention
        output = self.feed_forward(output)

        # expression = torch.einsum(
        #     "b i j, k j -> b i k", expression, self.expression_params[:-1, :]
        # )
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
