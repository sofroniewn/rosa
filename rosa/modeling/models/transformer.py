from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from performer_pytorch import Performer

from ...utils.config import ModelConfig
from .components import BinnedEmbed, LinearEmbed, LinearHead


class RosaTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        config: ModelConfig,
    ):
        super(RosaTransformer, self).__init__()

        # Create expression embedding
        self.expression_embedding = BinnedEmbed(
            config.n_bins + 1, config=config.expression_embed
        )

        # Create var embedding
        # self.var_embedding = LinearEmbed(in_dim, config=config.var_embed)
        self.var_embedding = BinnedEmbed(19431, config=config.var_embed)
        self.var_embedding.requires_grad_(False)

        # Add transformer if using
        if config.transformer.depth == 0:
            self.transformer = None  # type: Optional[nn.Module]
        else:
            self.transformer = Performer(
                dim=config.transformer.dim,
                depth=config.transformer.depth,
                heads=config.transformer.heads,
                dim_head=config.transformer.dim_head,
                ff_dropout=config.transformer.dropout,
                attn_dropout=config.transformer.dropout,
            )

        self.expression_head = LinearHead(
            config.dim,
            1,
            config=config.expression_head,
        )

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, torch.distributions.Distribution]:
        # Embedd expression and var
        expression = self.expression_embedding(batch["expression_input"])
        var = self.var_embedding(batch["var_input"])

        # Sum embeddings
        x = expression + var

        # attention mask is true for values where attention can look,
        # false for values that should be ignored
        if self.transformer is not None:
            x = self.transformer(x, mask=~batch["mask"])  # type: ignore

        return self.expression_head(x)  # type: ignore
    
    def base_parameters(self):
        return [
                {'params': self.expression_embedding.parameters()},
                {'params': self.transformer.parameters()},
                {'params': self.expression_head.parameters()},
                ]

    def var_parameters(self):
        return {'params': self.var_embedding.parameters()}
