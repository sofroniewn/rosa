from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...utils.config import ModelConfig
from .components import BinnedEmbed, LinearEmbed, LinearHead
from .core import Core


class RosaTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        config: ModelConfig,
    ):
        super(RosaTransformer, self).__init__()

        # Create expression embedding
        self.expression_params = nn.Parameter(
            torch.randn(config.n_bins + 1, config.transformer.dim)
        )
        self.expression_embedding = nn.Embedding.from_pretrained(
            self.expression_params, freeze=False
        )

        # self.expression_embedding = BinnedEmbed(
        #     config.n_bins + 1, config=config.expression_embed
        # )

        # Create var embedding
        if config.var_embed is None:
            self.var_embedding = nn.Identity(3072)
        else:
            self.var_embedding = LinearEmbed(in_dim, config=config.var_embed)
        # self.var_embedding = BinnedEmbed(19431, config=config.var_embed)
        # self.var_embedding.requires_grad_(False)

        # Add transformer if using
        if config.transformer.depth == 0:
            self.transformer = None  # type: Optional[nn.Module]
        else:
            self.transformer = Core(dim=config.transformer.dim)

        # self.expression_head = nn.Linear(config.transformer.dim, config.n_bins)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, torch.distributions.Distribution]:
        # Embedd expression and var
        expression = self.expression_embedding(batch["expression_input"])
        var = self.var_embedding(batch["var_input"])

        # attention mask is true for values where attention can look,
        # false for values that should be ignored
        if self.transformer is not None:
            # x = self.transformer(var, expression)
            expression = self.transformer(var, expression, mask=~batch["mask"])

        expression = torch.einsum(
            "b i j, k j -> b i k", expression, self.expression_params[:-1, :]
        )
        # expression = self.expression_head(expression)
        return expression

    # def base_parameters(self):
    #     param = [
    #         {"params": self.expression_params},
    #     ]
    #     if self.transformer is not None:
    #         param.append({"params": self.transformer.parameters()})
    #     return param

    # def var_parameters(self):
    #     return {"params": self.var_embedding.parameters()}
