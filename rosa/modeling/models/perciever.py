from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...utils.config import ModelConfig
from .components import BinnedEmbed, LinearEmbed, LinearHead
from perceiver_pytorch import PerceiverIO


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
            config.n_bins + 1, config=config.expression_embed
        )

        # Create var embedding
        if config.var_embed is None:
            self.var_embedding = nn.Identity(3072)
        else:
            self.var_embedding = LinearEmbed(in_dim, config=config.var_embed)
        # self.var_embedding = BinnedEmbed(19431, config=config.var_embed)
        # self.var_embedding.requires_grad_(False)

        # Add transformer if using
        # if config.transformer.depth == 0:
        #     self.transformer = None  # type: Optional[nn.Module]
        # else:
        self.transformer = PerceiverIO(
            dim = 1024,             # dimension of sequence to be encoded
            queries_dim = 1024,          # dimension of decoder queries
            logits_dim = config.n_bins,  # dimension of final logits
            depth = 10,                   # depth of net
            num_latents = 512,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 512,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
            seq_dropout_prob = 0.2       # fraction of the tokens from the input sequence to dropout (structured dropout, for saving compute and regularizing effects)
        )

        # decoder_cross_attn = self.transformer.decoder_cross_attn
        # decoder_cross_attn.fn.to_q = nn.Identity(3072)
        # decoder_cross_attn.fn.to_kv = nn.Linear(512, 3072 * 2, bias = False)
        # decoder_cross_attn.fn.to_out = nn.Identity(3072) #nn.Linear(3072, 3072)
        # decoder_cross_attn.fn.heads = 48
        # decoder_cross_attn.fn.to_kv.requires_grad_(False)
        # decoder_cross_attn.fn.to_q.requires_grad_(False)
        # decoder_cross_attn.fn.to_out.requires_grad_(False)

        # self.expression_head = nn.Linear(config.transformer.dim, config.n_bins)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, torch.distributions.Distribution]:
        # Embedd expression and var
        expression = self.expression_embedding(batch["expression_input"])
        var = self.var_embedding(batch["var_input"])

        # attention mask is true for values where attention can look,
        # false for values that should be ignored
        input = var + expression
        # input = torch.concat([var, expression], dim=-1)
        queries = var # batch["var_input"]
        expression = self.transformer(input, mask=~batch["mask"], queries=queries)

        # expression = torch.einsum(
        #     "b i j, k j -> b i k", expression, self.expression_params[:-1, :]
        # )
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
