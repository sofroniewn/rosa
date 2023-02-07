from collections import OrderedDict

import torch
import torch.nn as nn

from ...utils.config import FeedForwardConfig


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: FeedForwardConfig) -> None:
        super(FeedForward, self).__init__()
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("linear_1", nn.Linear(in_dim, config.hidden_dim)),
                    ("gelu", nn.GELU()),
                    ("linear_2", nn.Linear(config.hidden_dim, out_dim)),
                    ("dropout", nn.Dropout(config.dropout_prob)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
