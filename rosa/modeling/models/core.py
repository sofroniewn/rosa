import torch
import torch.nn as nn
from math import sqrt


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
):
    dim_k = query.shape[-1]
    scores = torch.einsum("b i j, b k j -> b i k", query, key) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(
            ~mask.unsqueeze(dim=-2).expand_as(scores), -torch.inf
        )
    scores = torch.softmax(scores, dim=-1)
    return torch.einsum("b i j, b j k -> b i k", scores, value)


class CrossAttentionHead(nn.Module):
    def __init__(
        self,
        embed_dim_key: int,
        head_dim_key: int,
        embed_dim_value: int,
        head_dim_value: int,
    ):
        super(CrossAttentionHead, self).__init__()
        self.q = nn.Linear(embed_dim_key, head_dim_key)
        self.k = nn.Linear(embed_dim_key, head_dim_key)
        self.v = nn.Linear(embed_dim_value, head_dim_value)

    def forward(self, hidden_state_key, hidden_state_value, mask=None):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state_key),
            self.k(hidden_state_key),
            self.v(hidden_state_value),
            mask=mask,
        )
        return attn_outputs


class MultiHeadCrossAttention(nn.Module):
    def __init_(self, embed_dim_key:int, embed_dim_value:int, num_heads:int):
        super(MultiHeadCrossAttention, self).__init__()

        head_dim_key = embed_dim_key // num_heads
        head_dim_value = embed_dim_value // num_heads

        self.heads = nn.ModuleList(
            [CrossAttentionHead(embed_dim_key, head_dim_key, embed_dim_value, head_dim_value) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim_value, embed_dim_value)

    def forward(self, hidden_state_key, hidden_state_value, mask=None):
        x = torch.cat([h(hidden_state_key, hidden_state_value, mask=mask) for h in self.heads], dim=-1)
        return self.output_linear(x)


class Core(nn.Module):
    def __init__(self, dim):
        super(Core, self).__init__()
        self.core = MultiHeadCrossAttention(3072, 320, 20)

    def forward(self, x, y, mask=None):
        return self.core(x, y, mask=mask)
