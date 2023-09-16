from src.transformer import (
    MultiHeadedAttention,
    FeedForwardNet,
    ResidualConnection,
    LayerNorm,
)
import torch.nn as nn
import copy


class EncoderLayer(nn.Module):
    def __init__(self, dimension, head=8, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadedAttention(dimension, head, dropout)
        self.ffnn = FeedForwardNet(dimension, dropout=dropout)
        self.resconn1 = ResidualConnection(dimension, dropout)
        self.resconn2 = ResidualConnection(dimension, dropout)

        self.norm = LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_vec, mask=None):
        attn = self.resconn1(input_vec, lambda x: self.attn(x, x, x, mask))
        return self.resconn2(attn, self.ffnn), attn


class Encoder(nn.Module):
    def __init__(self, number_of_layers, head, dimension, dropout):
        super().__init__()
        self.enclays = nn.ModuleList(
            [
                copy.deepcopy(EncoderLayer(dimension, head, dropout))
                for _ in range(number_of_layers)
            ]
        )
        self.norm = LayerNorm(dimension)

    def forward(self, input_vec, mask=None):
        for layer in self.enclays:
            input_vec, _ = layer(input_vec, mask)

        input_vec = self.norm(input_vec)
        return input_vec
