from transformer.model import (
    MultiHeadedAttention,
    FeedForwardNet,
    ResidualConnection,
    LayerNorm,
)
import torch.nn as nn
import copy


class DecoderLayer(nn.Module):
    def __init__(self, dimension, heads=8, dropout=0.1):
        super().__init__()
        self.ffnn = FeedForwardNet(dimension, dropout=dropout)
        self.resconn = nn.ModuleList(
            [copy.deepcopy(ResidualConnection(dimension, dropout)) for _ in range(3)]
        )
        self.attn = nn.ModuleList(
            [
                copy.deepcopy(MultiHeadedAttention(dimension, heads, dropout))
                for _ in range(2)
            ]
        )

    def forward(self, input_vec, encoder_output, encmask, decmask):
        selfattn = self.resconn[0](input_vec, lambda x: self.attn[0](x, x, x, decmask))

        encdecattn = self.resconn[1](
            selfattn, lambda x: self.attn[1](x, encoder_output, encoder_output, encmask)
        )

        return self.resconn[2](encdecattn, self.ffnn)


class Decoder(nn.Module):
    def __init__(self, number_of_layers, head, dimension, dropout):
        super().__init__()
        self.declays = nn.ModuleList(
            [
                copy.deepcopy(DecoderLayer(dimension, head, dropout))
                for i in range(number_of_layers)
            ]
        )
        self.norm = LayerNorm(dimension)

    def forward(self, input_vec, encoder_output, encmask, decmask):

        for layer in self.declays:
            input_vec = layer(input_vec, encoder_output, encmask, decmask)
        return self.norm(input_vec)
