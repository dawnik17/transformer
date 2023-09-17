import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocab_size, dimension):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)
        self.dimension = dimension

    def forward(self, input_vec):
        return self.embedding(input_vec) * math.sqrt(self.dimension)


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, dimension, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positional_enc = torch.zeros(max_seq_len, dimension)

        den = torch.pow(
            10000, torch.div(torch.arange(0, dimension / 2) * 2, float(dimension))
        )
        num = torch.arange(0, max_seq_len).unsqueeze(1)

        positional_enc[:, 0::2], positional_enc[:, 1::2] = (
            torch.sin(num / den),
            torch.cos(num / den),
        )
        positional_enc = positional_enc.unsqueeze(0)
        self.register_buffer("positional_enc", positional_enc)

    def forward(self, input_vec):
        seq_len = input_vec.size(1)
        return self.dropout(input_vec + Variable(self.positional_enc[:, :seq_len]))

# class FeedForwardNet(nn.Module):
#     def __init__(self, dimension, dff=2048, dropout=0.1):
#         super().__init__()
#         self.l = nn.Linear(dimension, dff)
#         self.out = nn.Linear(dff, dimension)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, input_vec):
#         return self.out(self.dropout(F.relu(self.l(input_vec))))

class FeedForwardNet(nn.Module):
    def __init__(self, dimension, hidden_dimension=1024, dropout=0.1, activation="gelu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, dimension),
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, dimension, delta=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        self.delta = delta

    def forward(self, input_vec):
        mean = torch.mean(input_vec, dim=-1, keepdim=True)
        std = torch.std(input_vec, dim=-1, keepdim=True) + self.delta
        return (self.gain / std) * (input_vec - mean) + self.bias


class ResidualConnection(nn.Module):
    def __init__(self, dimension, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_vec, sublayer):
        return input_vec + self.dropout(sublayer(self.norm(input_vec)))
