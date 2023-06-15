import torch.nn as nn
import torch
import math


class MultiHeadedAttention(nn.Module):
    def __init__(self, dimension, heads, dropout=0.0):
        super().__init__()
        assert dimension % heads == 0

        self.heads = heads
        self.dimension = dimension
        self.wq = nn.Linear(dimension, dimension)
        self.wk = nn.Linear(dimension, dimension)
        self.wv = nn.Linear(dimension, dimension)
        self.out = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

        print(f"{heads} heads {dimension} dimension in MultiHeadedAttention")

    def forward(self, xq, xk, xv, mask=None):
        """
        xq, xk, xv shape - [Batch, SegLen, Dimension]
        """
        assert self.dimension == xq.size(-1)
        batch_size = query.size(0)

        # converting inputs to query, key and value
        query = self.wq(xq)
        key = self.wk(xk)
        value = self.wv(xv)

        """
        Query, Key Value = 
        
        [Batch, SegLen, Dimension]
                     |
        [Batch, SegLen, Heads, Dimension // Heads]
                     |
        [Batch, Heads, SegLen, Dimension // Heads]
        """
        query = query.view(
            batch_size, -1, self.heads, query.size(-1) // self.heads
        ).transpose(1, 2)
        key = key.view(
            batch_size, -1, self.heads, key.size(-1) // self.heads
        ).transpose(1, 2)

        value = value.view(
            batch_size, -1, self.heads, value.size(-1) // self.heads
        ).transpose(1, 2)

        # attention
        attn = self.attention(query, key, value, mask, self.dropout)

        # concatenate/merge all the heads back to form the final output
        # shape - [Batch, SeqLen, Dimension]
        concat = attn.transpose(1, 2).reshape(
            batch_size, -1, query.size(-1) * self.heads
        )

        # Final MLP Layer
        # shape - [Batch, SeqLen, Dimension]
        return self.out(concat)

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        Weight Matrix -

        Q.K{transpose} / SQRT(Dimension)
        shape - [Batch, Heads, SeqLen, SeqLen]
        """
        qk = torch.div(
            torch.matmul(query, key.transpose(-2, -1)), math.sqrt(query.size(-1))
        )

        # Masking
        # mask shape - [Batch, 1, SeqLen]
        if mask is not None:
            # mask shape - [Batch, 1, 1, SeqLen]
            # because QK is a 4 dimensional tensor
            mask = mask.unsqueeze(1)

            # convert weights to zero where the value is <PAD>
            # i.e. where mask is False or 0
            qk = qk.masked_fill(mask == 0, -1e9)

        # Softmax ansd dropout AFTER masking
        qk = nn.Softmax(dim=-1)(qk)
        qk = self.dropout(qk) if dropout is not None else qk

        # for each position, sum all values times their weight from all positions
        # (QKtranspose).value
        # shape - [Batch, Heads, SeqLen, Dimension // Heads]
        return torch.matmul(qk, value)
