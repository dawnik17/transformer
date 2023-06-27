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


class MultiQueryAttention(nn.Module):
    def __init__(self, dimension, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dimension = dimension
        
        self.head_dim = self.dimension // self.heads
        self.qkv = nn.Linear(dimension, dimension + 2 * self.head_dim)
        self.outl = nn.Linear(dimension, dimension)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        MQA is only for Self-Attention
        x - [batch, seqlen, dimension]
        """
        assert self.dimension == x.size(-1)
        batch_size = x.size(0)
        
        x = self.qkv(x)
        query, key, value = torch.split(x, split_size_or_sections=[self.dimension, self.head_dim, self.head_dim], dim=-1)

        # query - [batch, seqlen, dimension]
        # key, value - [batch, seqlen, head_dim]
        attn = self.attention(query, key, value, mask, self.dropout)
        return self.outl(attn)


    def attention(self, query, key, value, mask=None, dropout=None):
        # query - [batch, seqlen, dimension]
        # key, value - [batch, seqlen, head_dim]
        
        batch_size = query.size(0)
        query_length = query.size(1)
        key_length = key.size(1)
        
        attn_shape = (batch_size, self.heads, query_length, key_length)
        
        # query from - [batch, seqlen, dimension] -> [batch, seqlen * heads, head_dim]
        query_shape = query.shape
        query = query.reshape(batch_size, self.heads * query_length, self.head_dim)
        
        # query - [batch, seqlen * heads, head_dim]
        # key - [batch, seqlen, head_dim]
        # q.k_t - [batch, seqlen * heads, seqlen]
        # (q.k_t).view(attn_shape) - [batch, heads, seqlen, seqlen]
        qk = torch.div(
            torch.matmul(query, key.transpose(-2, -1)), math.sqrt(self.head_dim)
        ).view(attn_shape)

        if mask is not None:
            mask = mask.unsqueeze(1)
            qk = qk.masked_fill(mask == 0, -1e9)

        qk = nn.Softmax(dim=-1)(qk)
        qk = self.dropout(qk) if dropout is not None else qk

        # qk from - [batch, heads, seqlen, seqlen] -> [batch, heads * seqlen, seqlen]
        # value - [batch, seqlen, head_dim]
        # torch.matmul(qk, value) -> [batch, heads * seqlen, head_dim]
        # torch.matmul(qk, value).view(query_shape) - [batch, seqlen, dimension]
        qk = qk.view(batch_size, self.heads * query_length, key_length)
        return torch.matmul(qk, value).view(query_shape) 
