import torch.nn as nn
import torch.nn.functional as F
from src.transformer.utils import Embedding, PositionalEncoding
from src.transformer import Encoder, Decoder


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        envocab_size,
        devocab_size,
        kernel_sizes,
        enc_max_seq_len,
        dec_max_seq_len,
        head,
        number_of_layers,
        dimension,
        dropout,
    ):
        super().__init__()
        self.enc_emb = nn.Sequential(
            Embedding(envocab_size, dimension=dimension),
            PositionalEncoding(enc_max_seq_len, dimension, dropout)
        )

        self.dec_emb = nn.Sequential(
            Embedding(devocab_size, dimension=dimension),
            PositionalEncoding(dec_max_seq_len, dimension, dropout)
        )

        self.encoder = Encoder(
            envocab_size,
            number_of_layers,
            head,
            enc_max_seq_len,
            dimension,
            dropout,
            kernel_sizes,
        )
        self.decoder = Decoder(
            devocab_size, number_of_layers, head, dec_max_seq_len, dimension, dropout
        )
        self.ffnn = nn.Linear(dimension, devocab_size)

    def forward(self, enc_input_vec, dec_input_vec, encmask, decmask):
        """
        enc_input_vec - [batch, enc_seq_len]
        dec_input_vec - [batch, dec_seq_len]
        encmask - [batch, 1, enc_seq_len]
        decmask - [batch, dec_seq_len, dec_deq_len] 

        Args:
            enc_input_vec (_type_): _description_
            dec_input_vec (_type_): _description_
            encmask (_type_): _description_
            decmask (_type_): _description_

        Returns:
            _type_: _description_
        """
        enc_input_vec = self.enc_emb(enc_input_vec)
        dec_input_vec = self.dec_emb(dec_input_vec)

        encout = self.encoder(enc_input_vec, encmask)
        decout = self.decoder(dec_input_vec, encout, encmask, decmask)

        return F.log_softmax(self.ffnn(decout), dim=-1) 
