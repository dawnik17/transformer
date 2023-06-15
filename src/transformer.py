import torch.nn as nn
import torch.nn.functional as F
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
        encout, _ = self.encoder(enc_input_vec, encmask)
        decout = self.decoder(dec_input_vec, encout, encmask, decmask)

        return F.log_softmax(self.ffnn(decout), dim=-1)
