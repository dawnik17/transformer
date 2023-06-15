import torch
from torch.autograd import Variable


class Mask:
    def __init__(self, src, trg=None, device="cpu", pad=0):
        """
        * <PAD> Mask for Encoder
        * src shape - [Batch, EncSeqLen]
        * src_mask shape - [Batch, 1, EncSeqLen]
        * True for values, False for <PAD>
        """
        self.src_mask = (src != pad).unsqueeze(2)

        if trg is not None:
            """
            <PAD> and Future Mask for Decoder
            trg shape - [Batch, DecSeqLen]
            trg_pad_mask shape - [Batch, 1, DecSeqLen-1]
            future_mask shape - [Batch, DecSeqLen-1, DecSeqLen-1]
            trg_mask shape - [Batch, DecSeqLen-1, DecSeqLen-1]
            """

            self.trg = trg[:, :-1]

            # <PAD> Mask for decoder
            trg_pad_mask = (self.trg != pad).unsqueeze(-2)

            # Future Masking
            dimension = self.trg.size(-1)
            future_mask = torch.tril(torch.ones(1, dimension, dimension, device=device))
            future_mask = Variable(future_mask > 0)

            if self.trg.is_cuda:
                future_mask.cuda()

            # Final Decoder Mask
            # "AND" condition on <PAD> Mask and Future Mask
            self.trg_mask = trg_pad_mask & future_mask
