import torch.nn as nn
import torch
import torch.nn.functional as F


class SequenceGenerationLoss(nn.Module):
    def __init__(self, vocab_size, pad_index, alpha):
        super().__init__()
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.pad_index = pad_index

    def forward(self, prediction, target):
        """
        prediction ->

            [batch, seqlen-1, vocab_size]
                          |
            [batch * (seqlen-1), vocab_size]


        target ->

            [batch, seqlen-1]
                    |
            [batch * (seqlen - 1)]
        """
        prediction = prediction.contiguous().view(-1, prediction.size(-1))
        target = target.contiguous().view(-1)

        """
        one_hot_target -> [batch * (seqlen - 1), vocab_size]
        
        we then smoothen the target/label by dispersing the probabilities a bit.
        [0, 0, 1, 0, 0] becomes [0.33, 0.33, 0.93, 0.33, 0.33]
        this makes the model less over confident
        
        we the make value at pad_idx as 0, because if the token is padding, 
        there will be 1 at the pad_idx position in the target
        we make that 0, to remove it's input in calculating the loss
        we do this because padding holds no value to us, so wee do not want the model to optimise that
        
        to accomplish the above point completely,
        we also make all the probabilities of the pad token to 0
        
        so finally the probabilites of padding idx for non pad target token is made 0.
        also, for padding tokens in target, all probabilities of vocab_size tokens is made 0.
        
        finally -
        
        the target which initially was,
        
        tensor([2, 1, 0])
        
        
        now looks like, (is stored in one_hot_target)
        
        tensor([[0.0000, 0.0333, 0.9333, 0.0333, 0.0333],
                [0.0000, 0.9333, 0.0333, 0.0333, 0.0333],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
        
        """
        one_hot_target = torch.nn.functional.one_hot(
            target, num_classes=prediction.size(-1)
        )

        one_hot_target = (one_hot_target * (1 - self.alpha)) + (
            self.alpha / (self.vocab_size - 2)
        )

        one_hot_target[:, self.pad_index] = 0
        one_hot_target.masked_fill_((target == self.pad_index).unsqueeze(1), 0)

        return F.kl_div(prediction, one_hot_target, reduction="sum")


if __name__ == "__main__":
    """
    batch size below = 1
    """
    # [batch, seqlen-1, vocab_size]
    predict = torch.FloatTensor(
        [[[0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0]]]
    )
    predict = predict.log()

    # [batch, seqlen-1]
    target = torch.LongTensor([[2, 1, 0]])

    ls = SequenceGenerationLoss(vocab_size=5, pad_index=0, alpha=0.1)
    ls(prediction=predict, target=target)
