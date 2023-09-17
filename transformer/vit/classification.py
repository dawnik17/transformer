import torch.nn as nn
from transformer import Encoder


class PatchEmbedding(nn.Module):
    def __init__(self, dimension, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=dimension, 
                      kernel_size=kernel_size,
                      stride=kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=dimension,
                      out_channels=dimension,
                      kernel_size=1)
        )

    def forward(self, x):
        """
        x shape: [batch, 3, 32, 32]
        self.conv shape: [batch, dimension, 8, 8]
        """
        batch_size = x.shape[0]
        x = self.conv(x)

        dimension = x.shape[1]
        return x.permute(0, 2, 3, 1).view(batch_size, -1, dimension)

class ViTClassification(nn.Module):
    def __init__(self, dimension, kernel_size, nlayers, nheads, nclasses, dropout=0.1):
        super().__init__()
        self.patch = PatchEmbedding(dimension=dimension, kernel_size=kernel_size, dropout=dropout)
        self.encoder = Encoder(
            number_of_layers=nlayers,
            head=nheads,
            dimension=dimension,
            dropout=dropout
        )
        self.ffnn = nn.Linear(dimension, nclasses)

    def forward(self, x):
        """
        x: [batch, 3, 32, 32]
        x from self.encoder: [batch, seqlen, dimension]
        x[:, 0]: [batch, dimenion]
        """
        x = self.patch(x)
        x = self.encoder(x)
        return self.ffnn(x[:, 0])
    

if __name__ == "__main__":
    import torch
    from torchsummary import summary

    device = torch.device("mps")
    model = ViTClassification(dimension=128, 
                kernel_size=4, 
                nlayers=3, 
                nheads=4, 
                nclasses=10)

    summary(model, input_size=(3, 32, 32), device="cpu")