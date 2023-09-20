"""
ViT (Vision Transformer) Implementation
This class can be used for classification as well as image segmentation

This module defines a Vision Transformer (ViT) model for image classification and reconstruction tasks. It consists of three main components: PatchEmbedding, Encoder, and Decoder.

PatchEmbedding: Converts the input image into a sequence of patches and adds positional embeddings.

ViT: Combines PatchEmbedding, an Encoder (transformer), and a Decoder to perform image classification and reconstruction.

Criterion: Defines a custom loss function for training the ViT model, which combines classification and reconstruction losses.

Usage:
1. Import the necessary modules and classes from this script.
2. Create an instance of the ViT model with the desired configuration.
3. Use the Criterion class for loss computation during training.

Example:
    import torch
    from torchsummary import summary

    # Define and initialize the ViT model
    device = torch.device("cpu")
    model = ViT(
        dimension=128,
        patch_size=4,
        nlayers=3,
        nheads=4,
        nclasses=10,
        use_cls_token=True  # Set to True if using a classification token
    )

    # Display model summary
    summary(model, input_size=(3, 32, 32), device=device)

Note:
- Make sure to customize the model configuration as needed for your specific task.
- The ViT model expects input images of shape (batch_size, 3, 32, 32) by default.
- You can adjust hyperparameters such as dimension, patch size, number of layers, and more.
- The Criterion class defines a custom loss function that combines classification and reconstruction losses with adjustable weights.
- Adjust the device (e.g., "cpu" or "cuda") according to your hardware.
"""


import math
import torch.nn as nn
from transformer import Encoder


class PatchEmbedding(nn.Module):
    def __init__(
        self, dimension, patch_size, npatches=4, dropout=0.1, use_cls_token=False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if use_cls_token:
            num_patches = npatches + 1
            self.cls_token = nn.Parameter(torch.randn(1, 1, dimension))
            # self.mask_token = nn.Parameter(torch.zeros(1, 1, dimension)) if use_mask_token else None
        else:
            num_patches = npatches
            self.cls_token = None

        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, dimension))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: [batch, 3, 32, 32]
        self.conv shape: [batch, dimension, 8, 8]
        return [batch, 64, dimension]
        """
        batch_size, num_channels, height, width = x.shape
        embeddings = self.conv(x).flatten(2).transpose(1, 2)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = embeddings + self.position_embeddings
        return self.dropout(embeddings)


class ViT(nn.Module):
    def __init__(
        self,
        dimension,
        patch_size,
        nlayers,
        nheads,
        nclasses,
        use_cls_token=False,
        dropout=0.1,
    ):
        super().__init__()
        self.patch = PatchEmbedding(
            dimension=dimension,
            patch_size=patch_size,
            dropout=dropout,
            use_cls_token=use_cls_token,
        )

        self.encoder = Encoder(
            number_of_layers=nlayers,
            head=nheads,
            dimension=dimension,
            dropout=dropout,
            hidden_dimension=dimension * 2,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=dimension,
                out_channels=patch_size**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(patch_size),
        )

        self.ffnn = nn.Sequential(
            nn.LayerNorm(normalized_shape=dimension), nn.Linear(dimension, nclasses)
        )

    def forward(self, x):
        """
        x: [batch, 3, 32, 32]
        x from self.encoder: [batch, seqlen, dimension]
        torch.mean(x, dim=-1): [batch, dimenion]
        """
        x = self.patch(x)
        x = self.encoder(x)

        if self.patch.cls_token is not None:
            classification = self.ffnn(x[:, 0])
            x = x[:, 1:]

        else:
            classification = self.ffnn(torch.mean(x, dim=1))

        # Reshape to (batch_size, num_channels, height, width)
        batch_size, sequence_length, num_channels = x.shape
        height = width = math.floor(sequence_length**0.5)
        x = x.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        x = self.decoder(x)
        return classification, x
    

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.reconstruct = nn.functional.l1_loss

        self.lambda_classification = 0.70
        self.lambda_reconstruct = 0.30

    def forward(self, class_pred, class_trg, image_pred, image_target):
        return self.lambda_classification * self.cross_entropy(
            class_pred, class_trg
        ) + self.lambda_reconstruct * self.reconstruct(
            image_target, image_pred, reduction="mean"
        )
    

if __name__ == "__main__":
    import torch
    from torchsummary import summary

    device = torch.device("mps")
    model = ViT(dimension=128, 
                kernel_size=4, 
                nlayers=3, 
                nheads=4, 
                nclasses=10)

    summary(model, input_size=(3, 32, 32), device="cpu")