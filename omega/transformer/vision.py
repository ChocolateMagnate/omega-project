import torch
import torch.nn as nn
from torch import Tensor

import omega.transformer.hyperparameters as hp


class ImageEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels=3,  # General-purpose vision transformers work with RGB.
            out_channels=hp.HIDDEN_SIZE,
            kernel_size=hp.PATCH_SIZE,
            stride=hp.PATCH_SIZE
        )

    def forward(self, image: Tensor) -> Tensor:
        assert image.dim() == 4 and image.size(1) == 3, \
            f"Expected input shape (batch_size, 3, height, width), got {image.shape}"

        convoluted = self.convolution(image)    # [batch_size, hidden_size, height, width]
        # We flatten the height and width dimensions to concatenate the patches and get the sequence length.
        concatenated = convoluted.flatten(2)    # [batch_size, hidden_size, height * width]
        swapped = concatenated.transpose(1, 2)  # [batch_size, height * width, hidden_size]
        return swapped
