from io import BytesIO

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from PIL import Image

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


class ImageTokenizer:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        # ImageNet normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, image_bytes: bytes) -> Tensor:
        image = Image.open(BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        ratio = max(self.image_size / image.width, self.image_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        left = (image.width - self.image_size) // 2
        top = (image.height - self.image_size) // 2
        image = image.crop((left, top, left + self.image_size, top + self.image_size))
        
        tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1)
        tensor = tensor / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0)  # [1, 3, H, W]