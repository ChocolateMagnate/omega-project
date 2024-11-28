from typing import TypeAlias, Union

from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau

# We use these aliases to convey the intention behind the shape of tensor.
Scalar: TypeAlias = Tensor
Vector: TypeAlias = Tensor
Matrix: TypeAlias = Tensor

LRScheduler: TypeAlias = Union[CosineAnnealingLR, LinearLR, ReduceLROnPlateau]
