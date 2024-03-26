"""
LOSS Function
"""

__version__ = '1.0.1'
__all__ = [
    "contextual_loss", "ContextualLoss",  # CX 对显存要求特别大
    "_ContextualBilateralLoss",  # Cobi 对显存要求特别大,VGG模式更大
    "PerceptualLoss"
]

from .CX import contextual_loss, ContextualLoss
from .Cobi import _ContextualBilateralLoss
from .Perceptual import PerceptualLoss
