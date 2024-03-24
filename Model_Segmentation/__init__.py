"""
Pytorch model
Classification
"""
__version__ = "1.0.1"
__all__ = [
    "HRnet",  # 分割
    "UNet",
    "UNetPP"
]

from .hrnet import HRnet
from .unet import UNet
from .unetpp import UNetPP
