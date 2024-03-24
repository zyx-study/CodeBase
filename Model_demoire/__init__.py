"""
Pytorch model
Demoire
"""
__version__ = "1.0.1"
__all__ = [
    "MoireCNN",  # 分割
    "MoireMDDM",
]

from .MCNN import MoireCNN
from .MDDM import MoireMDDM
