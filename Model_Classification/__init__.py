"""
Pytorch model
Classification
"""
__version__ = "1.0.1"
__all__ = [
    "AlexNet",
    "ResNet18",
    "MobileNetV3_Small", "MobileNetV3_Large",
    "SwinTransformer",
    "EfficientNet",
    "DenseNet",  # 内存要求比较大
    "LeNet",
    "ZFNet",
    "GoogLeNet",
    "Inception_ResNetv2",
    "GoogLeNetV3",
    "Inceptionv4",
    "xception",
    "ShuffleNet",
    "ShuffleNetV2",
    "regnetx_002", "regnetx_004", "regnetx_006", "regnetx_008", "regnetx_016", "regnetx_032", "regnetx_040",
    "regnetx_064", "regnetx_080", "regnetx_120", "regnetx_160", "regnetx_320",
]
from .alexnet import AlexNet
from .vgg16 import VGG16
from .resnet18 import ResNet18
from .mobilenet import MobileNetV3_Small, MobileNetV3_Large
from .swintransformer import SwinTransformer
from .efficientnet import EfficientNet
from .densenet import DenseNet
from .lenet import LeNet
from .zfnet import ZFNet
from .inceptionv1 import GoogLeNet
from .inceptionv2 import Inception_ResNetv2
from .inceptionv3 import GoogLeNetV3
from .inceptionv4 import Inceptionv4
from .xception import xception
from .shufflenet import ShuffleNet
from .shufflenetv2 import ShuffleNetV2
from .regnet import regnetx_002, regnetx_004, regnetx_006, regnetx_008, regnetx_016, regnetx_032, regnetx_040, \
    regnetx_064, regnetx_080, regnetx_120, regnetx_160, regnetx_320
