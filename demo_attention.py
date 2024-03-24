from Attention import *
from torchsummary import summary
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GamBlock(in_channels=3, out_channels=3, reduction=2).to(device)
summary(model, (3, 64, 64))  # 输入张量的大小为 (batch_size, channels, height, width)
