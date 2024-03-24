import torch
import torch.nn as nn


class NamBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(NamBlock, self).__init__()
        self.channels = in_channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        return x


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NamBlock(in_channels=3).to(device)
    summary(model, (3, 64, 64))  # 输入张量的大小为 (batch_size, channels, height, width)
