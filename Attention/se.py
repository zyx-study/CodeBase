import torch
import torch.nn as nn


class SeBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SeBlock(in_channels=3, reduction=2).to(device)
    summary(model, (3, 64, 64))  # 输入张量的大小为 (batch_size, channels, height, width)
