import torch.nn as nn
import torch


class GamBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(GamBlock, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / reduction), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / reduction), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / reduction)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / reduction), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GamBlock(in_channels=3, out_channels=3, reduction=2).to(device)
    summary(model, (3, 64, 64))  # 输入张量的大小为 (batch_size, channels, height, width)
