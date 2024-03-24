import torch
import torch.nn as nn


class CaBlock(nn.Module):
    def __init__(self, in_channels, reduction):
        super(CaBlock, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction, kernel_size=1,
                                  stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channels // reduction)

        self.F_h = nn.Conv2d(in_channels=in_channels // reduction, out_channels=in_channels, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=in_channels // reduction, out_channels=in_channels, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # x batch_size,c,h,w
        _, _, h, w = x.size()

        # x batch_size,c,h, w => x batch_size,c,h,1 => x batch_size,c,1,h(通过转置来为下一步做准备)
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        # x batch_size,c,h, w  => x batch_size,c,1,w
        x_w = torch.mean(x, dim=2, keepdim=True)

        # x batch_size,c,h,1 cat batch_size,c,1,w => batch_size,c,1,w+h
        # batch_size, c, 1, w + h => batch_size, c/r, 1, w + h
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        # batch_size, c/r, 1, w + h => batch_size, c/r, 1,h and  batch_size, c/r, 1,w (再次把长 宽 分开)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        # batch_size, c / r, 1, h => # batch_size, c / r, h,1 => batch_size,c,h,1
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # batch_size, c / r, 1, w => batch_size, c , 1, w
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CaBlock(in_channels=3, reduction=2).to(device)
    summary(model, (3, 64, 64))  # 输入张量的大小为 (batch_size, channels, height, width)
