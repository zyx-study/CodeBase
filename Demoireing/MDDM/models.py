# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F


class upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.shuffler(self.conv(x)))


# Mixed Link Block architecture
class MLB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(MLB, self).__init__()
        # delete

    def forward(self, x):
        return x


class CARB(nn.Module):
    # channel attention residual block
    def __init__(self, nChannels):
        super(CARB, self).__init__()
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.ca1 = FRM(nChannels, 16)
        self.ca2 = FRM(nChannels, 16)

    def forward(self, x):
        out = self.conv2(self.relu1(self.conv1(x)))
        b1 = self.ca1(out) + x

        out = self.relu2(self.conv3(b1))
        b2 = self.ca2(self.conv4(out)) + b1
        return b2


## Channel Attention (CA) Layer
class FRM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FRM, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## non_local module
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']
        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None
        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size, C, H, W = x.shape
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _dot_product(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _concatenation(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8, sub_sample=False, bn_layer=True):
        super(Nonlocal_CA, self).__init__()
        # nonlocal module
        self.non_local = (
            NONLocalBlock2D(in_channels=in_feat, inter_channels=inter_feat, sub_sample=sub_sample, bn_layer=bn_layer))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## divide feature map into 4 part
        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)
        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]
        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd
        return nonlocal_feat


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        self.features = nn.Sequential(

            # input is (1) x 128 x 128
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 32 x 32
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 16 x 16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 16 x 16
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.features(input)
        # print(out.shape)
        # state size. (512) x 8 x 8
        out = out.view(out.size(0), -1)

        # state size. (512 x 8 x 8)
        # print(out.shape)
        out = self.fc1(out)

        # state size. (1024)
        out = self.LeakyReLU(out)

        out = self.fc2(out)
        # state size. (1)

        out = out.mean(0)

        # out = self.sigmoid(out)
        return out.view(1)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def din(content_feat, encode_feat, eps=None):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    encode_mean, encode_std = calc_mean_std(encode_feat)
    if eps == None:
        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
    else:
        normalized_feat = (content_feat - content_mean.expand(
            size)) / (content_std.expand(size) + eps)
    return normalized_feat * encode_std.expand(size) + encode_mean.expand(size)


class Down2(nn.Module):
    def __init__(self, c_in, c_out):
        super(Down2, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.down = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        out = self.down(out)
        LR_2x = self.convt_R1(out)
        return LR_2x


class Branch1(nn.Module):
    def __init__(self):
        super(Branch1, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        conv1 = self.conv_input2(out)
        return conv1


class Branch2(nn.Module):
    def __init__(self):
        super(Branch2, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        # style encode
        self.s_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        # -------------
        self.u1 = upsample_block(64, 256)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)
        s1 = self.s_conv1(out)
        convt_F11 = din(convt_F11, s1)
        convt_F12 = self.convt_F12(convt_F11)
        s2 = self.s_conv2(s1)
        convt_F12 = din(convt_F12, s2)
        convt_F13 = self.convt_F13(convt_F12)
        s3 = self.s_conv3(s2)
        convt_F13 = din(convt_F13, s3)
        combine = out + convt_F13
        up = self.u1(combine)
        clean = self.convt_shape1(up)
        return clean


class Branch3(nn.Module):
    def __init__(self):
        super(Branch3, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        self.s_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.u1 = upsample_block(64, 256)
        self.u2 = upsample_block(64, 256)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64 // 8, reduction=8, sub_sample=False, bn_layer=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)
        s1 = self.s_conv1(out)
        convt_F11 = din(convt_F11, s1)
        convt_F12 = self.convt_F12(convt_F11)
        s2 = self.s_conv2(s1)
        convt_F12 = din(convt_F12, s2)
        convt_F13 = self.convt_F13(convt_F12)
        s3 = self.s_conv3(s2)
        convt_F13 = din(convt_F13, s3)
        # convt_F14 = self.convt_F14(convt_F13)
        # s4 =self.s_conv4(s3)
        # convt_F14 = din(convt_F14,s4)
        # convt_F14 = self.non_local(convt_F14)
        combine = out + convt_F13
        up = self.u1(combine)
        up = self.u2(up)
        clean = self.convt_shape1(up)

        return clean


class Branch4(nn.Module):
    def __init__(self):
        super(Branch4, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        self.convt_F15 = CARB(64)
        self.convt_F16 = CARB(64)
        self.s_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.u1 = upsample_block(64, 256)
        self.u2 = upsample_block(64, 256)
        self.u3 = upsample_block(64, 256)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64 // 8, reduction=8, sub_sample=False, bn_layer=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)
        s1 = self.s_conv1(out)
        convt_F11 = din(convt_F11, s1)
        convt_F12 = self.convt_F12(convt_F11)
        s2 = self.s_conv2(s1)
        convt_F12 = din(convt_F12, s2)
        convt_F13 = self.convt_F13(convt_F12)
        s3 = self.s_conv3(s2)
        convt_F13 = din(convt_F13, s3)
        convt_F14 = self.convt_F14(convt_F13)
        s4 = self.s_conv4(s3)
        convt_F14 = din(convt_F14, s4)
        convt_F15 = self.convt_F15(convt_F14)
        s5 = self.s_conv5(s4)
        convt_F15 = din(convt_F15, s5)
        # convt_F16 = self.convt_F16(convt_F15)
        # s6 =self.s_conv6(s5)
        # convt_F16 = din(convt_F16,s6)
        # convt_F16 = self.non_local(convt_F16)
        combine = out + convt_F15
        up = self.u1(combine)
        up = self.u2(up)
        up = self.u3(up)
        clean = self.convt_shape1(up)

        return clean


class Branch5(nn.Module):
    def __init__(self):
        super(Branch5, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        self.convt_F15 = CARB(64)
        self.convt_F16 = CARB(64)
        self.convt_F17 = CARB(64)
        self.convt_F18 = CARB(64)
        self.s_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.u1 = upsample_block(64, 256)
        self.u2 = upsample_block(64, 256)
        self.u3 = upsample_block(64, 256)
        self.u4 = upsample_block(64, 256)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64 // 8, reduction=8, sub_sample=False, bn_layer=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)
        s1 = self.s_conv1(out)
        convt_F11 = din(convt_F11, s1)
        convt_F12 = self.convt_F12(convt_F11)
        s2 = self.s_conv2(s1)
        convt_F12 = din(convt_F12, s2)
        convt_F13 = self.convt_F13(convt_F12)
        s3 = self.s_conv3(s2)
        convt_F13 = din(convt_F13, s3)
        convt_F14 = self.convt_F14(convt_F13)
        s4 = self.s_conv4(s3)
        convt_F14 = din(convt_F14, s4)
        convt_F15 = self.convt_F15(convt_F14)
        s5 = self.s_conv5(s4)
        convt_F15 = din(convt_F15, s5)
        convt_F16 = self.convt_F16(convt_F15)
        s6 = self.s_conv6(s5)
        convt_F16 = din(convt_F16, s6)
        convt_F17 = self.convt_F17(convt_F16)
        s7 = self.s_conv7(s6)
        convt_F17 = din(convt_F17, s7)
        convt_F18 = self.convt_F18(convt_F17)
        s8 = self.s_conv8(s7)
        convt_F18 = din(convt_F18, s8)
        convt_F18 = self.non_local(convt_F18)
        combine = out + convt_F18
        up = self.u1(combine)
        up = self.u2(up)
        up = self.u3(up)
        up = self.u4(up)
        clean = self.convt_shape1(up)

        return clean


class Branch6(nn.Module):
    def __init__(self):
        super(Branch6, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_input2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.convt_F11 = CARB(64)
        self.convt_F12 = CARB(64)
        self.convt_F13 = CARB(64)
        self.convt_F14 = CARB(64)
        self.convt_F15 = CARB(64)
        self.convt_F16 = CARB(64)
        self.convt_F17 = CARB(64)
        self.convt_F18 = CARB(64)
        self.s_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.s_conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.s_conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.u1 = upsample_block(64, 256)
        self.u2 = upsample_block(64, 256)
        self.u3 = upsample_block(64, 256)
        self.u4 = upsample_block(64, 256)
        self.u5 = upsample_block(64, 256)
        self.convt_shape1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.non_local = Nonlocal_CA(in_feat=64, inter_feat=64 // 8, reduction=8, sub_sample=False, bn_layer=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        convt_F11 = self.convt_F11(out)
        s1 = self.s_conv1(out)
        convt_F11 = din(convt_F11, s1)
        convt_F12 = self.convt_F12(convt_F11)
        s2 = self.s_conv2(s1)
        convt_F12 = din(convt_F12, s2)
        convt_F13 = self.convt_F13(convt_F12)
        s3 = self.s_conv3(s2)
        convt_F13 = din(convt_F13, s3)
        convt_F14 = self.convt_F14(convt_F13)
        s4 = self.s_conv4(s3)
        convt_F14 = din(convt_F14, s4)
        convt_F15 = self.convt_F15(convt_F14)
        s5 = self.s_conv5(s4)
        convt_F15 = din(convt_F15, s5)
        convt_F16 = self.convt_F16(convt_F15)
        s6 = self.s_conv6(s5)
        convt_F16 = din(convt_F16, s6)
        convt_F17 = self.convt_F17(convt_F16)
        s7 = self.s_conv7(s6)
        convt_F17 = din(convt_F17, s7)
        convt_F18 = self.convt_F18(convt_F17)
        s8 = self.s_conv8(s7)
        convt_F18 = din(convt_F18, s8)
        convt_F18 = self.non_local(convt_F18)
        combine = out + convt_F18
        up = self.u1(combine)
        up = self.u2(up)
        up = self.u3(up)
        up = self.u4(up)
        up = self.u5(up)
        clean = self.convt_shape1(up)

        return clean


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.down2_1 = Down2(3, 64)
        self.down2_2 = Down2(64, 64)
        self.down2_3 = Down2(64, 64)
        self.down2_4 = Down2(64, 64)
        self.down2_5 = Down2(64, 64)
        # Branches
        self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.branch3 = Branch3()
        self.branch4 = Branch4()
        self.branch5 = Branch5()
        self.branch6 = Branch6()
        # scale
        self.scale1 = ScaleLayer()
        self.scale2 = ScaleLayer()
        self.scale3 = ScaleLayer()
        self.scale4 = ScaleLayer()
        self.scale5 = ScaleLayer()
        self.scale6 = ScaleLayer()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # out = self.relu(self.conv_input(x))
        # print('1')
        b1 = self.branch1(x)
        b1 = self.scale1(b1)
        # ---------
        # print('2')
        feat_down2 = self.down2_1(x)
        b2 = self.branch2(feat_down2)
        b2 = self.scale2(b2)
        # ---------
        # print('3')
        feat_down3 = self.down2_2(feat_down2)
        b3 = self.branch3(feat_down3)
        b3 = self.scale3(b3)
        # ---------\
        # print('4')
        feat_down4 = self.down2_3(feat_down3)
        b4 = self.branch4(feat_down4)
        b4 = self.scale4(b4)
        # ---------
        # print('5')
        feat_down5 = self.down2_4(feat_down4)
        b5 = self.branch5(feat_down5)
        b5 = self.scale5(b5)
        # ---------
        feat_down6 = self.down2_5(feat_down5)
        b6 = self.branch6(feat_down6)
        b6 = self.scale6(b6)
        # clean = x + b1 + b2 + b3 + b4
        clean = b1 + b2 + b3 + b4 + b5 + b6
        # clean = self.convt_shape1(combine)

        return clean


class ScaleLayer(nn.Module):

    def __init__(self, init_value=1.0):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        #    print(self.scale)
        return x * self.scale


if __name__ == '__main__':
    """
    MDDM必须是1024*1024，不然会出现Nan的情况
    且此模型对显存要求比较高
    """
    from torchkeras import summary
    import torch

    device = torch.device("cpu")
    model = Net().to(device)
    summary(model, input_shape=(3, 1024, 1024))
