import argparse
import os
import numpy as np
import math
import itertools
import sys

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from losses import *
from models import *
from datasets import *
from metrics import *
from scheduler import *

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

root = sys.path[0]  # 获取项目根目录
os.makedirs(os.path.join(root, "out_images"), exist_ok=True)
os.makedirs(os.path.join(root, "saved_models"), exist_ok=True)

img_align_celeba = os.path.join(root, 'image', 'train')  # 设置图片路径

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0)  # 开始训练的epoch 大于0将读取之前保存的模型继续训练
parser.add_argument("--n_epochs", type=int, default=10)  # 训练多少epoch
parser.add_argument("--dataset_name", type=str, default=img_align_celeba)  # 图像数据路径
parser.add_argument("--batch_size", type=int, default=4)  # batch_size大小 默认为4
parser.add_argument("--lr", type=float, default=5e-4)  # 学习率
parser.add_argument("--b1", type=float, default=0.5)  # AdamW优化器一阶矩估计的衰减率
parser.add_argument("--b2", type=float, default=0.999)  # AdamW优化器二阶矩估计的衰减率
parser.add_argument("--decay_epoch", type=int, default=20)  # 学习率衰减默认从20epoch开始
parser.add_argument("--n_cpu", type=int, default=0)  # 数据加载时的 CPU 进程数量
parser.add_argument("--hr_height", type=int, default=256)  # 输入图像的高
parser.add_argument("--hr_width", type=int, default=256)  # 输入图像的宽
parser.add_argument("--channels", type=int, default=3)  # 输入图像通道 默认为3 彩色图像
parser.add_argument("--sample_interval", type=int, default=200)  # 保存输出图像（效果图与指标曲线）的batch间隔
parser.add_argument("--checkpoint_interval", type=int, default=1)  # 保存模型参数的epoch间隔
parser.add_argument("--adv_loss", type=str, default='mse')  # 对抗损失函数 mse or bce。默认使用MSELoss，效果更好
opt = parser.parse_args()
# print(opt)

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# 初始化生成器与判别器
model = MoireCNN()

# 定义损失函数
criterion = torch.nn.MSELoss() if opt.adv_loss == 'mse' else torch.nn.BCELoss()

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

if opt.epoch != 0:
    # 加载权重继续训练
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(os.path.join(root, "saved_models", f'moire_{opt.epoch}.pth'), map_location=device))

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 学习率衰减器
sch = scheduler(sch='MultiStepLR').get_scheduler(optimizer)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset(opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  训练逻辑代码
# ----------
g_loss = []
psnr = []
msssims = []
ciede2000 = []
for epoch in range(opt.epoch, opt.n_epochs):

    for i, imgs in enumerate(dataloader):

        # 模型输入
        imgs_sr = Variable(imgs["img_source"].type(Tensor))
        imgs_tg = Variable(imgs["img_target"].type(Tensor))

        # ------------------
        #  训练模型
        # ------------------

        optimizer.zero_grad()

        # 从输入图像输入生成输出图像
        gen_tg = model(imgs_sr)

        # 感知损失
        loss = criterion(gen_tg, imgs_sr)

        loss.backward()
        optimizer.step()

        g_loss.append(loss.item())

        # ---------------------
        # 计算各项指标值
        # ---------------------

        # 计算psnr
        psnr.append(PSNR(gen_tg, imgs_tg))

        # --------------
        #  打印训练日志
        # --------------

        # sys.stdout.write(
        #     "[Epoch:%d/%d] [Batch:%d/%d] [D loss:%f] [G loss:%f]\n"
        #     % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        # )
        print(
            f'[Epoch:{epoch}/{opt.n_epochs}][Batch:{i}/{len(dataloader)}][loss:{loss.item():.3f}][PSNR:{psnr[-1]:.3f}]')
        batches_done = epoch * len(dataloader) + i  # 目前的bactch数量
        if batches_done % opt.sample_interval == 0:
            # 保存source图像、target图像与修复图对比结果
            imgs_sr = make_grid(imgs_sr, nrow=1, normalize=True)
            gen_tg = make_grid(gen_tg, nrow=1, normalize=True)
            imgs_tg = make_grid(imgs_tg, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_sr, gen_tg, imgs_tg), -1)
            save_image(img_grid, os.path.join(root, "out_images", f"{batches_done}.png"), nrow=1, normalize=False)

            # 绘制指标在训练过程中的变化曲线图像
            # ------------------------------------------------------------------------------------------------------------------------
            x = list(range(len(g_loss)))
            # 设置画布大小
            fig = plt.figure(figsize=(6, 4))
            # 设置字体
            font = plt.matplotlib.font_manager.FontProperties()
            font.set_family('serif')
            font.set_style('italic')
            # font.set_size(12)
            # 绘制loss曲线
            plt.plot(x, g_loss, c='g', label='G', linestyle='-', linewidth=2, alpha=0.8)
            # 设置x轴和y轴标签
            plt.xlabel('Batch', fontsize=10,
                       fontweight='bold')  # fontsize：字体大小， fontweight:设置字体粗细，'normal'（正常）、'bold'（粗体）
            plt.ylabel('LOSS Value', fontsize=10, fontweight='bold')
            # 设置图例
            plt.legend(loc='best', fontsize=8)
            # 设置标题
            plt.title('TRAIN LOSS', fontsize=12, fontweight='bold')
            # 设置坐标轴范围
            # plt.xlim(0, len(dataloader))
            plt.ylim(0, 2)
            # 设置网格线
            '''
            在matplotlib中，plt.grid()函数用于绘制网格线。它有以下几个参数：
            True或False：可选参数，表示是否绘制网格线。默认值为False。如果设置为True，则绘制网格线；如果设置为False，则不绘制网格线。
            which：可选参数，表示绘制哪种类型的网格线。可选值有'both'（绘制水平和垂直网格线）、'x'（绘制水平网格线）、'y'（绘制垂直网格线）。默认值为'both'。如果设置为'x'或'y'，则只绘制相应轴的网格线。
            axis：可选参数，表示绘制网格线的轴。可选值有'both'（绘制所有轴的网格线）、'0'（绘制x轴的网格线）、'1'（绘制y轴的网格线）。默认值为'both'。如果设置为'0'或'1'，则只绘制相应轴的网格线。
            color：可选参数，表示网格线的颜色。可以是一个颜色字符串，如'black'、'gray'等，或者是一个颜色元组，如(0.5, 0.5, 0.5)。默认值为'black'。
            linestyle：可选参数，表示网格线的样式。可以是一个字符串，如'-'（实线）、'--'（虚线）等。默认值为'-'。
            linewidth：可选参数，表示网格线的宽度。可以是一个正数，单位为像素，如2。默认值为1。
            alpha：可选参数，表示网格线的透明度。可以是一个介于0（完全透明）和1（完全不透明）之间的浮点数。默认值为1。
            '''
            plt.grid(True, which='both', axis='both', color='k', linestyle='--', linewidth=1, alpha=0.5)
            # 自适应布局
            plt.tight_layout()
            plt.savefig(os.path.join(root, 'LOSS.png'))  # 保存到项目根目录
            plt.clf()  # 清空画板
            # ------------------------------------------------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------------------------------------------------
            # PSNR
            # 设置画布大小
            fig = plt.figure(figsize=(6, 4))
            # 设置字体
            font = plt.matplotlib.font_manager.FontProperties()
            font.set_family('serif')
            font.set_style('italic')
            # 绘制峰值信噪比曲线
            plt.plot(x, psnr, c='g', label='G', linestyle='-', linewidth=2, alpha=0.8)
            # 设置x轴和y轴标签
            plt.xlabel('Batch', fontsize=10, fontweight='bold')
            plt.ylabel('PSNR Value', fontsize=10, fontweight='bold')
            # 设置图例
            plt.legend(loc='best', fontsize=8)
            # 设置标题
            plt.title('TRAIN PSNR', fontsize=12, fontweight='bold')
            # 设置坐标轴范围
            # plt.xlim(0, len(dataloader))
            plt.ylim(0, 100)
            plt.grid(True, which='both', axis='both', color='k', linestyle='--', linewidth=1, alpha=0.5)
            # 自适应布局
            plt.tight_layout()
            plt.savefig(os.path.join(root, 'PSNR.png'))
            plt.clf()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # 保存模型
        torch.save(model.state_dict(), os.path.join(root, "saved_models", f"Moire_{epoch}.pth"))

    if epoch + 1 > opt.decay_epoch:
        sch.step()  # 更新学习率
