from datasets import *
from metrics import *
from models import *
from losses import *
import argparse
import numpy as np
import os
import sys

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

root = sys.path[0]
img_align_celeba = os.path.join(root, 'image', 'test')  # 设置图片路径

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_name", type=str, default="moire_best_MCNN.pth")  # 模型权重名称
parser.add_argument("--dataset_name", type=str, default=img_align_celeba)  # 图像数据路径
parser.add_argument("--batch_size", type=int, default=4)  # batch_size大小 默认为4
parser.add_argument("--n_cpu", type=int, default=0)  # 数据加载时的 CPU 进程数量
parser.add_argument("--hr_height", type=int, default=256)  # 输入图像的高
parser.add_argument("--hr_width", type=int, default=256)  # 输入图像的宽
opt = parser.parse_args()
hr_shape = (opt.hr_height, opt.hr_width)

# 加载训练好的模型
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

model = Net().to(device)
model.load_state_dict(torch.load(os.path.join(root, 'saved_models', opt.checkpoint_name), map_location=device))
model.eval()

# 加载测试集
dataloader = DataLoader(
    ImageDataset(img_align_celeba, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
)

# 迭代测试集，评估模型
psnr = []
with torch.no_grad():  # 禁止梯度计算
    for n, imgs in enumerate(dataloader):
        # 模型输入
        imgs_sr = Variable(imgs["img_source"].type(Tensor))
        imgs_tg = Variable(imgs["img_target"].type(Tensor))

        # 从输入图像输入生成输出图像
        gen_tg = model(imgs_sr)

        # 峰值信噪比
        psnr.append(PSNR(gen_tg, imgs_tg))
        print(f'第{n + 1}个batch: psnr:{psnr[-1]:.3f}')

print(
    f'测试结束！\n三个指标的全局平均值为：psnr={(sum(psnr) / len(psnr)):.3f}')
