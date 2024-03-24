import os
import cv2
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def random_crop(moire, clean, crop_size, im_size=1024):
    if crop_size == im_size:
        return moire, clean
    else:
        rand_num_x = np.random.randint(im_size - crop_size - 1)
        rand_num_y = np.random.randint(im_size - crop_size - 1)
        moire = np.array(moire)
        clean = np.array(clean)
        nm = moire[rand_num_x:rand_num_x + crop_size, rand_num_y:rand_num_y + crop_size, :]
        nc = clean[rand_num_x:rand_num_x + crop_size, rand_num_y:rand_num_y + crop_size, :]
        nm = Image.fromarray(nm)
        nc = Image.fromarray(nc)
        return nm, nc


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        super(ImageDataset, self).__init__()
        self.HR_list = sorted(glob.glob(root + "/target" + "/*.*"))
        self.LR_list = sorted(glob.glob(root + "/source" + "/*.*"))
        # print(self.HR_list[:10])
        # print(self.LR_list[:10])
        self.transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        im_h = Image.open(self.HR_list[index % len(self.HR_list)])
        im_l = Image.open(self.LR_list[index % len(self.LR_list)])
        # im_l = Image.open(os.path.join(self.LR,self.HR_list[index][:-10]+'source.png'))
        im_l, im_h = random_crop(im_l, im_h, crop_size=1024, im_size=1024)
        HR = transforms.ToTensor()(im_h)
        LR = transforms.ToTensor()(im_l)

        # # 调整图像大小为1024x1024像素
        # im_h = im_h.resize((1024, 1024))
        # im_l = im_l.resize((1024, 1024))
        # HR = self.transform(im_h)
        # LR = self.transform(im_l)
        return {'img_source': LR, 'img_target': HR}

    def __len__(self):
        len_h = len(self.HR_list)
        len_l = len(self.LR_list)
        if len_h >= len_l:
            len_file = len_l
        else:
            len_file = len_h
        return len_file
