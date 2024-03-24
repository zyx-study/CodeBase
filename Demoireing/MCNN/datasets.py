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


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # 图像的预处理
        self.transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.CenterCrop(256),
            ]
        )
        self.files_source = sorted(glob.glob(root + "/source" + "/*.*"))
        self.files_target = sorted(glob.glob(root + "/target" + "/*.*"))

    def __getitem__(self, index):
        # 读取图片并预处理
        img_source = cv2.imread(self.files_source[index % len(self.files_source)])
        img_target = cv2.imread(self.files_target[index % len(self.files_target)])

        # cv2.imshow('', img_l)
        # cv2.waitKey()
        # exit()
        img_source = Image.fromarray(img_source)
        img_source = self.transform(img_source)

        img_target = Image.fromarray(img_target)
        img_target = self.transform(img_target)

        return {'img_source': img_source, 'img_target': img_target}

    def __len__(self):
        return len(self.files_source)
