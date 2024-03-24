import cv2
import sys
import os
import numpy as np
import torch
from models import *
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import random


class ModelGUI:
    def __init__(self):
        super(ModelGUI, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MoireCNN().to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(sys.path[0], 'saved_models', 'moire_best_MCNN.pth'), map_location=self.device))
        self.model.eval()
        # 'D:\\CodeBase\\Moire\\saved_models\\moire_best_MCNN.pth'
        # 图像的预处理
        self.transform = transforms.Compose(
            [
                # transforms.Resize((224, 224), InterpolationMode.BICUBIC),  # 缩放到224*224
                transforms.ToTensor(),
                transforms.CenterCrop(256),
            ]
        )

    def detect(self, image):
        h, w = image.shape[:2]  # 记录大小
        # x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = Image.fromarray(image)
        x = self.transform(x).unsqueeze(0)

        x = x.to(self.device)

        # 推理
        with torch.no_grad():
            outs = self.model(x).detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            outs = outs * 255

        outs = outs.astype(np.uint8)
        # outs = cv2.cvtColor(outs, cv2.COLOR_RGB2BGR)
        # outs = cv2.resize(outs, (w, h)) # 将大小改变回来

        return outs


if __name__ == "__main__":
    model = ModelGUI()
    file_path = "D:/CodeBase/Moire/image/train/source/image_test_part001_00000980_source.png"
    image = cv2.imread(file_path)

    output = model.detect(image)

    # 显示图像
    cv2.imshow("Output Image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
