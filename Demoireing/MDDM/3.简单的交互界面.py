from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
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


class SwinUNet:
    def __init__(self):
        super(SwinUNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MoireCNN().to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(sys.path[0], 'saved_models', 'Moire_2.pth'), map_location=self.device))
        self.model.eval()
        self.model.eval()

        # 图像的预处理
        self.transform = transforms.Compose(
            [
                # transforms.Resize((224, 224), InterpolationMode.BICUBIC),  # 缩放到224*224
                transforms.CenterCrop(256),
                transforms.ToTensor(),
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

        outs = outs.astype(np.uint8)  # 这边的颜色转化要注意一下
        # outs = cv2.cvtColor(outs, cv2.COLOR_RGB2BGR)
        # outs = cv2.resize(outs, (w, h)) # 将大小改变回来

        return outs


class ImageProcessingWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("图像还原")
        self.setGeometry(100, 100, 1000, 600)

        self.initUI()
        self.model = SwinUNet()
        self.root = sys.path[0]

        # 创建一个保存结果的文件夹
        os.makedirs('result', exist_ok=True)

    def initUI(self):
        self.selectButton = QPushButton("选择图像", self)
        self.selectButton.setGeometry(20, 20, 100, 30)
        self.selectButton.clicked.connect(self.selectImage)

        self.processButton = QPushButton("修复", self)
        self.processButton.setGeometry(20, 70, 100, 30)
        self.processButton.clicked.connect(self.processImage)

        self.originalLabel = QLabel(self)
        self.originalLabel.setGeometry(150, 20, 400, 400)
        self.originalLabel.setAlignment(Qt.AlignCenter)

        self.resultLabel = QLabel(self)
        self.resultLabel.setGeometry(550, 20, 400, 400)
        self.resultLabel.setAlignment(Qt.AlignCenter)

    def selectImage(self):
        self.filePath, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                                       "Image Files (*.jpg *.png *.bmp *.JPG *.JPEG *.PNG *.jpeg *.BMP)")
        if self.filePath:
            pixmap = QPixmap(self.filePath)
            self.originalLabel.setPixmap(pixmap.scaled(224, 224, Qt.KeepAspectRatio))

    def processImage(self):
        if self.filePath:
            # 读取选择的图片，然后曝光
            cvImage = cv2.imread(self.filePath)

            resultImage = self.model.detect(cvImage)
            save_path = os.path.join(self.root, 'result', os.path.split(self.filePath)[-1])
            cv2.imwrite(save_path, resultImage)  # 保存结果
            print('图像已保存至 ', save_path)

            resultImage = resultImage[..., ::-1].copy()

            resultImage = QImage(resultImage.data, resultImage.shape[1], resultImage.shape[0], resultImage.strides[0],
                                 QImage.Format_RGB888)
            resultPixmap = QPixmap.fromImage(resultImage)

            self.resultLabel.setPixmap(resultPixmap.scaled(224, 224, Qt.KeepAspectRatio))


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = ImageProcessingWindow()
    window.show()
    sys.exit(app.exec_())
