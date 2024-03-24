import torch
import torch.nn as nn
from torchvision import models


# 定义ResNet模型
class ResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet18, self).__init__()
        self.features = models.resnet18(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(num_classes=10).to(device)
    summary(model, (3, 224, 224))  # 输入图像尺寸为 (3, 224, 224)
