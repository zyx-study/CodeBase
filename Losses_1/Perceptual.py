import torch
from torch import nn
from torchvision.models.vgg import vgg16

__all__ = [
    "PerceptualLoss"
]


class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        return perception_loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成随机数据
    img1 = torch.rand(4, 3, 224, 224).to(device)
    img2 = torch.rand(4, 3, 224, 224).to(device)

    # 初始化PerceptualLoss模块
    perceptual_loss = PerceptualLoss().to(device)

    # 计算损失
    loss = perceptual_loss(img1, img2)

    print("Perceptual Loss:", loss.item())
