import torch
from Losses_1 import *

"""
参考：https://piqa.readthedocs.io/en/stable/api/piqa.psnr.html
     https://github.com/francois-rozet/piqa
     
分为求指标和loss的梯度下降
"""

#####################################################################
# TODO:PSNR
# metric
x = torch.rand(5, 3, 256, 256)
y = torch.rand(5, 3, 256, 256)
l = psnr(x, y)
l = mse(x, y)

# loss
criterion = PSNR()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
y = torch.rand(5, 3, 256, 256)
l = -criterion(x, y)
l.backward()

#####################################################################
# TODO:SSIM
# metric
x = torch.rand(5, 3, 64, 64, 64)
y = torch.rand(5, 3, 64, 64, 64)
kernel = gaussian_kernel(7).repeat(3, 1, 1)
ss, cs = ssim(x, y, kernel)

# loss
criterion = SSIM()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
y = torch.rand(5, 3, 256, 256)
l = 1 - criterion(x, y)
l.backward()

#####################################################################
# TODO:MS_SSIM
# metric
x = torch.rand(5, 3, 256, 256)
y = torch.rand(5, 3, 256, 256)
kernel = gaussian_kernel(7).repeat(3, 1, 1)
weights = torch.rand(5)
l = ms_ssim(x, y, kernel, weights)

# loss
criterion = MS_SSIM()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
y = torch.rand(5, 3, 256, 256)
l = 1 - criterion(x, y)
l.backward()

#####################################################################
# TODO:LPIPS
# loss
criterion = LPIPS()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
y = torch.rand(5, 3, 256, 256)
l = criterion(x, y)
l.backward()

#####################################################################
# TODO:GMSD
# metric
x = torch.rand(5, 1, 256, 256)
y = torch.rand(5, 1, 256, 256)
kernel = gradient_kernel(prewitt_kernel())
l = gmsd(x, y, kernel)

# loss
criterion = GMSD()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
y = torch.rand(5, 3, 256, 256)
l = criterion(x, y)
l.backward()

#####################################################################
# TODO:MS_GMSD
# metric
x = torch.rand(5, 1, 256, 256)
y = torch.rand(5, 1, 256, 256)
kernel = gradient_kernel(prewitt_kernel())
weights = torch.rand(4)
l = ms_gmsd(x, y, kernel, weights)

# loss
criterion = MS_GMSD()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
y = torch.rand(5, 3, 256, 256)
l = criterion(x, y)
l.backward()

#####################################################################
# TODO:MDSI
# metric
x = torch.rand(5, 1, 256, 256)
y = torch.rand(5, 1, 256, 256)
kernel = gradient_kernel(prewitt_kernel())
l = mdsi(x, y, kernel)

# loss
criterion = MDSI()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
y = torch.rand(5, 3, 256, 256)
l = criterion(x, y)
l.backward()

#####################################################################
# TODO:HaarPSI
# metric
x = torch.rand(5, 3, 256, 256)
y = torch.rand(5, 3, 256, 256)
l = haarpsi(x, y)

# loss
criterion = HaarPSI()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
y = torch.rand(5, 3, 256, 256)
l = 1 - criterion(x, y)
l.backward()

#####################################################################
# TODO:VSI
# metric
x = torch.rand(5, 3, 256, 256)
y = torch.rand(5, 3, 256, 256)
filtr = sdsp_filter(x)
vs_x, vs_y = sdsp(x, filtr), sdsp(y, filtr)
kernel = gradient_kernel(scharr_kernel())
l = vsi(x, y, vs_x, vs_y, kernel)

# loss
criterion = VSI()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
y = torch.rand(5, 3, 256, 256)
l = 1 - criterion(x, y)
l.backward()

#####################################################################
# TODO:TV
# metric
x = torch.rand(5, 3, 256, 256)
l = tv(x)

# loss
criterion = TV()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
l = criterion(x)
l.backward()

#####################################################################
# TODO:FSIM
# metric
x = torch.rand(5, 3, 256, 256)
y = torch.rand(5, 3, 256, 256)
filters = pc_filters(x)
pc_x = phase_congruency(x[:, :1], filters)
pc_y = phase_congruency(y[:, :1], filters)
kernel = gradient_kernel(scharr_kernel())
l = fsim(x, y, pc_x, pc_y, kernel)

# loss
criterion = FSIM()
x = torch.rand(5, 3, 256, 256, requires_grad=True)
y = torch.rand(5, 3, 256, 256)
l = 1 - criterion(x, y)
l.backward()

#####################################################################
# TODO:FID
# loss
criterion = FID()
x = torch.randn(1024, 256, requires_grad=True)
y = torch.randn(2048, 256)
l = criterion(x, y)
l.backward()

#####################################################################
# TODO:CX
# metric
x = torch.rand(5, 3, 64, 64)
y = torch.rand(5, 3, 64, 64)
l = contextual_loss(x, y)

# loss
x = torch.rand(5, 3, 64, 64, requires_grad=True)
y = torch.rand(5, 3, 64, 64)
criterion = ContextualLoss()
l = criterion(x, y)
l.backward()

#####################################################################
# TODO:Cobi
# 设置测试参数
feature_model_extractor_nodes = ["features.2", "features.7", "features.12"]
feature_model_normalize_mean = [0.485, 0.456, 0.406]
feature_model_normalize_std = [0.229, 0.224, 0.225]
patch_size = 10
stride = 1
weight_spatial = 0.5
bandwidth = 1.0
loss_type = "vgg"  # "vgg" 或者 "rgb"

# 创建测试数据
batch_size = 2
channels = 3
height = 64
width = 64
sr_tensor = torch.rand(batch_size, channels, height, width).cuda()
gt_tensor = torch.rand(batch_size, channels, height, width).cuda()

# 创建_ContextualBilateralLoss对象
criterion = _ContextualBilateralLoss(
    feature_model_extractor_nodes=feature_model_extractor_nodes,
    feature_model_normalize_mean=feature_model_normalize_mean,
    feature_model_normalize_std=feature_model_normalize_std,
    patch_size=patch_size,
    stride=stride,
    weight_spatial=weight_spatial,
    bandwidth=bandwidth,
    loss_type=loss_type
).cuda()

# 运行前向传播
l = criterion(sr_tensor, gt_tensor)
l.requires_grad = True  # 设置requires_grad=True
l.backward()
