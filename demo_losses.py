import torch
from Losses_1 import *

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
