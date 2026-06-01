import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim

import numpy as np
import math
import random


def ssim_loss(gt, x_output):
    """
    计算 SSIM 损失函数
    :param gt: 真实图像，形状为 (batch_size, height, width)
    :param x_output: 生成图像，形状为 (batch_size, height, width)
    :return: 返回 SSIM 损失
    """
    # 初始化损失值
    batch_size = gt.shape[0]
    total_loss = 0

    # 对每个图像计算 SSIM
    for i in range(batch_size):
        gt_img = gt[i].cpu().data.numpy().reshape(256, 256)  # 真实图像
        x_img = x_output[i].cpu().data.numpy().reshape(256, 256)  # 重建图像

        # 确保图像值在 [0, 1] 之间
        x_img = np.clip(x_img, 0, 1).astype(np.float64)
        gt_img = gt_img.astype(np.float64)

        # 计算每张图像的 SSIM
        rec_SSIM = ssim(x_img, gt_img, data_range=1)  # 默认最大值为 1

        # 累加损失
        total_loss += (1 - rec_SSIM)

    # 计算批量图像的平均损失
    loss = total_loss / batch_size
    return loss

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def get_model_size(model):
    total_params = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()  # 只计算参数的数量

    return total_params  # 返回总参数量

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2 or data.size(-3) == 2
    return (data ** 2).sum(dim=-1).sqrt() if data.size(-1) == 2 else (data ** 2).sum(dim=-3).sqrt()

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)           # Set seed for python built-in random
    np.random.seed(seed_value)        # Set seed for numpy
    torch.manual_seed(seed_value)     # Set seed for pytorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def normalize_zero_to_one(data, eps=0.):
    data_min = float(data.min())
    data_max = float(data.max())
    return (data - data_min) / (data_max - data_min + eps)