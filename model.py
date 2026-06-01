import torch
import torch.nn as nn
import math
from torch.nn import init
import torch.nn.functional as F
from timm.layers import DropPath
import numpy as np
from utils import *
from data import transforms


class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, full_mask):
        x = transforms.r2c(x)
        x_in_k_space = torch.fft.fft2(x)
        masked_x_in_k_space = x_in_k_space * full_mask
        masked_x = torch.fft.ifft2(masked_x_in_k_space)
        masked_x = transforms.c2r(masked_x)
        return masked_x

class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self):
        super(DataConsistencyInKspace, self).__init__()

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def data_consistency(self,k, k0, mask):
        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        mask = mask.unsqueeze(0).unsqueeze(-1)  # 变成 [1, 256, 256, 1]
        mask = mask.expand(k.shape[0], -1, -1, k.shape[-1])  # 变成 [4, 256, 256, 2]
        out = (1 - mask) * k + mask * k0
        return out

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        x = x.permute(0, 2, 3, 1)
        k0 = k0.permute(0, 2, 3, 1)

        k = transforms.fft2(x)
        out = self.data_consistency(k, k0, mask)
        x_res = transforms.ifft2(out)
        x_res = x_res.permute(0, 3, 1, 2)
        return x_res

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)

class ChannelMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1/4, shift_pixel=1, hidden_rate=2,
                 key_norm=True):
        super().__init__()
        self.n_embd = n_embd
        self._init_weights()
        self.shift_pixel = shift_pixel
        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self):
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(xr)) * kv
        return x


class CCMix(nn.Module):
    def __init__(self, in_dims, target_dim, target_size):
        super(CCMix, self).__init__()
        self.target_dim = target_dim
        self.target_size = target_size
        # Projection layers to unify dimensions
        self.projections = nn.ModuleList([nn.Conv2d(in_dim, target_dim, kernel_size=1) for in_dim in in_dims])
        self.ln1 = nn.LayerNorm(target_dim * 3)
        self.drop_path = DropPath(0.05) if drop_path else nn.Identity()
        self.channel = ChannelMix(n_embd=target_dim * 3, channel_gamma=1 / 4, shift_pixel=1, hidden_rate=2)
        self.final_projections = nn.ModuleList([nn.Conv2d(target_dim, in_dim, kernel_size=1) for in_dim in in_dims])
        self.original_sizes = [target_size // 4, target_size // 2, target_size]

    def forward(self, features):
        # Step 1: Up-sample and project each feature map
        upsampled_features = []
        output_features = []
        for i, feature in enumerate(features):
            # Upsample to target size
            feature = F.interpolate(feature, size=self.target_size, mode='bilinear', align_corners=False)
            # Project to target dimension directly
            feature = self.projections[i](feature)
            upsampled_features.append(feature)

        # Step 2: Concatenate features along the channel axis
        concatenated = torch.cat(upsampled_features, dim=1)
        # Prepare for MHSA: (B, C, H, W) -> (B, N, C) where N=H*W
        B, C, H, W = concatenated.shape
        concatenated = concatenated.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        # Apply MHSA
        attn_output = concatenated + self.drop_path(
            self.ln1(self.channel(concatenated, (self.target_size, self.target_size))))

        # Step 3: Reshape back to (B, C, H, W)
        B, n_patch, hidden = attn_output.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidde
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        attn_output = attn_output.permute(0, 2, 1)
        attn_output = attn_output.contiguous().view(B, hidden, h, w)

        # Step 4: Split output back into four feature maps with original dimensions
        split_features = torch.split(attn_output, self.target_dim, dim=1)
        for i, split_feature in enumerate(split_features):
            # Project back to original dimension
            split_feature = self.final_projections[i](split_feature)  # (B, H, W, original_dim)
            # Resize to original size
            split_feature = F.interpolate(split_feature, size=self.original_sizes[i], mode='bilinear',
                                          align_corners=False)
            output_features.append(split_feature)

        return output_features

class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.ccmix = CCMix([32, 32, 32],32,256)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1, bias=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1, bias=True),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1,bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, 1, 1, bias=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1,bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, 1, 1, bias=True)
        )
    def forward(self, x, fft_forback, PhiTb, mask, h):
        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x

        if h[0] == None:
            x_D = F.conv2d(x_input, self.conv_D, padding=1)
        else:
            x_D = F.conv2d(x_input + h[0], self.conv_D, padding=1)

        if h[1] == None:
            x1 = self.encoder1(x_D)
        else:
            x1 = self.encoder1(x_D + h[1])

        if h[2] == None:
            x2 = self.encoder2(x1)
        else:
            x2 = self.encoder2(x1 + h[2])

        x_list1 = [x2, x1, x_D]
        x_list2 = self.ccmix(x_list1)

        x3 = self.decoder1(x2 + x_list2[0])
        x4 = self.decoder2(x3 + x_list2[1])
        x_G = F.conv2d(x4 + x_list2[2], self.conv_G, padding=1)
        x_pred = x_G + x_input
        x_list3 = [x_G, x4, x3]

        return x_pred, x_list3


class FDMATNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(FDMATNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

        self.conv_1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv_2 = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

        self.DC_layer = DataConsistencyInKspace()

    def forward(self, PhiTb, k0, mask):

        PhiTb = F.conv2d(PhiTb, self.conv_1, padding=1) # [4,32,256,256]
        x = PhiTb

        h = [None, None, None]

        for i in range(self.LayerNo):
            x, h = self.fcs[i](x, self.fft_forback, PhiTb, mask, h)

        x_out = F.conv2d(x, self.conv_2, padding=1)
        x_final = self.DC_layer(x_out, k0, mask)

        return x_final