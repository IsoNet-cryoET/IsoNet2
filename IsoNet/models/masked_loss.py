import torch
from torch import fft
import torch.nn as nn
import torch.nn.functional as F

def ssim_loss(x, y, window_size=11, size_average=True):
    # Gaussian kernel for SSIM computation
    def gaussian_window(window_size, sigma):
        #gauss = torch.Tensor([torch.exp(-(z - window_size // 2) ** 2 / (2 * sigma ** 2)) for z in range(window_size)])
        z = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(-(z - window_size // 2) ** 2 / (2 * sigma ** 2))
        gauss /= gauss.sum()
        return gauss / gauss.sum()

    # Create 3D Gaussian window
    channels = x.size(1)
    window = gaussian_window(window_size, 1.5).unsqueeze(1).repeat(1, channels, 1, 1, 1).to(x.device)
    mu_x = F.conv3d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv3d(y, window, padding=window_size // 2, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = F.conv3d(x * x, window, padding=window_size // 2, groups=channels) - mu_x_sq
    sigma_y = F.conv3d(y * y, window, padding=window_size // 2, groups=channels) - mu_y_sq
    sigma_xy = F.conv3d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim_map.mean() if size_average else 1 - ssim_map

def simple_loss(model_output, target, rot_mw_mask,loss_func='L2'):
    masked_tomo = apply_fourier_mask_to_tomo(tomo=target - model_output, mask=rot_mw_mask, output="real")
    if loss_func == "L2":
        loss = nn.MSELoss()
        return loss(masked_tomo,torch.zeros_like(masked_tomo))
    elif loss_func == "smoothL1":
        loss = nn.SmoothL1Loss()
        return loss(masked_tomo,torch.zeros_like(masked_tomo))
    elif loss_func == "smoothL1-SSIM":
        loss = nn.SmoothL1Loss()
        filtered_model_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=rot_mw_mask, output="real")
        return loss(masked_tomo,torch.zeros_like(masked_tomo)) + ssim_loss(filtered_model_output,target)
    else:
        print("loss name is not correct")

def masked_loss2(model_output, target, rot_mw_mask, mw_mask, mw_weight=2.0, loss_func=None):
    """
    The self-supervised per-sample loss function for denoising and missing wedge reconstruction.
    """
    outside_mw_mask = rot_mw_mask * mw_mask
    outside_mw_loss = (
        apply_fourier_mask_to_tomo(
            tomo=target - model_output, mask=outside_mw_mask, output="real"
        )
        .abs()
        .pow(2)
        .mean()
    )
    inside_mw_mask = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
    inside_mw_loss = (
        apply_fourier_mask_to_tomo(
            tomo=target - model_output, mask=inside_mw_mask, output="real"
        )
        .abs()
        .pow(2)
        .mean()
    )
    #loss = outside_mw_loss + mw_weight * inside_mw_loss
    #return loss
    return outside_mw_loss,inside_mw_loss

def masked_loss(model_output, target, rot_mw_mask, mw_mask, loss_func=None):
    # This is essence of deepdewedge
    # inside_mw_loss is for IsoNet
    # outside_mw_loss is for noise2noise
    outside_mw_mask = rot_mw_mask * mw_mask
    outside_mw_tomo = apply_fourier_mask_to_tomo(tomo=target - model_output, mask=outside_mw_mask, output="real")

    inside_mw_mask = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
    inside_mw_tomo = apply_fourier_mask_to_tomo(tomo=target - model_output, mask=inside_mw_mask, output="real")

    zero_target = torch.zeros_like(target)
    if loss_func == None:
        print("loss name is not correct")
    else:
        #filtered_model_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=rot_mw_mask, output="real")
        return [loss_func(outside_mw_tomo,zero_target),loss_func(inside_mw_tomo,zero_target)]

# def masked_loss(model_output, target, rot_mw_mask, mw_mask, mw_weight=2.0, loss_func='L2'):
#     # This is essence of deepdewedge
#     # inside_mw_loss is for IsoNet
#     # outside_mw_loss is for noise2noise
#     outside_mw_mask = rot_mw_mask * mw_mask
#     outside_mw_loss = (
#         apply_fourier_mask_to_tomo(
#             tomo=target - model_output, mask=outside_mw_mask, output="real"
#         )
#         .abs()
#         .pow(2)
#         .mean()
#     )
#     inside_mw_mask = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
#     inside_mw_loss = (
#         apply_fourier_mask_to_tomo(
#             tomo=target - model_output, mask=inside_mw_mask, output="real"
#         )
#         .abs()
#         .pow(2)
#         .mean()
#     )
#     # loss = outside_mw_loss + mw_weight * inside_mw_loss
#     return outside_mw_loss, inside_mw_loss

def fft_3d(tomo, norm="ortho"):
    fft_dim = (-1, -2, -3)
    return fft.fftshift(fft.fftn(tomo, dim=fft_dim, norm=norm), dim=fft_dim)


def ifft_3d(tomo, norm="ortho"):
    fft_dim = (-1, -2, -3)
    return fft.ifftn(fft.ifftshift(tomo, dim=fft_dim), dim=fft_dim, norm=norm)


def apply_fourier_mask_to_tomo(tomo, mask, output="real"):
    tomo_ft = fft_3d(tomo)
    tomo_ft_masked = tomo_ft * mask
    vol_filt = ifft_3d(tomo_ft_masked)
    if output == "real":
        return vol_filt.real
    elif output == "complex":
        return vol_filt


