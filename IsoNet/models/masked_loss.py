import torch
from torch import fft
import torch.nn as nn
import torch.nn.functional as F


class FSCLoss(nn.Module):
    def __init__(self, eps=1e-6, min_shell=1):
        super().__init__()
        self.eps = eps
        self.min_shell = min_shell
        self._shell_cache = {}

    def _get_shell_index(self, shape, device):
        key = (tuple(shape), str(device))
        if key not in self._shell_cache:
            z = torch.arange(shape[0], device=device) - shape[0] // 2
            y = torch.arange(shape[1], device=device) - shape[1] // 2
            x = torch.arange(shape[2], device=device) - shape[2] // 2
            Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
            shell_index = torch.sqrt(Z.float() ** 2 + Y.float() ** 2 + X.float() ** 2).long()
            self._shell_cache[key] = shell_index.reshape(-1)
        return self._shell_cache[key]

    def forward(self, model_output, target):
        model_output = model_output.to(torch.float32)
        target = target.to(torch.float32)

        output_ft = fft_3d(model_output)
        target_ft = fft_3d(target)

        shell_index = self._get_shell_index(model_output.shape[-3:], model_output.device)
        max_shell = min(model_output.shape[-3:]) // 2
        valid_points = shell_index <= max_shell
        shell_index = shell_index[valid_points]

        output_ft = output_ft.reshape(output_ft.shape[0] * output_ft.shape[1], -1)
        target_ft = target_ft.reshape(target_ft.shape[0] * target_ft.shape[1], -1)

        losses = []
        for i in range(output_ft.shape[0]):
            output_shell = output_ft[i][valid_points]
            target_shell = target_ft[i][valid_points]

            numerator = torch.zeros(max_shell + 1, device=model_output.device, dtype=torch.float32)
            output_power = torch.zeros_like(numerator)
            target_power = torch.zeros_like(numerator)
            counts = torch.zeros_like(numerator)

            numerator.scatter_add_(0, shell_index, (output_shell * target_shell.conj()).real)
            output_power.scatter_add_(0, shell_index, output_shell.abs().pow(2))
            target_power.scatter_add_(0, shell_index, target_shell.abs().pow(2))
            counts.scatter_add_(0, shell_index, torch.ones_like(output_shell.real))

            fsc_curve = numerator / torch.sqrt(output_power * target_power + self.eps)
            fsc_curve = torch.clamp(fsc_curve, -1.0, 1.0)

            valid_shells = counts > 0
            valid_shells[:self.min_shell] = False
            if valid_shells.any():
                losses.append(1.0 - fsc_curve[valid_shells].mean())
            else:
                losses.append(torch.tensor(1.0, device=model_output.device, dtype=torch.float32))

        return torch.stack(losses).mean()

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
    filtered_model_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=rot_mw_mask, output="real")
    filtered_target = apply_fourier_mask_to_tomo(tomo=target, mask=rot_mw_mask, output="real")
    if loss_func == "L2":
        loss = nn.MSELoss()
        return loss(filtered_model_output, filtered_target)
    elif loss_func == "smoothL1":
        loss = nn.SmoothL1Loss()
        return loss(filtered_model_output, filtered_target)
    elif loss_func == "smoothL1-SSIM":
        loss = nn.SmoothL1Loss()
        return loss(filtered_model_output, filtered_target) + ssim_loss(filtered_model_output, target)
    elif loss_func == "FSC":
        return FSCLoss()(filtered_model_output, filtered_target)
    else:
        print("loss name is not correct")

def masked_loss2(model_output, target, rot_mw_mask, mw_mask, mw_weight=2.0, loss_func=None):
    """
    The self-supervised per-sample loss function for denoising and missing wedge reconstruction.
    """
    outside_mw_mask = rot_mw_mask * mw_mask
    outside_mw_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=outside_mw_mask, output="real")
    outside_mw_target = apply_fourier_mask_to_tomo(tomo=target, mask=outside_mw_mask, output="real")

    inside_mw_mask = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
    inside_mw_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=inside_mw_mask, output="real")
    inside_mw_target = apply_fourier_mask_to_tomo(tomo=target, mask=inside_mw_mask, output="real")

    if loss_func is None:
        loss_func = nn.MSELoss()

    outside_mw_loss = loss_func(outside_mw_output, outside_mw_target)
    inside_mw_loss = loss_func(inside_mw_output, inside_mw_target)
    #loss = outside_mw_loss + mw_weight * inside_mw_loss
    #return loss
    return outside_mw_loss,inside_mw_loss

def masked_loss(model_output, target, rot_mw_mask, mw_mask, loss_func=None):

    inside_mask = rot_mw_mask * mw_mask
    inside_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=inside_mask, output="real")
    inside_target = apply_fourier_mask_to_tomo(tomo=target, mask=inside_mask, output="real")

    outside_mask = rot_mw_mask * (torch.ones_like(mw_mask) - mw_mask)
    outside_output = apply_fourier_mask_to_tomo(tomo=model_output, mask=outside_mask, output="real")
    outside_target = apply_fourier_mask_to_tomo(tomo=target, mask=outside_mask, output="real")

    if loss_func == None:
        print("loss name is not correct")
    else:
        return [loss_func(outside_output, outside_target), loss_func(inside_output, inside_target)]

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


