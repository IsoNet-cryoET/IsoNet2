
import torch
from torch import fft

# This is essence of deepdewedge
# inside_mw_loss it the same as IsoNet
# outside_mw_loss is for noise2noise

def masked_loss(model_output, target, rot_mw_mask, mw_mask, mw_weight=2.0):
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
    loss = outside_mw_loss + mw_weight * inside_mw_loss
    return loss


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