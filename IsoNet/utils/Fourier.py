from numpy.fft import fftshift, fftn, ifftn, fft2, ifft2
import numpy as np
import torch

def apply_F_filter(input_map,F_map):
    # TODO if the two input size does not match

    F_input = fftn(input_map)
    out = ifftn(F_input*fftshift(F_map))
    out =  np.real(out).astype(np.float32)
    
    #TODO test something like this line 
    #deconv = np.real(scipy.fft.ifftn(scipy.fft.fftn(vol, overwrite_x=True, workers=ncpu) * ramp, overwrite_x=True, workers=ncpu))

    return out

def apply_F_filter_torch(input_map,F_map):
    fft_input = torch.fft.fftshift(torch.fft.fftn(input_map, dim=(-1, -2, -3)),dim=(-1, -2, -3))
    # mw_shift = torch.fft.fftshift(F_map, dim=(-1, -2, -3))
    out = torch.fft.ifftn(torch.fft.fftshift(fft_input*F_map, dim=(-1, -2, -3)),dim=(-1, -2, -3))
    out = torch.real(out)
    return out

# def apply_wedge(ori_data, mw=None, ld1 = 1, ld2 =0):
#     mw = np.fft.fftshift(mw)
#     mw = mw * ld1 + (1-mw) * ld2
#     f_data = np.fft.fftn(ori_data)
#     outData = mw*f_data
#     inv = np.fft.ifftn(outData)
#     outData = np.real(inv).astype(np.float32)
#     return outData

def apply_F_filter_2D(input_map,F_map):
    F_input = fft2(input_map)
    out = ifft2(F_input*fftshift(F_map, axes=(-2,-1)))
    out =  np.real(out).astype(np.float32)
    return out




