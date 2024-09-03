
from numpy.fft import fftshift, fftn, ifftn, fft2, ifft2
import numpy as np

def apply_F_filter(input_map,F_map):
    # TODO if the two input size does not match

    F_input = fftn(input_map)
    out = ifftn(F_input*fftshift(F_map))
    out =  np.real(out).astype(np.float32)
    
    #TODO test something like this line 
    #deconv = np.real(scipy.fft.ifftn(scipy.fft.fftn(vol, overwrite_x=True, workers=ncpu) * ramp, overwrite_x=True, workers=ncpu))

    return out

def apply_F_filter_2D(input_map,F_map):
    F_input = fft2(input_map)
    out = ifft2(F_input*fftshift(F_map, axes=(-2,-1)))
    out =  np.real(out).astype(np.float32)
    return out



