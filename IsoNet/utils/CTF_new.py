import numpy  as np
from scipy.fft import fft2,ifft2, fftshift

def get_ctf1d_old(angpix, voltage, cs, defocus, amplitude, phaseshift, bfactor, length=2048):
    angpix = angpix*1e-10
    voltage = voltage * 1e3
    cs = cs * 1e-3
    defocus = -defocus*1e-6
    phaseshift = phaseshift / 180 * np.pi
    ny = 1 / angpix
    lambda1 = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
    lambda2 = lambda1 * 2
    points = np.arange(0,length)
    points = points.astype(float)
    points = points/(2 * length)*ny
    k2 = points**2
    term1 = lambda1**3 * cs * k2**2
    w = np.pi / 2. * (term1 + lambda2 * defocus * k2) - phaseshift
    acurve = np.cos(w) * amplitude
    pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w)
    bfactor = np.exp(-bfactor * k2 * 0.25)
    return (pcurve + acurve)*bfactor

def get_ctf1d(angpix, voltage, cs, defocus, amplitude, phaseshift, bfactor, length=2048, clip_first_peak=False):
    angpix = angpix * 1e-10  # Å to meters
    voltage = voltage * 1e3  # kV to V
    cs = cs * 1e-3           # mm to meters
    defocus = -defocus * 1e-6  # um to meters
    phaseshift = phaseshift / 180 * np.pi  # degrees to radians

    ny = 1 / angpix
    lambda1 = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
    lambda2 = lambda1 * 2

    points = np.arange(0, length).astype(float)
    points = points / (2 * length) * ny
    k2 = points ** 2

    term1 = lambda1**3 * cs * k2**2
    w = np.pi / 2. * (term1 + lambda2 * defocus * k2) - phaseshift

    acurve = np.cos(w) * amplitude
    pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w)
    ctf = (pcurve + acurve)

    bfactor_term = np.exp(-bfactor * k2 * 0.25)
    ctf *= bfactor_term

    if clip_first_peak:
        for item in range(len(ctf)-1):
            if ctf[item-1] < ctf[item] and ctf[item+1]<=ctf[item]:
                break
        ctf[:item] = ctf[item]
    return ctf


def get_ctf2d(angpix, voltage, cs, defocus, amplitude, phaseshift, bfactor, shape, clip_first_peak):
    """
    Returns 2D CTF array interpolated from 1D CTF curve.

    Parameters:
        shape (tuple): shape of the output filter (ny, nx)
    """
    ny, nx = shape
    length = max(ny, nx)  # used for 1D CTF curve resolution

    # Get 1D CTF
    ctf1d = get_ctf1d(
        angpix=angpix,
        voltage=voltage,
        cs=cs,
        defocus=defocus,
        amplitude=amplitude,
        phaseshift=phaseshift,
        bfactor=bfactor,
        length=length,
        clip_first_peak=clip_first_peak
    )

    # Normalized radius map over [0,1]
    y = np.linspace(-1, 1, ny, endpoint=False)
    x = np.linspace(-1, 1, nx, endpoint=False)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2)
    r = np.clip(r, 0, 1)

    # Interpolate CTF 1D to 2D
    ctf2d = np.interp(r, np.linspace(0, 1, length), ctf1d).astype(np.float32)
    return ctf2d

import numpy as np

def get_ctf3d(angpix, voltage, cs, defocus, amplitude, phaseshift, bfactor, shape, clip_first_peak=False):
    """
    Returns 3D CTF array interpolated from 1D CTF curve.

    Parameters:
        shape (tuple): shape of the output filter (nz, ny, nx)
    """
    nz, ny, nx = shape
    length = max(nz, ny, nx)  # match max dimension for sampling in 1D CTF

    # Get 1D CTF
    ctf1d = get_ctf1d(
        angpix=angpix,
        voltage=voltage,
        cs=cs,
        defocus=defocus,
        amplitude=amplitude,
        phaseshift=phaseshift,
        bfactor=bfactor,
        length=length,
        clip_first_peak=clip_first_peak
    )

    # Normalized 3D radius map over [0,1]
    z = np.linspace(-1, 1, nz, endpoint=False)
    y = np.linspace(-1, 1, ny, endpoint=False)
    x = np.linspace(-1, 1, nx, endpoint=False)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='xy')
    r = np.sqrt(xv**2 + yv**2 + zv**2)
    r = np.clip(r, 0, 1)

    # Interpolate CTF 1D to 3D
    ctf3d = np.interp(r, np.linspace(0, 1, length), ctf1d).astype(np.float32)
    return ctf3d

def get_wiener_1d(angpix, voltage, cs, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift, amplitude, length):
    #data = np.arange(0,1+1/2047.,1/2047.)
    data = np.linspace(0,1,length)
    highpass = np.minimum(np.ones(data.shape[0]), data/highpassnyquist) * np.pi
    highpass = 1-np.cos(highpass)
    eps = 1e-6
    snr = np.exp(-data * snrfalloff * 100 / angpix) * (10**deconvstrength) * highpass + eps
    ctf = get_ctf1d(angpix=angpix, voltage=voltage, cs=cs, defocus=defocus, amplitude=amplitude, phaseshift=phaseshift, bfactor=0, length=length)
    if phaseflipped:
        ctf = abs(ctf)
    wiener = ctf/(ctf*ctf+1/snr)
    return wiener

def get_wiener_2d(angpix, voltage, cs, defocus, snrfalloff, deconvstrength,
                  highpassnyquist, phaseflipped, phaseshift, amplitude, shape):
    """
    shape: tuple of (ny, nx) for the target 2D filter shape
    """
    ny, nx = shape
    length = max(ny, nx)  # consistent length

    wiener1d = get_wiener_1d(
        angpix=angpix,
        voltage=voltage,
        cs=cs,
        defocus=defocus,
        snrfalloff=snrfalloff,
        deconvstrength=deconvstrength,
        highpassnyquist=highpassnyquist,
        phaseflipped=phaseflipped,
        phaseshift=phaseshift,
        amplitude=amplitude,
        length=length
    )

    # Generate 2D coordinate grid normalized to [-1, 1)
    y = np.linspace(-1, 1, ny, endpoint=False)
    x = np.linspace(-1, 1, nx, endpoint=False)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2)
    r = np.clip(r, 0, 1)

    # Interpolate using correct length
    wiener2d = np.interp(r, np.linspace(0, 1, length), wiener1d).astype(np.float32)
    return wiener2d

def get_wiener_3d(angpix, voltage, cs, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift, amplitude, length):
    wiener1d = get_wiener_1d(angpix, voltage, cs, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift,amplitude, length)
    

    s1 = - int(length / 2)
    f1 = s1 + length - 1
    m1 = np.arange(s1,f1+1)

    s2 = - int(length / 2)
    f2 = s2 + length - 1
    m2 = np.arange(s2,f2+1)

    s3 = - int(length / 2)
    f3 = s3 + length - 1
    m3 = np.arange(s3,f3+1)

    x, y, z = np.meshgrid(m1,m2,m3)

    x = x.astype(np.float32)
    x = x / np.abs(s1)
    x = x**2

    y = y.astype(np.float32) 
    y = y / np.abs(s2)
    y = y**2

    z = z.astype(np.float32) 
    z = z / np.maximum(1, np.abs(s3))
    z = z**2

    r = x + y + z

    r = np.sqrt(r)
    r = np.minimum(1, r)
    ramp = np.interp(r, np.linspace(0,1,length), wiener1d).astype(np.float32)
    return ramp

def apply_filter_2d(image, filter2d):
    if image.shape != filter2d.shape:
        raise ValueError(f"Image shape {image.shape} does not match filter shape {filter2d.shape}")
    deconv = fft2(image, overwrite_x=True, workers=8)
    deconv = np.real(ifft2(deconv * fftshift(filter2d), overwrite_x=True, workers=8))
    deconv = deconv.astype(np.float32)

    return deconv

def apply_wiener_2d(image, defocus, angpix=1, voltage=300, cs=2.7,
                           snrfalloff=0, deconvstrength=0, highpassnyquist=0.01,
                           phaseflipped=0, phaseshift=0, amplitude=0.1):
    # ny, nx = data.shape
    # length = max(ny, nx)
    # sy = (length - ny) // 2
    # sx = (length - nx) // 2

    wiener_filter = get_wiener_2d(
        angpix=angpix,
        voltage=voltage,
        cs=cs,
        defocus=defocus,
        snrfalloff=snrfalloff,
        deconvstrength=deconvstrength,
        highpassnyquist=highpassnyquist,
        phaseflipped=phaseflipped,
        phaseshift=phaseshift,
        amplitude=amplitude,
        shape=image.shape
    )

    # wiener_filter_cropped = wiener_filter[sy:sy+ny, sx:sx+nx]
    return apply_filter_2d(image, wiener_filter)

def apply_ctf2d(image, defocus, angpix=1, voltage=300, cs=2.7,
                amplitude=0.1, phaseshift=0, bfactor=0, sign_only=False,clip_first_peak=True):
    """
    Applies 2D CTF to an input image using Fourier transform.

    Parameters:
        image (ndarray): input 2D image
        defocus (float): defocus in micrometers
    """
    ctf_filter = get_ctf2d(
        angpix=angpix,
        voltage=voltage,
        cs=cs,
        defocus=defocus,
        amplitude=amplitude,
        phaseshift=phaseshift,
        bfactor=bfactor,
        shape=image.shape,
        clip_first_peak=clip_first_peak
    )
    if sign_only:
        ctf_filter = np.sign(ctf_filter)
    return apply_filter_2d(image, ctf_filter)

if __name__ == '__main__':
    w=get_ctf3d(angpix=8, voltage=300, cs=2.7, defocus=2.4, amplitude=0.1, phaseshift=0, bfactor=0, shape=[96,96,96], clip_first_peak=False)
    # c = get_wiener_2d(angpix=5.4, voltage=300, cs=2.7, defocus=3.8,amplitude=0.1, snrfalloff=0, deconvstrength=0, highpassnyquist=0.02, phaseflipped=0, phaseshift=0,length=96)

    from fileio import write_mrc, read_mrc
    # write_mrc('ctf.mrc',c)
    write_mrc('ctf3-keepfirstpeaks.mrc',abs(w))

