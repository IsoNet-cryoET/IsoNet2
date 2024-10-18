import numpy  as np
def get_ctf1d(pixelsize, voltage, cs, defocus, amplitude, phaseshift, bfactor, length=2048):
    pixelsize = pixelsize*1e-10
    voltage = voltage * 1e3
    cs = cs * 1e-3
    defocus = -defocus*1e-6
    phaseshift = phaseshift / 180 * np.pi
    ny = 1 / pixelsize
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

def get_ctf2d(pixelsize, voltage, cs, defocus, amplitude, phaseshift, bfactor, length=2048):
    ctf = get_ctf1d(pixelsize, voltage, cs, defocus, amplitude, phaseshift, bfactor, length)
    s1 = - int(length / 2)
    f1 = s1 + length - 1
    m1 = np.arange(s1,f1+1)

    s2 = - int(length / 2)
    f2 = s2 + length - 1
    m2 = np.arange(s2,f2+1) 

    x, y = np.meshgrid(m1,m2)
    x = x.astype(np.float32) / np.abs(s1)
    y = y.astype(np.float32) / np.abs(s2)
    r = np.sqrt(x**2+y**2)

    data = np.arange(0,1+1/(length-1),1/(length-1))
    ramp = np.interp(r, data, ctf).astype(np.float32)
    return ramp

def fake_3DCTF_1(ctf2d, angles):
    assert ctf2d.shape[0] == len(angles)
    from IsoNet.util.WBP import backprojection
    ctf_ones = np.ones_like(ctf2d)
    ft_images = np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ctf2d+1, axes=(-2,-1))), axes=(-2,-1)))
    ft_ones = np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ctf_ones, axes=(-2,-1))), axes=(-2,-1)))
    out = backprojection(ft_images, angles, filter_name='ramp')
    out_ones = backprojection(ft_ones, angles, filter_name='ramp')
    out = np.abs(np.fft.fftshift(np.fft.ifftn(out))).astype(np.float32)
    out_ones = np.abs(np.fft.fftshift(np.fft.ifftn(out_ones))).astype(np.float32)
    diff = (out-out_ones)
    return diff/np.max(diff)

def fake_3DCTF_2(ctf2d, angles, repeats = 100):
    from IsoNet.util.Fourier import apply_F_filter_2D
    from IsoNet.util.WBP import backprojection
    from joblib import Parallel, delayed
    #result = np.zeros((ctf2d.shape[1], ctf2d.shape[1], ctf2d.shape[1]), dtype=np.float32)
    def simulate_one():
        noise = np.random.normal(size=ctf2d.shape)
        out = apply_F_filter_2D(noise, ctf2d)
        out = backprojection(out, angles, filter_name='ramp')
        F_result = np.fft.fftshift(np.fft.fftn(out))
        out = np.abs(F_result).astype(np.float32)
        #out = (np.real(F_result)).astype(np.float32)
        return out
    results = Parallel(n_jobs=-1)(delayed(simulate_one)() for _ in range(repeats))
    result = np.average(results, axis=0)
    return result



if __name__ == '__main__':
    ctf2d = get_ctf2d(10,300,2.7, 8, 0.1, 0, 0, 128)
    ctf2d = np.tile(ctf2d, (41,1,1))
    angles = np.linspace(-60,60,41)
    print(angles.shape)
    ctf3d = fake_3DCTF_1(ctf2d, angles)
    import mrcfile
    with mrcfile.new('ctf3d.mrc', overwrite=True) as mrc:
        mrc.set_data(ctf3d)
    pass