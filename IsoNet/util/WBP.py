
from skimage.transform import iradon
import numpy as np
def backprojection(projections, angles, filter_name='ramp'):
    # x in the angle, y, x order
    reconstructed_volume = np.zeros((projections.shape[2],projections.shape[2],projections.shape[2]))
    sinogram = projections.transpose()
    # to x y angle
    for i in range(projections.shape[1]):
        reconstructed_slice = iradon(sinogram[:,i,:], theta=angles, filter_name=filter_name)#, circle=True)
        reconstructed_volume[i] = reconstructed_slice
    return reconstructed_volume.astype(np.float32).transpose((1,0,2))


#TODO fourier insersion from 2D to 3D
# relion uses a iterative gridding reconstruction algorithm for 3D reconstruction
# https://www.sciencedirect.com/science/article/pii/S1047847712002481
# https://www.sciencedirect.com/science/article/pii/S0969212615002798
# Also RELION4 said 3DCTF can not correct represent low resolution information where Fourier component overlap
# This is kind of make sense but I still do not fully understand




if __name__ == '__main__':
    pass
    # import numpy as np
    # image = np.ones((41,128,128), dtype=np.float32)
    # vol = fake_3DCTF(image, np.linspace(-60,60,41))
    # import mrcfile
    # with mrcfile.new('noise.mrc', overwrite=True) as mrc:
    #     mrc.set_data(vol)