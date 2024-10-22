
import mrcfile
import numpy as np
def read_mrc(filename):
    with mrcfile.open(filename, permissive=True) as mrc:
        data = mrc.data
        voxel_size = mrc.voxel_size.x
    return data, voxel_size

def write_mrc(filename, data, voxel_size=1, origin=0):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = voxel_size
        mrc.header.origin = origin

def read_defocus_file(format="ctffind4"):
    #TODO
    pass