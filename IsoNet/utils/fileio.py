
import mrcfile
import numpy as np
import logging
import shutil
import os
import shutil

def create_folder(folder, remove=True):
    try:
        os.makedirs(folder)
    except FileExistsError:
        if remove:
            logging.warning(f"The {folder} folder already exists. The old {folder} folder will be moved to {folder}~")
            if os.path.exists(folder+'~'):
                shutil.rmtree(folder+'~')
            os.system('mv {} {}'.format(folder, folder+'~'))
            os.makedirs(folder)
        else:
            logging.info(f"The {folder} folder already exists, outputs will write into this folder")

def read_mrc(filename, inplace=False):
    if inplace:
        with mrcfile.mmap(filename, permissive=True) as mrc:
            data = mrc.data
            voxel_size = mrc.voxel_size.x
    else:
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