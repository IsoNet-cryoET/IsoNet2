
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



# def create_folder(directory_name):
#     if os.path.exists(directory_name):
#         new_name = directory_name + '~'
#         if os.path.exists(new_name):
#             shutil.rmtree(new_name)
#         shutil.move(directory_name, new_name)
#         print(f"Directory '{directory_name}' already existed. Renamed to '{new_name}'.")
    
#     os.makedirs(directory_name)
#     print(f"Created new empty directory '{directory_name}'.")

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