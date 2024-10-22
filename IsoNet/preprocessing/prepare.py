import logging
from IsoNet.utils.processing import normalize
from IsoNet.utils.Fourier import apply_F_filter
from multiprocessing import Pool
import numpy as np
from functools import partial
from IsoNet.utils.rotations import rotation_list
from IsoNet.utils.fileio import read_mrc, write_mrc
from IsoNet.utils.noise import get_noise
from IsoNet.utils.rotations import rotate_cubes
from IsoNet.utils.processing import crop_to_size
from IsoNet.utils.missing_wedge import mw3D

def get_cubes(inp, settings):
    '''
    current iteration mrc(in the 'results') + infomation from orignal subtomo
    normalized predicted + normalized orig -> normalize
    rotate by rotation_list and feed to get_cubes_one
    '''
    '''
    current iteration mrc(in the 'results') + infomation from orignal subtomo
    normalized predicted + normalized orig -> normalize
    rotate by rotation_list and feed to get_cubes_one
    '''
    origional_subtomo, wedge_data, start = inp
    root_name = origional_subtomo.split('/')[-1].split('.')[0]
    current_subtomo = '{}/{}_iter{:0>2d}.mrc'.format(settings.output_dir,root_name,settings.iter_count-1)
    iw_data, _ = read_mrc(origional_subtomo)
    ow_data, _ = read_mrc(current_subtomo)

    orig_data = apply_F_filter(ow_data, 1 - wedge_data) + apply_F_filter(iw_data, wedge_data)

    rotated_data = rotate_cubes(orig_data)
    
    datax = apply_F_filter(rotated_data, wedge_data)

    for i in range(len(rotation_list)): 
        data_X = crop_to_size(datax[i], settings.crop_size, settings.cube_size)
        data_Y = crop_to_size(rotated_data[i], settings.crop_size, settings.cube_size)
        if settings.noise_level_current > 0.0000001:
            data_X = data_X + get_noise(settings.noise_dir) * settings.noise_level_current 
        start += 1
        write_mrc('{}/train_x/x_{}.mrc'.format(settings.data_dir, start), data_X)
        write_mrc('{}/train_y/y_{}.mrc'.format(settings.data_dir, start), data_Y)


def get_cubes_list(settings):
    '''
    generate new training dataset:
    map function 'get_cubes' to mrc_list from subtomo_dir
    seperate 10% generated cubes into test set.
    '''
    import os
    dirs_tomake = ['train_x', 'train_y']
    # TODO create data_dir folder with functions
    if not os.path.exists(settings.data_dir):
        os.makedirs(settings.data_dir)
    for d in dirs_tomake:
        folder = '{}/{}'.format(settings.data_dir, d)
        if not os.path.exists(folder):
            os.makedirs(folder)


    wedge_data = mw3D(settings.crop_size)
    inp=[]
    for i,mrc in enumerate(settings.mrc_list):
        inp.append((mrc, wedge_data, i*len(rotation_list)))
    
    # inp: list 0f (mrc_dir, index * rotation times)

    if settings.ncpus > 1:
        func = partial(get_cubes, settings=settings)
        with Pool(settings.ncpus) as p:
            p.map(func,inp)
    else:
        for i in inp:
            logging.info("{}".format(i))
            get_cubes(i, settings)



def generate_first_iter_mrc(mrc,settings):
    '''
    Apply mw to the mrc and save as xx_iter00.xx
    '''
    # with mrcfile.open("fouriermask.mrc",'r') as mrcmask:
    root_name = mrc.split('/')[-1].split('.')[0]
    extension = mrc.split('/')[-1].split('.')[1]
    #with mrcfile.open(mrc) as mrcData:
    #    orig_data = normalize(mrcData.data.astype(np.float32)*-1, percentile = settings.normalize_percentile)
    orig_data, _ = read_mrc(mrc)
    orig_data = apply_F_filter(orig_data*-1, mw3D(orig_data.shape[-1]))
    
    #prefill = True
    # if settings.prefill==True:
    #     rot_data = np.rot90(orig_data, axes=(0,2))
    #     rot_data = apply_wedge(rot_data, ld1=0, ld2=1)
    #     orig_data = rot_data + orig_data

    #TODO this normalize was removed necessary
    #orig_data = normalize(orig_data, percentile = settings.normalize_percentile)
    write_mrc('{}/{}_iter00.{}'.format(settings.output_dir,root_name, extension),-orig_data)
    #with mrcfile.new(, overwrite=True) as output_mrc:
    #    output_mrc.set_data(-orig_data)

def prepare_first_iter(settings):
    if settings.ncpus >1:
        with Pool(settings.ncpus) as p:
            func = partial(generate_first_iter_mrc, settings=settings)
            p.map(func, settings.mrc_list)
    else:
        for i in settings.mrc_list:
            generate_first_iter_mrc(i,settings)
    return settings

