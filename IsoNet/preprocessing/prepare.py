import logging
from IsoNet.utils.Fourier import apply_F_filter
from multiprocessing import Pool
from functools import partial
from IsoNet.utils.rotations import rotation_list
from IsoNet.utils.fileio import read_mrc, write_mrc
from IsoNet.utils.noise import get_noise
import numpy as np
from IsoNet.utils.processing import crop_to_size
from IsoNet.utils.missing_wedge import mw3D

def rotate_cubes(data):
    rotated_data = np.zeros((len(rotation_list), *data.shape))
    old_rotation = True
    if old_rotation:
        for i,r in enumerate(rotation_list):
            data_copy = np.rot90(data, k=r[0][1], axes=r[0][0])
            data_copy = np.rot90(data_copy, k=r[1][1], axes=r[1][0])
            rotated_data[i] = data_copy
    else:
        from scipy.ndimage import affine_transform
        from scipy.stats import special_ortho_group 
        for i in range(len(rotation_list)):
            rot = special_ortho_group.rvs(3)
            center = (np.array(data.shape) -1 )/2.
            offset = center-np.dot(rot,center)
            rotated_data[i] = affine_transform(data,rot,offset=offset)
    return rotated_data

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
    current_subtomo = '{}/{}_iter{:0>2d}.mrc'.format(settings['output_dir'],root_name,settings['iter_count']-1)
    iw_data, _ = read_mrc(origional_subtomo)
    ow_data, _ = read_mrc(current_subtomo)

    iw_data = normalize(-iw_data)
    ow_data = normalize(-ow_data)

    orig_data = apply_F_filter(ow_data, 1 - wedge_data) + apply_F_filter(iw_data, wedge_data)
    orig_data = normalize(orig_data)

    rotated_data = rotate_cubes(orig_data)
    
    datax = apply_F_filter(rotated_data, wedge_data)

    for i in range(len(rotation_list)): 
        data_X = crop_to_size(datax[i], settings['crop_size'], settings['cube_size'])
        data_Y = crop_to_size(rotated_data[i], settings['crop_size'], settings['cube_size'])
        if settings['noise_level_current'] > 0.0000001:
            data_X = data_X + get_noise(settings['noise_dir']) * settings['noise_level_current'] 
        start += 1
        write_mrc('{}/train_x/x_{}.mrc'.format(settings['data_dir'], start), data_X)
        write_mrc('{}/train_y/y_{}.mrc'.format(settings['data_dir'], start), data_Y)


def get_cubes_list(settings):
    '''
    generate new training dataset:
    map function 'get_cubes' to mrc_list from subtomo_dir
    seperate 10% generated cubes into test set.
    '''
    import os
    dirs_tomake = ['train_x', 'train_y']
    # TODO create data_dir folder with functions
    if not os.path.exists(settings['data_dir']):
        os.makedirs(settings['data_dir'])
    for d in dirs_tomake:
        folder = '{}/{}'.format(settings['data_dir'], d)
        if not os.path.exists(folder):
            os.makedirs(folder)


    wedge_data = mw3D(settings['crop_size'])
    inp=[]
    for i,mrc in enumerate(settings['mrc_list']):
        inp.append((mrc, wedge_data, i*len(rotation_list)))
    
    # inp: list 0f (mrc_dir, index * rotation times)

    if settings['ncpus'] > 1:
        func = partial(get_cubes, settings=settings)
        with Pool(settings['ncpus']) as p:
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
    #    orig_data = normalize(mrcData.data.astype(np.float32)*-1, percentile = settings['normalize_percentile)
    orig_data, _ = read_mrc(mrc)
    orig_data = apply_F_filter(orig_data*-1, mw3D(orig_data.shape[-1]))
    
    #prefill = True
    # if settings['prefill==True:
    #     rot_data = np.rot90(orig_data, axes=(0,2))
    #     rot_data = apply_wedge(rot_data, ld1=0, ld2=1)
    #     orig_data = rot_data + orig_data

    #TODO this normalize was removed necessary
    #orig_data = normalize(orig_data, percentile = settings['normalize_percentile)
    write_mrc('{}/{}_iter00.{}'.format(settings['output_dir'],root_name, extension),-orig_data)
    #with mrcfile.new(, overwrite=True) as output_mrc:
    #    output_mrc.set_data(-orig_data)

def prepare_first_iter(settings):
    if settings['ncpus'] >1:
        with Pool(settings['ncpus']) as p:
            func = partial(generate_first_iter_mrc, settings=settings)
            p.map(func, settings['mrc_list'])
    else:
        for i in settings['mrc_list']:
            generate_first_iter_mrc(i,settings)
    return settings

import starfile
from IsoNet.utils.fileio import create_folder
from IsoNet.utils.processing import normalize
from IsoNet.preprocessing.cubes import create_cube_seeds, extract_subvolume, extract_with_overlap
import mrcfile
import tqdm
def extract(star_file: str,
            input_column: str = "rlnDeconvTomoName",
            subtomo_dir="subtomos", 
            cube_size = 64,
            crop_size = None, 
            tomo_idx = None,
            uniform_extract=False):
    
    if crop_size is None:
        crop_size = cube_size + 16

    tomo_star = starfile.read(star_file)
    tomo_columns = tomo_star.columns.to_list()
    create_folder(subtomo_dir)
    particle_list = []
    for i, row in tqdm.tqdm(tomo_star.iterrows(), total=len(tomo_star), desc="Extracting subtomograms"):
    # for i, row in tomo_star.iterrows():
        if tomo_idx is None or str(row.rlnIndex) in tomo_idx: 

            tomo_name = row[input_column]
            tomo, _ = read_mrc(tomo_name)
            tomo = -normalize(-tomo)

            if "rlnMaskName" in tomo_columns and row["rlnMaskName"] not in [None, "None"]:
                mask_file = row["rlnMaskName"]
                with mrcfile.open(mask_file,permissive=True) as mrc:
                    mask=mrc.data
            else:
                mask = np.ones_like(tomo)
            count_start = len(particle_list)
            if uniform_extract:
                #TODO uniform extract with mask
                subtomos_names = extract_with_overlap(tomo, crop_size, cube_size, subtomo_dir, prefix='', wedge=None)
            else:
                n_subtomo_per_tomo = row["rlnNumberSubtomo"]
                seeds=create_cube_seeds(tomo, n_subtomo_per_tomo, crop_size, mask)
                subtomos_names = extract_subvolume(tomo, seeds, crop_size, subtomo_dir, count_start, wedge=None)
            
            
            for i in range(len(subtomos_names)):
                im_name = '{}/subvolume{}_{:0>6d}.mrc'.format(subtomo_dir, '', count_start+i)
                particle_list.append(im_name)

    return  particle_list