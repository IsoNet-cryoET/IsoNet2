import os 
import sys
import logging
import sys
import mrcfile
from IsoNet.preprocessing.cubes import create_cube_seeds,crop_cubes,DataCubes
from IsoNet.preprocessing.img_processing import normalize
from IsoNet.util.missing_wedge import apply_wedge
from multiprocessing import Pool
import numpy as np
from functools import partial
from IsoNet.util.rotations import rotation_list
# from difflib import get_close_matches
#Make a new folder. If exist, nenew it
# Do not set basic config for logging here
# logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)

def extract_subtomos(settings):
    '''
    extract subtomo from whole tomogram based on mask
    and feed to generate_first_iter_mrc to generate xx_iter00.xx
    '''
    md = MetaData()
    md.read(settings.star_file)
    if len(md)==0:
        sys.exit("No input exists. Please check it in input folder!")

    subtomo_md = MetaData()
    subtomo_md.addLabels('rlnSubtomoIndex','rlnImageName','rlnCubeSize','rlnCropSize','rlnPixelSize')
    count=0
    for it in md:
        if settings.tomo_idx is None or str(it.rlnIndex) in settings.tomo_idx:
            pixel_size = it.rlnPixelSize
            if settings.use_deconv_tomo and "rlnDeconvTomoName" in md.getLabels() and os.path.isfile(it.rlnDeconvTomoName):
                logging.info("Extract from deconvolved tomogram {}".format(it.rlnDeconvTomoName))
                with mrcfile.open(it.rlnDeconvTomoName) as mrcData:
                    orig_data = mrcData.data.astype(np.float32)
            else:        
                logging.info("Extract from origional tomogram {}".format(it.rlnMicrographName))
                with mrcfile.open(it.rlnMicrographName) as mrcData:
                    orig_data = mrcData.data.astype(np.float32)
            
            if "rlnMaskName" in md.getLabels() and it.rlnMaskName not in [None, "None"]:
                with mrcfile.open(it.rlnMaskName) as m:
                    mask_data = m.data
            else:
                mask_data = None
                logging.info(" mask not been used for tomogram {}!".format(it.rlnIndex))

            seeds=create_cube_seeds(orig_data, it.rlnNumberSubtomo, settings.crop_size,mask=mask_data)
            subtomos=crop_cubes(orig_data,seeds,settings.crop_size)

            # save sampled subtomo to {results_dir}/subtomos instead of subtomo_dir (as previously does)
            base_name = os.path.splitext(os.path.basename(it.rlnMicrographName))[0]
            
            for j,s in enumerate(subtomos):
                im_name = '{}/{}_{:0>6d}.mrc'.format(settings.subtomo_dir, base_name, j)
                with mrcfile.new(im_name, overwrite=True) as output_mrc:
                    count+=1
                    subtomo_it = Item()
                    subtomo_md.addItem(subtomo_it)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnSubtomoIndex'), str(count))
                    subtomo_md._setItemValue(subtomo_it,Label('rlnImageName'), im_name)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnCubeSize'),settings.cube_size)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnCropSize'),settings.crop_size)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnPixelSize'),pixel_size)
                    output_mrc.set_data(s.astype(np.float32))
    subtomo_md.write(settings.subtomo_star)

def rotate_cubes(data):
    rotated_data = np.zeros((len(rotation_list), *data.shape))
    old_rotation = True
    if old_rotation:
        for i,r in enumerate(rotation_list):
            data = np.rot90(data, k=r[0][1], axes=r[0][0])
            data = np.rot90(data, k=r[1][1], axes=r[1][0])
            rotated_data[i] = data
    else:
        from scipy.ndimage import affine_transform
        from scipy.stats import special_ortho_group 
        for i in range(len(rotation_list)):
            rot = special_ortho_group.rvs(3)
            center = (np.array(data.shape) -1 )/2.
            offset = center-np.dot(rot,center)
            rotated_data[i] = affine_transform(data,rot,offset=offset)
    return rotated_data

def get_cubes(inp):
    '''
    current iteration mrc(in the 'results') + infomation from orignal subtomo
    normalized predicted + normalized orig -> normalize
    rotate by rotation_list and feed to get_cubes_one
    '''
    mrc1, mrc2, wedge_file, start, data_dir = inp

    with mrcfile.open(mrc1) as mrcData:
        data1 = mrcData.data.astype(np.float32) * -1
    with mrcfile.open(mrc2) as mrcData:
        data2 = mrcData.data.astype(np.float32) * -1
    with mrcfile.open(wedge_file) as mrcData:
        wedge = mrcData.data.astype(np.float32)
    
    rot_cube1 = rotate_cubes(data1)
    rot_cube2 = rotate_cubes(data2)
    rot_cube1_wedged = np.zeros_like(rot_cube1)
    for i in range(rot_cube1.shape[0]):
        with mrcfile.new('{}/train_x/x_{}.mrc'.format(data_dir, start+i), overwrite=True) as output_mrc:
            output_mrc.set_data(apply_wedge(rot_cube1[i], wedge).astype(np.float32))
        with mrcfile.new('{}/train_y/y_{}.mrc'.format(data_dir, start+i), overwrite=True) as output_mrc:
            output_mrc.set_data(rot_cube2[i].astype(np.float32))

def get_cubes_list(star, data_dir, ncpus):
    '''
    generate new training dataset:
    map function 'get_cubes' to mrc_list from subtomo_dir
    seperate 10% generated cubes into test set.
    '''
    import os
    dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for d in dirs_tomake:
        folder = '{}/{}'.format(data_dir, d)
        if not os.path.exists(folder):
            os.makedirs(folder)

    inp=[]
    mrc_list = star['rlnParticleName'].tolist()
    if 'rlnParticle2Name' in star.columns:
        mrc2_list = star['rlnParticle2Name'].tolist()
    wedge_list = star['rlnWedgeName'].tolist()
    for i,mrc in enumerate(mrc_list):
        inp.append((mrc,mrc2_list[i], wedge_list[i], i*len(rotation_list),data_dir))

    if ncpus > 1:
        with Pool(ncpus) as p:
            p.map(get_cubes,inp)
    else:
        for i in inp:
            get_cubes(i)


def get_noise_level(noise_level_tuple,noise_start_iter_tuple,iterations):
    assert len(noise_level_tuple) == len(noise_start_iter_tuple) and type(noise_level_tuple) in [tuple,list]
    noise_level = np.zeros(iterations+1)
    for i in range(len(noise_start_iter_tuple)-1):
        #remove this assert because it may not be necessary, and cause problem when iterations <3
        #assert i < iterations and noise_start_iter_tuple[i]< noise_start_iter_tuple[i+1]
        noise_level[noise_start_iter_tuple[i]:noise_start_iter_tuple[i+1]] = noise_level_tuple[i]
    assert noise_level_tuple[-1] < iterations 
    noise_level[noise_start_iter_tuple[-1]:] = noise_level_tuple[-1]
    return noise_level

def generate_first_iter_mrc(mrc,settings):
    '''
    Apply mw to the mrc and save as xx_iter00.xx
    '''
    # with mrcfile.open("fouriermask.mrc",'r') as mrcmask:
    root_name = mrc.split('/')[-1].split('.')[0]
    extension = mrc.split('/')[-1].split('.')[1]
    with mrcfile.open(mrc) as mrcData:
        orig_data = normalize(mrcData.data.astype(np.float32)*-1, percentile = settings.normalize_percentile)

    orig_data = apply_wedge(orig_data, ld1=1, ld2=0)
    
    #prefill = True
    if settings.prefill==True:
        rot_data = np.rot90(orig_data, axes=(0,2))
        rot_data = apply_wedge(rot_data, ld1=0, ld2=1)
        orig_data = rot_data + orig_data

    orig_data = normalize(orig_data, percentile = settings.normalize_percentile)
    with mrcfile.new('{}/{}_iter00.{}'.format(settings.result_dir,root_name, extension), overwrite=True) as output_mrc:
        output_mrc.set_data(-orig_data)

def prepare_first_iter(settings):
    if settings.ncpus >1:
        with Pool(settings.ncpus) as p:
            func = partial(generate_first_iter_mrc, settings=settings)
            p.map(func, settings.mrc_list)
    else:
        for i in settings.mrc_list:
            generate_first_iter_mrc(i,settings)
    return settings

