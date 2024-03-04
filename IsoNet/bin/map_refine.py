import logging
import numpy as np
from IsoNet.preprocessing.cubes import create_cube_seeds,crop_cubes
import mrcfile
from multiprocessing import Pool
from functools import partial
from IsoNet.util.utils import mkfolder
import skimage
from IsoNet.preprocessing.img_processing import normalize
import os
import sys
from IsoNet.util.plot_metrics import plot_metrics
import shutil
       
def rescale_fsc(fsc3d, crop_size):
    fsc3d = skimage.transform.resize(fsc3d, [crop_size,crop_size,crop_size])
    fsc3d[fsc3d<0] = 0
    if (fsc3d.max()-fsc3d.min()) >0.001:
        fsc3d = (fsc3d - fsc3d.min()) / (fsc3d.max()-fsc3d.min())
    return fsc3d


def extract_subvolume(current_map, seeds, crop_size, output_dir, prefix=''):
    subtomos=crop_cubes(current_map,seeds,crop_size)
    mrc_list = []
    for j,s in enumerate(subtomos):
        im_name = '{}/subvolume{}_{:0>6d}.mrc'.format(output_dir, prefix, j)
        with mrcfile.new(im_name, overwrite=True) as output_mrc:
            output_mrc.set_data(s.astype(np.float32))

        mrc_list.append(im_name)
    return mrc_list

def map_refine(halfmap, mask, fsc3d, alpha, voxel_size, epochs = 10, mixed_precision = False,
               output_dir = "results", output_base="half", n_subvolume = 50, pretrained_model=None,
               cube_size = 64, predict_crop_size=96, batch_size = 8, acc_batches=2, learning_rate= 3e-4, limit_res=None):

    data_dir = output_dir+"/"+output_base+"_data"
    mkfolder(data_dir)
    fsc3d_cube_small = rescale_fsc(fsc3d, cube_size)

    if limit_res is not None:
        from IsoNet.util.FSC import lowpass
        logging.info(f"IsoNet correction until resolution {limit_res}A!\n\
                     Information beyond {limit_res}A remains unchanged")
        halfmap = lowpass(halfmap, limit_res, voxel_size)
    else:
        logging.info(f"Does not limit resolution for IsoNet correction!")

    logging.info("Start preparing subvolumes!")
    halfmap = normalize(halfmap,percentile=False)
    seeds=create_cube_seeds(halfmap, n_subvolume, cube_size, mask)
    extract_subvolume(halfmap, seeds, cube_size, data_dir)
    logging.info("Done preparing subvolumes!")

    logging.info("Start training!")
    from IsoNet.models.network import Net
    network = Net(filter_base = 64,unet_depth=3, add_last=True)
    if pretrained_model is not None:
        print(f"loading previous model {pretrained_model}")
        network.load(pretrained_model)
    if epochs > 0:
        network.train(data_dir, output_dir, alpha=alpha, output_base=output_base, batch_size=batch_size, epochs = epochs, steps_per_epoch = 1000, 
                            mixed_precision=mixed_precision, acc_batches=acc_batches, learning_rate = learning_rate, fsc3d = fsc3d_cube_small) #train based on init model and save new one as model_iter{num_iter}.h5
    plot_metrics(network.metrics, f"{output_dir}/loss_{output_base}.png")

    logging.info("Start predicting!")           
    out_map = network.predict_map(halfmap, output_dir=output_dir, cube_size = cube_size, crop_size=predict_crop_size, output_base=output_base)

    if limit_res is None:
        out_name = f"{output_dir}/corrected_{output_base}.mrc"
    else:
        out_name = f"{output_dir}/corrected_{output_base}_filtered.mrc"

    with mrcfile.new(out_name, overwrite=True) as output_mrc:
        output_mrc.set_data(out_map.astype(np.float32))
        output_mrc.voxel_size = voxel_size

    files = os.listdir(output_dir)
    for item in files:
        if item == output_base+"_data" or item == output_base+"_data~":
            path = f'{output_dir}/{item}'
            shutil.rmtree(path)
        if item.startswith('subvolume'):
            path = f'{output_dir}/{item}'
            os.remove(path)      
    logging.info('Done predicting')
    

def map_refine_n2n(halfmap1, halfmap2, mask, fsc3d, alpha, beta, voxel_size, epochs = 10, mixed_precision = False,
               output_dir = "results", output_base1="half1", output_base2="half2", n_subvolume = 50, pretrained_model=None,
               cube_size = 64, predict_crop_size=96, batch_size = 8, acc_batches=2, learning_rate= 3e-4, debug_mode=False, limit_res=None):

    data_dir_1 = output_dir+"/"+output_base1+"_data"
    data_dir_2 = output_dir+"/"+output_base2+"_data"
    mkfolder(data_dir_1)
    mkfolder(data_dir_2)

    output_base0 = ""
    for count in range(len(output_base1)):
        if output_base1[count] == output_base2[count]:
            output_base0 += output_base1[count]

    if limit_res is not None:
        from IsoNet.util.FSC import lowpass
        logging.info(f"IsoNet correction until resolution {limit_res}A!\n\
                     Information beyond {limit_res}A remains unchanged")
        halfmap1 = lowpass(halfmap1, limit_res, voxel_size)
        halfmap2 = lowpass(halfmap2, limit_res, voxel_size)
    else:
        logging.info(f"Does not limit resolution for IsoNet correction!")

    fsc3d_cube_small = rescale_fsc(fsc3d, cube_size)

    logging.info("Start preparing subvolumes!")
    halfmap1 = normalize(halfmap1,percentile=False)
    halfmap2 = normalize(halfmap2,percentile=False)
    seeds=create_cube_seeds(halfmap1, n_subvolume, cube_size, mask)
    extract_subvolume(halfmap1, seeds, cube_size, data_dir_1)
    extract_subvolume(halfmap2, seeds, cube_size, data_dir_2)
    logging.info("Done preparing subvolumes!")

    logging.info("Start training!")
    from IsoNet.models.network_n2n import Net
    network = Net(filter_base = 64,unet_depth=3, add_last=True)
    if pretrained_model is not None:
        print(f"loading previous model {pretrained_model}")
        network.load(pretrained_model)

    debug_mode = True
    if epochs > 0:
        if debug_mode:
            for i in range(epochs):
                network.train([data_dir_1,data_dir_2], output_dir, alpha=alpha,beta=beta, output_base=output_base0, batch_size=batch_size, epochs = 1, steps_per_epoch = 1000, 
                            mixed_precision=mixed_precision, acc_batches=acc_batches, learning_rate = learning_rate, fsc3d = fsc3d_cube_small) #train based on init model and save new one as model_iter{num_iter}.h5
                in_map = (halfmap1+halfmap2)/2
                out_map = network.predict_map(in_map, output_dir=output_dir, cube_size = cube_size, crop_size=predict_crop_size, output_base=output_base1)
                with mrcfile.new(f"{output_dir}/corrected_epoch{i+1}.mrc", overwrite=True) as output_mrc:
                    output_mrc.set_data((out_map+in_map).astype(np.float32))
                    output_mrc.voxel_size = voxel_size
        else:
            network.train([data_dir_1,data_dir_2], output_dir, alpha=alpha,beta=beta, output_base=output_base0, batch_size=batch_size, epochs = epochs, steps_per_epoch = 1000, 
                mixed_precision=mixed_precision, acc_batches=acc_batches, learning_rate = learning_rate, fsc3d = fsc3d_cube_small) #train based on init model and save new one as model_iter{num_iter}.h5

    plot_metrics(network.metrics, f"{output_dir}/loss_{output_base1}.png")

    logging.info("Start predicting!")     
    out_map_full = network.predict_map((halfmap1+halfmap2)/2, output_dir=output_dir, cube_size = cube_size, crop_size=predict_crop_size, output_base=output_base1)
      
    out_map1 = network.predict_map(halfmap1, output_dir=output_dir, cube_size = cube_size, crop_size=predict_crop_size, output_base=output_base1)
    out_map2 = network.predict_map(halfmap2, output_dir=output_dir, cube_size = cube_size, crop_size=predict_crop_size, output_base=output_base2)

    if limit_res is None:
        out_name1 = f"{output_dir}/corrected_{output_base1}.mrc"
        out_name2 = f"{output_dir}/corrected_{output_base2}.mrc"
        out_name_full = f"{output_dir}/corrected_full.mrc"
    else:
        out_name1 = f"{output_dir}/corrected_{output_base1}_filtered.mrc"
        out_name2 = f"{output_dir}/corrected_{output_base2}_filtered.mrc"

    with mrcfile.new(out_name1, overwrite=True) as output_mrc:
        output_mrc.set_data(out_map1.astype(np.float32))
        output_mrc.voxel_size = voxel_size
    with mrcfile.new(out_name2, overwrite=True) as output_mrc:
        output_mrc.set_data(out_map2.astype(np.float32))
        output_mrc.voxel_size = voxel_size
    with mrcfile.new(out_name_full, overwrite=True) as output_mrc:
        output_mrc.set_data(out_map_full.astype(np.float32))
        output_mrc.voxel_size = voxel_size

    files = os.listdir(output_dir)
    for item in files:
        if item == output_base1+"_data" or item == output_base1+"_data~" or item == output_base2+"_data" or item == output_base2+"_data~" :
            path = f'{output_dir}/{item}'
            shutil.rmtree(path)
        if item.startswith('subvolume'):
            path = f'{output_dir}/{item}'
            os.remove(path)      
    logging.info('Done predicting')