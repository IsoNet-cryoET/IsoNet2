#!/usr/bin/env python3
import fire
import logging
import os, sys, traceback
from IsoNet.util.dict2attr import check_parse
from fire import core

class ISONET:
    """
    ISONET: Train on tomograms and restore missing-wedge\n
    for detail discription, run one of the following commands:

    IsoNet.py refine -h
    """


    def prepare_star(self, full_folder="None",
                     even_folder="None",
                     odd_folder="None",
                     tilt_file_folder="None",
                     mask_folder='None',
                     coordinate_folder='None',
                     n_subtomo = 300,
                     output_star='tomograms.star',
                     pixel_size = 10.0, defocus = 0.0, 
                     number_subtomos = 100):
        """
        \n
        If there is no evn odd seperation, please specify tomo_folder
        If you have even and odd tomograms, please use the even_folder and odd_folder parameters
        
        This command generates a tomograms.star file from a folder containing only tomogram files (.mrc or .rec).\n
        isonet.py prepare_star folder_name [--output_star] [--pixel_size] [--defocus] [--number_subtomos]
        :param folder_name: (None) directory containing tomogram(s). Usually 1-5 tomograms are sufficient.
        :param output_star: (tomograms.star) star file similar to that from "relion". You can modify this file manually or with gui.
        :param pixel_size: (10) pixel size in angstroms. Usually you want to bin your tomograms to about 10A pixel size.
        Too large or too small pixel sizes are not recommended, since the target resolution on Z-axis of corrected tomograms should be about 30A.
        :param defocus: (0.0) defocus in Angstrom. Only need for ctf deconvolution. For phase plate data, you can leave defocus 0.
        If you have multiple tomograms with different defocus, please modify them in star file or with gui.
        :param number_subtomos: (100) Number of subtomograms to be extracted in later processes.
        If you want to extract different number of subtomograms in different tomograms, you can modify them in the star file generated with this command or with gui.

        """
        import starfile
        import pandas as pd
        import numpy as np
        
        data = []
        label = []
        if full_folder != "None":
            tomograms_files = sorted(os.listdir(full_folder))
            tomograms_files = [f"{full_folder}/{item}" for item in tomograms_files]
            data.append(tomograms_files)
            label += ['rlnTomoName']
        if even_folder != "None":
            even_files = sorted(os.listdir(even_folder))
            label += ['rlnTomoReconstructedTomogramHalf1']
            even_files = [f"{even_folder}/{item}" for item in even_files]
            data.append(even_files)
        if odd_folder != "None":
            odd_files = sorted(os.listdir(odd_folder))
            label += ['rlnTomoReconstructedTomogramHalf2']
            odd_files = [f"{odd_folder}/{item}" for item in odd_files]
            data.append(odd_files)
        if tilt_file_folder != "None":
            tilt_files = sorted(os.listdir(tilt_file_folder))
            tilt_files = [f"{tilt_file_folder}/{item}" for item in tilt_files]
            label += ['rlnTiltFile']
            data.append(tilt_files)
        if mask_folder != "None":
            mask_files = sorted(os.listdir(mask_folder))
            mask_files = [f"{mask_folder}/{item}" for item in mask_files]
            label += ['rlnMaskName']
            data.append(mask_files)
        if coordinate_folder != "None":
            coordinate_files = sorted(os.listdir(coordinate_folder))
            coordinate_files = [f"{coordinate_folder}/{item}" for item in coordinate_files]
            # TODO read each coordinate file and add _rlnCoordinateX, Y, Z
        
        #TODO defocus file folder 
        from IsoNet.util.fileio import read_defocus_file
        #TODO decide the defocus file format from CTFFIND and GCTF
        
        label += ['rlnNumberSubtomo']
        data.append([n_subtomo]*len(tomograms_files))

        data_length = len(data[0])
        data = list(map(list, zip(*data)))
        df = pd.DataFrame(data = data, columns = label)
        df.insert(0, 'rlnIndex', np.arange(data_length)+1)
        starfile.write(df,output_star)

    def refine(self, 
                   i: str,
                   gpuID: str=None,

                   limit_res: str=None,

                   ncpus: int=16, 
                   output_dir: str="isonet_maps",
                   pretrained_model: str=None,

                   epochs: int=50,
                   n_subvolume: int=1000, 
                   cube_size: int=64,
                   predict_crop_size: int=80,
                   batch_size: int=None, 
                   acc_batches: int=1,
                   learning_rate: float=3e-4
                   ):

        """
        \nTrain neural network to correct preffered orientation\n
        IsoNet.py map_refine half.mrc FSC3D.mrc --mask mask.mrc --limit_res 3.5 [--gpuID] [--ncpus] [--output_dir] [--fsc_file]...
        :param i1: Input half map 1
        :param i2: Input half map 2
        :param aniso_file: 3DFSC file
        :param mask: Filename of a user-provided mask
        :param independent: Independently process half1 and half2, this will disable the noise2noise-based denoising but will provide independent maps for gold-standard FSC
        :param gpuID: The ID of gpu to be used during the training.
        :param alpha: Ranging from 0 to inf. Weighting between the equivariance loss and consistency loss.
        :param beta: Ranging from 0 to inf. Weighting of the denoising. Large number means more denoising. 
        :param limit_res: Important! Resolution limit for IsoNet recovery. Information beyong this limit will not be modified.
        :param ncpus: Number of cpu.
        :param output_dir: The name of directory to save output maps
        :param pretrained_model: The neural network model with ".pt" to continue training or prediction. 
        :param reference: Retain the low resolution information from the reference in the IsoNet refine process.
        :param ref_resolution: The limit resolution to keep from the reference. Ususlly  10-20 A resolution. 
        :param epochs: Number of epochs.
        :param n_subvolume: Number of subvolumes 
        :param predict_crop_size: The size of subvolumes, should be larger then the cube_size
        :param cube_size: Size of cubes for training, should be divisible by 16, e.g. 32, 64, 80.
        :param batch_size: Size of the minibatch. If None, batch_size will be the max(2 * number_of_gpu,4). batch_size should be divisible by the number of gpu.
        :param acc_batches: If this value is set to 2 (or more), accumulate gradiant will be used to save memory consumption.  
        :param learning_rate: learning rate. Default learning rate is 3e-4 while previous IsoNet tomography used 3e-4 as learning rate
        """

        from IsoNet.preprocessing.img_processing import normalize
        from IsoNet.bin.map_refine import map_refine, map_refine_n2n
        from IsoNet.util.utils import process_gpuID, process_ncpus, process_batch_size, mkfolder
        import mrcfile
        import numpy as np

        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   
        
        ngpus, gpuID, gpuID_list = process_gpuID(gpuID)
        ncpus = process_ncpus(ncpus)
        batch_size = process_batch_size(batch_size, ngpus)
        
        mkfolder(output_dir,remove=False)

        from IsoNet.bin.refine import run
        run(star_file=i,output_dir=output_dir, 
                    mixed_precision=False, epochs = epochs,
                   n_subvolume=n_subvolume, cube_size=cube_size, pretrained_model=pretrained_model,
                   batch_size = batch_size, acc_batches = acc_batches,predict_crop_size=predict_crop_size, learning_rate=learning_rate, limit_res= limit_res)

        logging.info("Finished")


    def predict(self, star_file: str, model: str, output_dir: str='./corrected_tomos', gpuID: str = None, cube_size:int=64,
    crop_size:int=96,use_deconv_tomo=True, batch_size:int=None,normalize_percentile: bool=True,log_level: str="info", tomo_idx=None):
        """
        \nPredict tomograms using trained model\n
        isonet.py predict star_file model [--gpuID] [--output_dir] [--cube_size] [--crop_size] [--batch_size] [--tomo_idx]
        :param star_file: star for tomograms.
        :param output_dir: file_name of output predicted tomograms
        :param model: path to trained network model .h5
        :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
        :param cube_size: (64) The tomogram is divided into cubes to predict due to the memory limitation of GPUs.
        :param crop_size: (96) The side-length of cubes cropping from tomogram in an overlapping patch strategy, make this value larger if you see the patchy artifacts
        :param batch_size: The batch size of the cubes grouped into for network predicting, the default parameter is four times number of gpu
        :param normalize_percentile: (True) if normalize the tomograms by percentile. Should be the same with that in refine parameter.
        :param log_level: ("debug") level of message to be displayed, could be 'info' or 'debug'
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
        :raises: AttributeError, KeyError
        """

        if True:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
            datefmt="%m-%d %H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
            datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])

        from IsoNet.util.utils import mkfolder
        mkfolder(output_dir)
        from IsoNet.models.network import Net
        network = Net(filter_base = 64,unet_depth=3, add_last=True)
        network.load(model)
        import starfile
        import mrcfile
        import numpy as np
        from IsoNet.preprocessing.img_processing import normalize
        star = starfile.read(star_file)
        for index, tomo_row in star.iterrows():
            print(tomo_row)
            with mrcfile.open(tomo_row['rlnTomogramName']) as mrc:
                tomo = mrc.data.copy()
            if 'rlnTomogram2Name' in star.columns:
                with mrcfile.open(tomo_row['rlnTomogram2Name']) as mrc:
                    tomo += mrc.data
            tomo = normalize(tomo,percentile=False)
            outData = network.predict_map(tomo, output_dir).astype(np.float32) #train based on init model and save new one as model_iter{num_iter}.h5

            file_base_name = os.path.basename(tomo_row['rlnTomogramName'])
            file_name, file_extension = os.path.splitext(file_base_name)
            with mrcfile.new(f"{output_dir}/corrected_{file_name}.mrc") as mrc:
                mrc.set_data(outData)



    def check(self):
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])

        from IsoNet.bin.predict import predict
        from IsoNet.bin.refine import run
        import skimage
        import PyQt5
        import tqdm
        logging.info('IsoNet --version 1.0 alpha installed')
        logging.info(f"checking gpu speed")
        from IsoNet.bin.verify import verify
        fp16, fp32 = verify()
        logging.info(f"time for mixed/half precsion and single precision are {fp16} and {fp32}. ")
        logging.info(f"The first number should be much smaller than the second one, if not please check whether cudnn, cuda, and pytorch versions match.")

    def gui(self):
        """
        \nGraphic User Interface\n
        """
        import IsoNet.gui.Isonet_star_app as app
        app.main()

def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)

def pool_process(p_func,chunks_list,ncpu):
    from multiprocessing import Pool
    with Pool(ncpu,maxtasksperchild=1000) as p:
        # results = p.map(partial_func,chunks_gpu_num_list,chunksize=1)
        results = list(p.map(p_func,chunks_list))
    # return results

def main():
    core.Display = Display
    logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)
    if len(sys.argv) > 1:
       check_parse(sys.argv[1:])
    fire.Fire(ISONET)


if __name__ == "__main__":
    exit(main())
