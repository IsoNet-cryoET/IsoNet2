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

    IsoNet.py fsc3d -h
    IsoNet.py refine -h
    """


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
        from IsoNet.util.utils import process_gpuID, mkfolder
        from multiprocessing import cpu_count
        import mrcfile
        import numpy as np

        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   
        
        #GPU
        if gpuID is None:
            import torch
            gpu_list = list(range(torch.cuda.device_count()))
            gpuID=','.join(map(str, gpu_list))
            print("using all GPUs in this node: %s" %gpuID)  

        ngpus, gpuID, gpuID_list = process_gpuID(gpuID)

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=gpuID

        if batch_size is None:
            if ngpus == 1:
                batch_size = 4
            else:
                batch_size = 2 * len(gpuID_list)

        #CPU
        cpu_system = cpu_count()
        if cpu_system < ncpus:
            logging.info("requested number of cpus is more than the number of the cpu cores in the system")
            logging.info(f"setting ncpus to {cpu_system}")
            ncpus = cpu_system

        mkfolder(output_dir,remove=False)

        from IsoNet.bin.refine import run_refine
        run_refine(star_file=i,output_dir=output_dir, 
                    mixed_precision=False, epochs = epochs,
                   n_subvolume=n_subvolume, cube_size=cube_size, pretrained_model=pretrained_model,
                   batch_size = batch_size, ncpus=ncpus, acc_batches = acc_batches,predict_crop_size=predict_crop_size, learning_rate=learning_rate, limit_res= limit_res)

        logging.info("Finished")

    def refine_old(self, 
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
        from IsoNet.util.utils import process_gpuID, mkfolder
        from multiprocessing import cpu_count
        import mrcfile
        import numpy as np

        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   
        
        #GPU
        if gpuID is None:
            import torch
            gpu_list = list(range(torch.cuda.device_count()))
            gpuID=','.join(map(str, gpu_list))
            print("using all GPUs in this node: %s" %gpuID)  

        ngpus, gpuID, gpuID_list = process_gpuID(gpuID)

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=gpuID

        if batch_size is None:
            if ngpus == 1:
                batch_size = 4
            else:
                batch_size = 2 * len(gpuID_list)

        #CPU
        cpu_system = cpu_count()
        if cpu_system < ncpus:
            logging.info("requested number of cpus is more than the number of the cpu cores in the system")
            logging.info(f"setting ncpus to {cpu_system}")
            ncpus = cpu_system

        mkfolder(output_dir,remove=False)

        from IsoNet.bin.refine import run
        run(star_file=i,output_dir=output_dir, 
                    mixed_precision=False, epochs = epochs,
                   n_subvolume=n_subvolume, cube_size=cube_size, pretrained_model=pretrained_model,
                   batch_size = batch_size, acc_batches = acc_batches,predict_crop_size=predict_crop_size, learning_rate=learning_rate, limit_res= limit_res)

        logging.info("Finished")

    def predict(self, star_file: str, model: str, output_dir: str='./corrected_tomos', predict_both=True, gpuID: str = None, cube_size:int=64,
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
                tomo1 = mrc.data.copy()
            if 'rlnTomogram2Name' in star.columns:
                with mrcfile.open(tomo_row['rlnTomogram2Name']) as mrc:
                    tomo2 = mrc.data
            if predict_both:
                tomo = normalize(tomo1,percentile=False)
                outData = network.predict_map(tomo, output_dir).astype(np.float32) #train based on init model and save new one as model_iter{num_iter}.h5
                file_base_name = os.path.basename(tomo_row['rlnTomogramName'])
                file_name, file_extension = os.path.splitext(file_base_name)
                with mrcfile.new(f"{output_dir}/corrected_{file_name}.mrc") as mrc:
                    mrc.set_data(outData)

                tomo = normalize(tomo2,percentile=False)
                outData = network.predict_map(tomo, output_dir).astype(np.float32) #train based on init model and save new one as model_iter{num_iter}.h5
                file_base_name = os.path.basename(tomo_row['rlnTomogram2Name'])
                file_name, file_extension = os.path.splitext(file_base_name)
                with mrcfile.new(f"{output_dir}/corrected_{file_name}.mrc") as mrc:
                    mrc.set_data(outData)

    def resize(self, star_file:str, apix: float=15, out_folder="tomograms_resized"):
        '''
        This function rescale the tomograms to a given pixelsize
        '''
        import mrcfile
        import starfile
        import os
        import shutil
        star = starfile.read(star_file)
        from scipy.ndimage import zoom
        new_star = star.copy(deep=True)
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        for index, row in star.iterrows():
            old_tomo1_name = row["rlnTomogramName"]
            old_tomo2_name = row["rlnTomogram2Name"]
            old_tilt_name = row["rlnTiltFile"]
            ori_apix = float(row["rlnPixelSize"])
            zoom_factor = float(ori_apix)/apix

            tomo_folder = os.path.basename(os.path.dirname(old_tomo1_name))
            if not os.path.isdir(out_folder+'/'+tomo_folder):
                os.makedirs(out_folder+'/'+tomo_folder)
            
            if old_tomo1_name is not None:
                output_tomo1_name = out_folder+'/'+tomo_folder+'/'+os.path.basename(old_tomo1_name)
                with mrcfile.open(old_tomo1_name, permissive=True) as mrc:
                    data = mrc.data
                print("scaling: {}".format(output_tomo1_name))
                new_data = zoom(data, zoom_factor,order=3, prefilter=False)
                with mrcfile.new(output_tomo1_name,overwrite=True) as mrc:
                    mrc.set_data(new_data)
                    mrc.voxel_size = apix

            if old_tomo2_name is not None:
                output_tomo2_name = out_folder+'/'+tomo_folder+'/'+os.path.basename(old_tomo2_name)
                with mrcfile.open(old_tomo2_name, permissive=True) as mrc:
                    data = mrc.data
                print("scaling: {}".format(output_tomo2_name))
                new_data = zoom(data, zoom_factor,order=3, prefilter=False)
                with mrcfile.new(output_tomo2_name,overwrite=True) as mrc:
                    mrc.set_data(new_data)
                    mrc.voxel_size = apix   

            if old_tilt_name is not None:
                output_tilt_name = out_folder+'/'+tomo_folder+'/'+os.path.basename(old_tilt_name)
                shutil.copy(old_tilt_name, output_tilt_name)
                

            new_star.loc[index, "rlnTomogramName"] = output_tomo1_name
            new_star.loc[index,"rlnTomogram2Name"] = output_tomo2_name
            new_star.loc[index,"rlnPixelSize"] =  apix
            new_star.loc[index,"rlnTiltFile"] =  output_tilt_name
        starfile.write(new_star,out_folder+'.star')        
        print("scale_finished: {}".format(out_folder+'.star'))

    def powerlaw_filtering(self, 
                    h1: str,
                    o: str = "weighting.mrc",
                    mask: str=None, 
                    low_res: float=50,
                    ):
        """
        \nFlattening Fourier amplitude within the resolution range. This will sharpen the map. Low resolution is typically 10 and high resolution limit is typicaly the resolution at FSC=0.143\n
        """
        import numpy as np
        import mrcfile
        from numpy.fft import fftshift, fftn, ifftn

        with mrcfile.open(h1,'r') as mrc:
            input_map = mrc.data
            nz,ny,nx = input_map.shape
            voxel_size = mrc.voxel_size.x
            print(voxel_size)
            if voxel_size == 0:
                voxel_size = 1
            #logging.info("voxel_size",float(voxel_size))

        if mask is not None:
            with mrcfile.open(mask,'r') as mrc:
                mask = mrc.data
            input_map_masked = input_map * mask
        else:
            input_map_masked = input_map

        limit_r_low = int(voxel_size * nz / low_res)

        # power spectrum
        f1 = fftshift(fftn(input_map_masked))
        ret = (np.real(np.multiply(f1,np.conj(f1)))**0.5).astype(np.float32)

        #vet whitening filter
        r = np.arange(nz)-nz//2
        [Z,Y,X] = np.meshgrid(r,r,r)
        index = np.round(np.sqrt(Z**2+Y**2+X**2))

        F_curve = np.zeros(nz//2)
        F_map = np.zeros_like(ret)
        for i in range(nz//2):
            F_curve[i] = np.average(ret[index==i])

        eps = 1e-4
        k=(np.log(F_curve[limit_r_low])-0.01*np.log(F_curve[limit_r_low]))/(np.log(limit_r_low**-1)-np.log((nz//2)**-1))
        #k=np.log(1)/np.log(limit_r_low)
        #print(k)
        b = np.log(F_curve[limit_r_low]) - k * np.log(limit_r_low**-1)
        print(b)
        for i in range(nz//2):
            if i > limit_r_low:
                F_map[index==i] = F_curve[limit_r_low]/(F_curve[i]+eps) * np.exp(b+k*np.log(i**-1))
            else:
                F_map[index==i] = F_curve[limit_r_low]
        F_map = F_map/F_curve[limit_r_low]
        # apply filter
        F_input = fftn(input_map)
        out = ifftn(F_input*fftshift(F_map))
        out =  np.real(out).astype(np.float32)

        # with mrcfile.new("F.mrc", overwrite=True) as mrc:
        #     mrc.set_data(F_map)

        with mrcfile.new(o, overwrite=True) as mrc:
            mrc.set_data(out)
            mrc.voxel_size = voxel_size

    def psf(self,size=128,tilt_file=None,w=8, output="wedge.mrc",between_tilts=False):
        import numpy as np
        
        if tilt_file is None or tilt_file == 'None':
            tilt = np.linspace(-60,60,121)
            np.savetxt("tmp.tlt", tilt, fmt="%.02f")
            tilt_file = "tmp.tlt"

        tilt = np.loadtxt(tilt_file)
        if between_tilts:
            from IsoNet.util.geometry import draw_a_line
            out_mat = np.zeros([size,size], dtype = np.float32)
            for angle in tilt:
                out_mat = draw_a_line(out_mat, -angle, w)
            out_mat = np.repeat(out_mat[:, np.newaxis, :], size, axis=1)
        else:
            from IsoNet.util.missing_wedge import get_F_wedge
            max_angle = np.max(tilt)
            min_angle = np.min(tilt)
            out_mat = get_F_wedge(size=size, angle=60)
        import mrcfile
        with mrcfile.new(output, overwrite=True) as mrc:
            mrc.set_data(out_mat)

    def prepare_star(self, folder_name, output_star='tomograms.star', apix = None, defocus = 0.0, number_subtomos = 100):
        """
        \nThis command generates a tomograms.star file from a folder containing only tomogram files (.mrc or .rec).\n
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
        import mrcfile
        tomo_list = sorted(os.listdir(folder_name))
        print(tomo_list)
        data = []
        label = ['rlnIndex','rlnTomogramName','rlnTomogram2Name','rlnTiltFile','rlnPixelSize','rlnDefocus']

        voxel_size_initial = apix
        for i,tomo_name in enumerate(tomo_list):
            tomo_path = os.path.join(folder_name,tomo_name)
            files=os.listdir(tomo_path)
            tomo_file = []
            tilt_file = "None"
            for item in files:
                item_path = os.path.join(tomo_path,item)
                if item[-4:]=='.mrc' or item[-4:]=='.rec':
                    tomo_file.append(item_path)
                if item[-3:]=='tlt':
                    tilt_file = item_path
    

            if voxel_size_initial is None:
                with mrcfile.open(tomo_file[0]) as mrc:
                    voxel_size = mrc.voxel_size
                voxel_size = voxel_size.x
                if voxel_size == 0:
                    voxel_size = 1.0
            else:
                voxel_size = voxel_size_initial


            if len(tomo_file) == 1:
                tomo_file.append("None")
            data.append([i, tomo_file[0], tomo_file[1], tilt_file, voxel_size, defocus])    

        df = pd.DataFrame(data, columns = label)
        starfile.write(df,output_star)

    def extract(self,  star, subtomo_folder="subtomos", between_tilts=False, cube_size=128, crop_size=None):

        if crop_size is None:
            crop_size = cube_size + 16
        n_subtomo_per_tomo = 50
        import starfile
        import mrcfile
        import pandas as pd
        import numpy as np
        df = starfile.read(star)
        os.makedirs(subtomo_folder, exist_ok=True)
        from IsoNet.preprocessing.cubes import extract_subvolume, create_cube_seeds
        from IsoNet.preprocessing.img_processing import normalize
        particle_list = []
        for index, row in df.iterrows():
            tomo_index = row["rlnIndex"]
            tomo_folder = f"TS_{tomo_index:05d}"
            even_folder = os.path.join(subtomo_folder, tomo_folder, 'subtomo0')
            os.makedirs(even_folder, exist_ok=True)
            with mrcfile.open(row["rlnTomogramName"]) as mrc:
                tomo = mrc.data
            mask = np.ones_like(tomo)
            tomo = normalize(tomo,percentile=False)
            seeds=create_cube_seeds(tomo, n_subtomo_per_tomo, cube_size, mask)
            extract_subvolume(tomo, seeds, cube_size, even_folder)
            
            if "rlnTomogram2Name" in list(df.columns):
                odd_folder = os.path.join(subtomo_folder, tomo_folder, 'subtomo1')
                os.makedirs(odd_folder, exist_ok=True)
                with mrcfile.open(row["rlnTomogram2Name"]) as mrc:
                    tomo = mrc.data
                tomo = normalize(tomo,percentile=False)
                extract_subvolume(tomo, seeds, cube_size, odd_folder)

            wedge_path = os.path.join(subtomo_folder, tomo_folder, 'wedge.mrc')
            if "rlnTiltFile" in list(df.columns):
                print(wedge_path)
                self.reconstruct_psf(size=128, tilt_file=row["rlnTiltFile"], output=wedge_path, between_tilts=between_tilts)
            else:
                self.reconstruct_psf(size=128, output=wedge_path, between_tilts=between_tilts)

            for i in range(n_subtomo_per_tomo):
                im_name1 = '{}/subvolume{}_{:0>6d}.mrc'.format(even_folder, '', i)
                if "rlnTomogram2Name" in list(df.columns):
                    im_name2 = '{}/subvolume{}_{:0>6d}.mrc'.format(odd_folder, '', i)
                    particle_list.append([im_name1,im_name2,wedge_path])
                    label = ["rlnParticleName","rlnParticle2Name","rlnWedgeName"]                    
                    #particle_list.append([row["rlnTomogramName"],row["rlnTomogram2Name"],im_name1,im_name2,wedge_path])
                    #label = ["rlnTomogramName","rlnTomogram2Name","rlnParticleName","rlnParticle2Name","rlnWedgeName"]
                else:
                    particle_list.append([im_name1,wedge_path])
                    label = ["rlnParticleName","rlnWedgeName"]
                    #particle_list.append([row["rlnTomogram1Name"],im_name1,wedge_path])
                    #label = ["rlnTomogramName","rlnParticleName","rlnWedgeName"]

        df = pd.DataFrame(particle_list, columns = label)
        starfile.write(df,"subtomos.star")        

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
