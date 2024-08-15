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


    def refine_old(self, 
                   i1: str,
                   i2: str=None,
                   aniso_file: str = None, 
                   mask: str=None, 

                   independent: bool=False,

                   gpuID: str=None,

                   alpha: float=1,
                   beta: float=0.5,
                   limit_res: str=None,

                   ncpus: int=16, 
                   output_dir: str="isonet_maps",
                   pretrained_model: str=None,

                   reference: str=None,
                   ref_resolution: float=10,

                   epochs: int=50,
                   n_subvolume: int=1000, 
                   cube_size: int=64,
                   predict_crop_size: int=80,
                   batch_size: int=None, 
                   acc_batches: int=2,
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

        # loading i1
        output_base1 = i1.split('/')[-1]
        output_base1 = output_base1.split('.')[:-1]
        output_base1 = "".join(output_base1)

        with mrcfile.open(i1, 'r') as mrc:
            halfmap1 = normalize(mrc.data,percentile=False)
            voxel_size = mrc.voxel_size.x
            if voxel_size == 0:
                voxel_size = 1
        logging.info("voxel_size {}".format(voxel_size))

        # loading i2
        if i2 is not None:
            output_base2 = i2.split('/')[-1]
            output_base2 = output_base2.split('.')[:-1]
            output_base2 = "".join(output_base2)
            with mrcfile.open(i2, 'r') as mrc:
                halfmap2 = normalize(mrc.data,percentile=False)

        # loading mask
        if mask is None:
            mask_vol = np.ones(halfmap1.shape, dtype = np.float32)
            logging.info("No mask is provided. Maybe without mask is better")
        else:
            with mrcfile.open(mask, 'r') as mrc:
                mask_vol = mrc.data

        # loading fsc3d
        if aniso_file is None:
            logging.warning("No fsc3d is provided. Only denosing")
            if (i2 is None) or independent:
                logging.warning("For denoising, please provide half2 and set independent to False")
            fsc3d = np.ones(halfmap1.shape, dtype = np.float32)
        else:
            with mrcfile.open(aniso_file, 'r') as mrc:
                fsc3d = mrc.data

        if limit_res is not None:
            limit_res = float(limit_res)
        
        if reference is not None:
            # TODO change the FSC3D 
            logging.info(f"Incoorporating low resolution information of the reference {reference}\n\
                         until the --ref_resolution {ref_resolution}")
            with mrcfile.open(reference,'r') as mrc:
                ref_map = mrc.data
            from IsoNet.util.FSC import combine_map_F
            halfmap1 = combine_map_F(ref_map,halfmap1,ref_resolution,voxel_size,mask_data=mask_vol)
            if i2 is not None:
                halfmap2 = combine_map_F(ref_map,halfmap2,ref_resolution,voxel_size,mask_data=mask_vol)

            from IsoNet.util.FSC import get_sphere
            sphere = get_sphere(voxel_size/float(ref_resolution)*fsc3d.shape[0], fsc3d.shape[0])
            fsc3d = np.maximum(fsc3d, sphere)
            # with mrcfile.new("tmp_combined_FSC.mrc", overwrite=True) as mrc:
            #     mrc.set_data(fsc3d.astype(np.float32))

        if independent:
            logging.info("processing half1")
            map_refine(halfmap1, mask_vol, fsc3d, alpha = alpha,  voxel_size=voxel_size, output_dir=output_dir, 
                   output_base=output_base1, mixed_precision=False, epochs = epochs,
                   n_subvolume=n_subvolume, cube_size=cube_size, pretrained_model=pretrained_model,
                   batch_size = batch_size, acc_batches = acc_batches,predict_crop_size=predict_crop_size, learning_rate=learning_rate, limit_res= limit_res)
        if (i2 is not None) and independent:
            logging.info("processing half2")
            map_refine(halfmap1, mask_vol, fsc3d, alpha = alpha,  voxel_size=voxel_size, output_dir=output_dir, 
                   output_base=output_base2, mixed_precision=False, epochs = epochs,
                   n_subvolume=n_subvolume, cube_size=cube_size, pretrained_model=pretrained_model,
                   batch_size = batch_size, acc_batches = acc_batches,predict_crop_size=predict_crop_size,learning_rate=learning_rate, limit_res= limit_res)
        if (i2 is not None) and (not independent):
            map_refine_n2n(halfmap1,halfmap2, mask_vol, fsc3d, alpha = alpha,beta=beta,  voxel_size=voxel_size, output_dir=output_dir, 
                   output_base1=output_base1, output_base2=output_base2, mixed_precision=False, epochs = epochs,
                   n_subvolume=n_subvolume, cube_size=cube_size, pretrained_model=pretrained_model,
                   batch_size = batch_size, acc_batches = acc_batches,predict_crop_size=predict_crop_size, learning_rate=learning_rate, limit_res= limit_res)

        if limit_res is not None:
            logging.info("combining")
            self.combine_map(f"{output_dir}/corrected_{output_base1}_filtered.mrc",i1, out_map=f"{output_dir}/corrected_{output_base1}.mrc",threshold_res=limit_res,mask_file= mask)
            if i2 is not None:
                self.combine_map(f"{output_dir}/corrected_{output_base2}_filtered.mrc",i2, out_map=f"{output_dir}/corrected_{output_base2}.mrc",threshold_res=limit_res,mask_file= mask)

        logging.info("Finished")

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

    def whitening(self, 
                    h1: str,
                    o: str = "whitening.mrc",
                    mask: str=None, 
                    high_res: float=3,
                    low_res: float=10,
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
            if voxel_size == 0:
                voxel_size = 1
            logging.info("voxel_size",voxel_size)

        if mask is not None:
            with mrcfile.open(mask,'r') as mrc:
                mask = mrc.data
            input_map_masked = input_map * mask
        else:
            input_map_masked = input_map

        limit_r_low = int(voxel_size * nz / low_res)
        limit_r_high = int(voxel_size * nz / high_res)

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
        
        for i in range(nz//2):
            if i > limit_r_low:
                if i < limit_r_high:
                    F_map[index==i] = F_curve[limit_r_low]/(F_curve[i]+eps)
                else:
                    F_map[index==i] = 1
            else:
                F_map[index==i] = 1

        # apply filter
        F_input = fftn(input_map)
        out = ifftn(F_input*fftshift(F_map))
        out =  np.real(out).astype(np.float32)

        # with mrcfile.new("F.mrc", overwrite=True) as mrc:
        #     mrc.set_data(F_map)

        with mrcfile.new(o, overwrite=True) as mrc:
            mrc.set_data(out)
            mrc.voxel_size = voxel_size

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

    def combine_map(self, 
                    low_map: str, 
                    high_map: str, 
                    out_map:str,
                    threshold_res: float,
                    mask_file: str=None):
        """
        \nCombine the low resolution info (lower than threshold_res) of one map and high resolution info (higher than (lower than threshold_res) of another map to produce a chimeric map. 
        """
        import numpy as np
        import mrcfile
        with mrcfile.open(low_map,'r') as mrc:
            low_data = mrc.data
            voxel_size = mrc.voxel_size.x
            if voxel_size == 0:
                voxel_size = 1
            logging.info(f"voxel_size {voxel_size}")

        with mrcfile.open(high_map,'r') as mrc:
            high_data = mrc.data
        
        if mask_file is not None:
            with mrcfile.open(mask_file,'r') as mrc:
                mask = mrc.data
        else:
            mask = np.ones_like(high_data)
            
        from IsoNet.util.FSC import combine_map_F
        out_data = combine_map_F(low_data, high_data, threshold_res, voxel_size, mask_data=mask)

        with mrcfile.new(out_map,overwrite=True) as mrc:
            mrc.set_data(out_data)
            mrc.voxel_size = voxel_size
        
    def fsc3d(self, 
                   h: str,
                   h2: str, 
                   mask: str=None, 
                   o: str="FSC3D.mrc",
                   ncpus: int=16, 
                   limit_res: float=None, 
                   cone_sampling_angle: float=10,
                   keep_highres: bool = False
                   ):

        """
        \n3D Fourier shell correlation\n
        IsoNet.py map_refine half1.mrc half2.mrc mask.mrc [--gpuID] [--ncpus] [--output_dir] [--fsc_file]...
        :param h: Input name of half1
        :param h2: Input name of half2
        :param mask: Filename of a user-provided mask
        :param ncpus: Number of cpu.
        :param limit_res: The resolution limit for recovery, default is the resolution of the map.
        :param fsc_file: 3DFSC file if not set, isonet will generate one.
        :param cone_sampling_angle: Angle for 3D fsc sampling for IsoNet generated 3DFSC. IsoNet default is 10 degrees, the default for official 3DFSC is 20 degrees
        :param keep_highres: Set high frequency region to 1 instead of 0. This should be False
        """
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   

        from IsoNet.preprocessing.img_processing import normalize
        import numpy as np
        from multiprocessing import cpu_count
        import mrcfile

        from IsoNet.util.FSC import get_FSC_map, ThreeD_FSC, recommended_resolution

        cpu_system = cpu_count()
        if cpu_system < ncpus:
            logging.info("requested number of cpus is more than the number of the cpu cores in the system")
            logging.info(f"setting ncpus to {cpu_system}")
            ncpus = cpu_system

        with mrcfile.open(h, 'r') as mrc:
            half1 = normalize(mrc.data,percentile=False)
            voxel_size = mrc.voxel_size.x
            if voxel_size == 0:
                voxel_size = 1

        with mrcfile.open(h2, 'r') as mrc:
            half2 = normalize(mrc.data,percentile=False)


        if mask is None:
            mask_vol = np.ones(half1.shape, dtype = np.float32)
            logging.warning("No mask is provided, please consider providing a soft mask")
        else:
            with mrcfile.open(mask, 'r') as mrc:
                mask_vol = mrc.data

        FSC_map = get_FSC_map([half1, half2], mask_vol)
        if limit_res is None:
            limit_res = recommended_resolution(FSC_map, voxel_size, threshold=0.143)
            logging.info("Global resolution at FSC={} is {}".format(0.143, limit_res))

        limit_r = int( (2.*voxel_size) / limit_res * (half1.shape[0]/2.) + 1)
        logging.info("Limit resolution to {} for IsoNet 3D FSC calculation. You can also tune this paramerter with --limit_res .".format(limit_res))

        logging.info("calculating fast 3DFSC, this will take few minutes")
        fsc3d = ThreeD_FSC(FSC_map, limit_r,angle=float(cone_sampling_angle), n_processes=ncpus)
        if keep_highres:
            from IsoNet.util.FSC import get_sphere
            fsc3d = np.maximum(1-get_sphere(limit_r-2,fsc3d.shape[0]),fsc3d)
        with mrcfile.new(o, overwrite=True) as mrc:
            mrc.set_data(fsc3d.astype(np.float32))
        logging.info("voxel_size {}".format(voxel_size))
  
    def fsd3d(self, 
                   star_file: str, 
                   map_dim: int,
                   apix: float=1.0,
                   o: str="FSD3D.mrc",
                   low_res: float=100,
                   high_res: float=1, 
                   number_subset: float=10000,
                   grid_size: float=64,
                   sym: str = "c1"
                   ):

        """
        \nFourier shell density, reimpliment from cryoEF, relies on relion star file and also relion installation.\n
        IsoNet.py map_refine half1.mrc half2.mrc mask.mrc [--gpuID] [--ncpus] [--output_dir] [--fsc_file]...
        :param h: Input name of half1
        :param h2: Input name of half2
        :param mask: Filename of a user-provided mask
        :param ncpus: Number of cpu.
        :param limit_res: The resolution limit for recovery, default is the resolution of the map.
        :param fsc_file: 3DFSC file if not set, isonet will generate one.
        :param cone_sampling_angle: Angle for 3D fsc sampling for IsoNet generated 3DFSC. IsoNet default is 10 degrees, the default for official 3DFSC is 20 degrees.
        """
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   
        import numpy as np
        import mrcfile
        from subprocess import check_output

        s = f"thetacol=`grep _rlnAngleTilt {star_file} | awk '{{print $2}}' | sed 's/#//'`;\
        phicol=`grep _rlnAngleRot {star_file} | awk '{{print $2}}' | sed 's/#//'`;\
        cat {star_file} | grep @ | awk -v thetacolvar=${{thetacol}} -v phicolvar=${{phicol}} '{{if (NF>2) print $thetacolvar, $phicolvar}}' > theta_phi_angles.dat"
        check_output(s, shell=True)

        coordinates = np.loadtxt("theta_phi_angles.dat",dtype=np.float32)
        index = np.random.choice(coordinates.shape[0], number_subset)#, replace=True)
        coordinates = coordinates[index]

        def generate_grid(coordinates, grid_size):
            # Create a 3D grid for calculation
            x = np.linspace(-1, 1, grid_size).astype(np.float32)
            y = np.linspace(-1, 1, grid_size).astype(np.float32)
            z = np.linspace(-1, 1, grid_size).astype(np.float32)
            x, y, z = np.meshgrid(x, y, z)

            # Initialize the matrix to store the sum of circles
            circle_matrix = np.zeros_like(x)

            # Define the distance threshold
            threshold = 5**0.5/grid_size
            coord = np.radians(coordinates)
            cos_coord = np.cos(coord)
            sin_coord = np.sin(coord)

            # This need to think carefully
            surface_points = np.stack([sin_coord[:,0]*cos_coord[:,1],sin_coord[:,0]*sin_coord[:,1],cos_coord[:,0]])
            #surface_points = np.stack([cos_coord[:,0],sin_coord[:,0]*sin_coord[:,1],sin_coord[:,0]*cos_coord[:,1]])
            #print(surface_points)
            pixels = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

            distances = np.abs(np.matmul(pixels,surface_points))
            circle = np.where(distances <= threshold, 1, 0)
            circle = np.sum(circle, axis=1)
            circle_matrix =  circle.reshape(grid_size, grid_size, grid_size)
            circle_matrix =  np.transpose(circle_matrix,[2,0,1])
            return circle_matrix
        
        out_mat = generate_grid(coordinates[:1000],grid_size)
        for i in range(1,10):
            out_mat += generate_grid(coordinates[i*1000:(i+1)*1000],grid_size)
        out_mat = out_mat / number_subset

        with mrcfile.new(o, overwrite=True) as mrc:
            mrc.set_data(out_mat.astype(np.float32))

        s = f"relion_image_handler --i {o} --o {o} --sym {sym}"
        check_output(s,shell=True)

        with mrcfile.open(o,'r') as mrc:
            input_map = mrc.data
            nz,ny,nx = input_map.shape
            voxel_size = mrc.voxel_size.x
            if voxel_size == 0:
                voxel_size = 1

        #apix_small = apix * map_dim / grid_size
        r = np.arange(nz)-nz//2
        limit_r_low = int(apix * nz / low_res)
        limit_r_high = int(apix * nz / high_res)


        [Z,Y,X] = np.meshgrid(r,r,r)
        index = np.round(np.sqrt(Z**2+Y**2+X**2))

        F_map = np.zeros_like(input_map)
        eps = 1e-4

        for i in range(nz//2):
            if i > limit_r_low:
                if i < limit_r_high:
                    F_map[index==i] = 1.1/np.max(input_map[index==i])
                else:
                    F_map[index==i] = 0
            else:
                F_map[index==i] = 1

        out_map = F_map*input_map
        for i in range(nz//2):
            if i > limit_r_low:
                if i > limit_r_high:
                    out_map[index==i] = 0
            else:
                out_map[index==i] = 1

        out_map[out_map>1] = 1
        import skimage
        out_map = skimage.transform.resize(out_map, [map_dim,map_dim,map_dim])
        with mrcfile.new(o, overwrite=True) as mrc:
            mrc.set_data(out_map)
            mrc.voxel_size = voxel_size

    def psf(self, diameter, wedge_angle):
        from IsoNet.bin.wedge import get_F_wedge
        get_F_wedge(diameter, wedge_angle)

    def reconstruct_psf(self,size=128,tilt_file=None,w=8, output="wedge.mrc",between_tilts=False):
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
        
        label += ['rlnNumberSubtomo']
        data.append([n_subtomo]*len(tomograms_files))

        data_length = len(data[0])
        data = list(map(list, zip(*data)))
        df = pd.DataFrame(data = data, columns = label)
        df.insert(0, 'rlnIndex', np.arange(data_length)+1)
        starfile.write(df,output_star)

    def extract(self,  star, 
                subtomo_folder="subtomos", 
                between_tilts=False, 
                cube_size=128, 
                crop_size=None):

        import starfile
        import mrcfile
        import pandas as pd
        import numpy as np
        from IsoNet.preprocessing.cubes import extract_subvolume, create_cube_seeds
        from IsoNet.preprocessing.img_processing import normalize

        if crop_size is None:
            crop_size = cube_size + 16

        # TODO read number of subtomo
        n_subtomo_per_tomo = 100

        df = starfile.read(star)
        os.makedirs(subtomo_folder, exist_ok=True)

        particle_list = []
        for index, row in df.iterrows():
            tomo_index = row["rlnIndex"]
            tomo_folder = f"TS_{tomo_index:05d}"
            even_folder = os.path.join(subtomo_folder, tomo_folder, 'subtomo0')
            os.makedirs(even_folder, exist_ok=True)
            with mrcfile.open(row["rlnTomoName"]) as mrc:
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

            #TODO wedge do the relion similar wedge and CTF
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
        


    def angular_whiten(self, in_name,out_name,low_res, high_res):
        """
        \nWhitening across different orientations. To prevent over representation at some of the directions
        """
        import mrcfile

        with mrcfile.open(in_name) as mrc:
            in_map = mrc.data.copy()
            voxel_size = mrc.voxel_size.x

        from IsoNet.util.FSC import angular_whitening
        out_map = angular_whitening(in_map,voxel_size,low_res,high_res)
        with mrcfile.new(out_name, overwrite=True) as mrc:
            mrc.set_data(out_map)
            mrc.voxel_size = tuple([voxel_size]*3) 

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
