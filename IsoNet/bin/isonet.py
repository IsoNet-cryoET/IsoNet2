#!/usr/bin/env python3
import fire
import logging
import os, sys, traceback
from IsoNet.utils.dict2attr import check_parse
from fire import core
import starfile
import mrcfile
import pandas as pd
import numpy as np
from IsoNet.utils.utils import read_and_norm, read_mrc
from IsoNet.utils.utils import mkfolder
from IsoNet.utils.utils import parse_cpu, parse_gpu

types ={
    0: ["rlnTomoReconstructedTomogramHalf1", "rlnTomoReconstructedTomogramHalf2"],
    1: ["rlnTomoName"],
    2: ["rlnDeconvTomoName"],
    3: ["rlnDenoisedTomoName"],
    4: ["rlnCorrectedTomoName"],
}
class ISONET:
    """
    ISONET: Train on tomograms and restore missing-wedge\n
    for detail discription, run one of the following commands:

    IsoNet.py fsc3d -h
    IsoNet.py refine -h
    """
    def denoise(self, 
                   star_file: str,
                   gpuID: str=None,

                   ncpus: int=16, 
                   output_dir: str="isonet_maps",
                   pretrained_model: str=None,

                   epochs: int=50,
                   cube_size: int=128,
                   crop_size: int=None,
                   batch_size: int=None, 
                   acc_batches: int=1,
                   learning_rate: float=3e-4
                   ):
        
        if crop_size is None:
            crop_size = cube_size + 16

        ngpus, gpuID, gpuID_list=parse_gpu(gpuID)

        if batch_size is None:
            if ngpus == 1:
                batch_size = 4
            else:
                batch_size = 2 * len(gpuID_list)

        # extract subtomograms
        data_list = self.extract(star_file, subtomo_folder=output_dir+"/subtomos", 
                between_tilts=False, random=False, cube_size=cube_size, crop_size=cube_size, 
                )
        # noise2noise training
        from IsoNet.bin.refine import run_training
        run_training([data_list[0] + data_list[1],data_list[1] + data_list[0]], epochs = epochs, mixed_precision = False,
                output_dir = output_dir, outmodel_path=f"{output_dir}/n2n.pt", pretrained_model=pretrained_model,
                ncpus=16, batch_size = batch_size, acc_batches=acc_batches, learning_rate= learning_rate)
        # noise2noise prediction
        self.predict(star_file, model=output_dir+"/n2n.pt", output_dir=output_dir, 
                    gpuID = gpuID, cube_size=cube_size,
                    crop_size=crop_size,tomo_type=0, out_type = 5,
                    batch_size = batch_size,normalize_percentile=False,
                    log_level="info", tomo_idx=None)

    def refine(self, 
                   star_file: str,
                   gpuID: str=None,
                   n2n: bool=False,
                   limit_res: str=None,
                   ncpus: int=16, 
                   output_dir: str="isonet_maps",
                   pretrained_model: str=None,


                   use_denoised=False, 
                   use_deconv=False,
                   epochs: int=10,
                   n_subvolume: int=1000, 
                   cube_size: int=64,
                   predict_crop_size: int=80,
                   batch_size: int=None, 
                   acc_batches: int=1,
                   learning_rate: float=3e-4
                   ):

        """
        \n\n
        IsoNet.py map_refine half.mrc FSC3D.mrc --mask mask.mrc --limit_res 3.5 [--gpuID] [--ncpus] [--output_dir] [--fsc_file]...
        :param i: Input half map 1
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

        from IsoNet.preprocessing.prepare import get_cubes_list
        from IsoNet.bin.refine import run_training
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   
        
        ngpus, gpuID, gpuID_list = parse_gpu(gpuID)
        ncpus = parse_cpu(ncpus)

        if batch_size is None:
            if ngpus == 1:
                batch_size = 4
            else:
                batch_size = 2 * len(gpuID_list)
        
        mkfolder(output_dir,remove=False)
        data_list = self.extract(star_file, subtomo_folder=f'{output_dir}/iter000', 
                between_tilts=False, cube_size=cube_size, crop_size=cube_size, 
                use_denoised=use_denoised, use_deconv=use_deconv, out_star = f"{output_dir}/particles_it000.star")
        prefix="train"   
        iterations = 1
       
        for i in range(iterations):
            particle_star_file = f"{output_dir}/particles_it{i:03d}.star"
            next_particle_star_file = f"{output_dir}/particles_it{i+1:03d}.star"

            star = starfile.read(particle_star_file)
            data_dir = output_dir+"/tmpdata"
            mkfolder(data_dir)
            
            if i==0:
                get_cubes_list(star, data_dir, ncpus=ncpus, start_over=True)
            else:
                get_cubes_list(star, data_dir, ncpus=ncpus, start_over=False)
                pretrained_model = f"{output_dir}/model_it{i:03d}.pt"
            model_path = f"{output_dir}/model_it{i+1:03d}.pt"
            path_all=[]
            for d in  [prefix+"_x", prefix+"_y"]:
                p = '{}/tmpdata/{}/'.format(output_dir, d)
                path_all.append(sorted([p+f for f in os.listdir(p)]))
            
            run_training(data_list=path_all,output_dir=output_dir, 
                        mixed_precision=False, epochs = epochs,
                        pretrained_model=pretrained_model,outmodel_path = model_path,
                    batch_size = batch_size, ncpus=ncpus, acc_batches = acc_batches,learning_rate=learning_rate)
            print(f'{output_dir}/iter{i}')
            star = self.predict_subtomos(particle_star_file, model_path, 
                                         f'{output_dir}/iter{i+1:03d}', gpuID=gpuID)
            starfile.write(star,next_particle_star_file) 

        logging.info("Finished")

    def predict_subtomos(self, subtomo_star, model, out_folder, gpuID='0,1,2,3'):
        
        ngpus, gpuID, gpuID_list = parse_gpu(gpuID)

        mkfolder(out_folder)
        from IsoNet.models.network import Net
        network = Net(filter_base = 64,unet_depth=3, add_last=True)
        network.load(model)
        from IsoNet.utils.utils import write_mrc
        import starfile
        star = starfile.read(subtomo_star)
        star["rlnCorrectedParticle"] = None
        subtomo_name = list(star['rlnParticleName'])
        outData = network.predict_subtomos(subtomo_name,out_folder)
        for i,item in enumerate(outData):
            #root_name = subtomo_name[i].split("/")[-1]
            file_name = f"{out_folder}/subvolume_{i:0>6d}.mrc"
            write_mrc(file_name, item)
            star.at[i,"rlnCorrectedParticle"]=file_name
        #starfile.write(star,subtomo_star)
        return star


    def predict(self, star_file: str, model: str, output_dir: str='./corrected_tomos', 
                gpuID: str = None, cube_size:int=96, out_type = 4,
                crop_size:int=128, tomo_type = 1, batch_size:int=None,
                normalize_percentile: bool=True,log_level: str="info", tomo_idx=None):
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


        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])

        ngpus, gpuID, gpuID_list = parse_gpu(gpuID)

        from IsoNet.models.network import Net
        from IsoNet.utils.utils import write_mrc
        import starfile
        import numpy as np
        from IsoNet.preprocessing.img_processing import normalize

        mkfolder(output_dir, remove=False)
        network = Net(filter_base = 64,unet_depth=3, add_last=True)
        network.load(model)

        star = starfile.read(star_file)
        star[types[out_type][0]] = None
        for index, tomo_row in star.iterrows():
            print(tomo_row)
            tomo = read_and_norm(tomo_row[types[tomo_type][0]])
            tomo2 = tomo
            if tomo_type < 1:
                tomo2 = read_and_norm(tomo_row[types[tomo_type][1]])

            tomo = normalize((tomo+tomo2)*-1,percentile=False)
            outData = network.predict_map(tomo, output_dir,cube_size=cube_size, crop_size=crop_size).astype(np.float32) #train based on init model and save new one as model_iter{num_iter}.h5
            file_base_name = os.path.basename(tomo_row[types[tomo_type][0]])
            file_name, file_extension = os.path.splitext(file_base_name)
            out_file_name = f"{output_dir}/corrected_{file_name}.mrc"
            write_mrc(out_file_name, outData*-1)        
            star.at[index, types[out_type][0]] = out_file_name
        starfile.write(star,star_file)

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
            from IsoNet.utils.geometry import draw_a_line
            out_mat = np.zeros([size,size], dtype = np.float32)
            for angle in tilt:
                out_mat = draw_a_line(out_mat, -angle, w)
            out_mat = np.repeat(out_mat[:, np.newaxis, :], size, axis=1)
        else:
            from IsoNet.utils.missing_wedge import get_F_wedge
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
        tomo_path = os.path.join(folder_name, tomo_list[0])
        data = []

        if os.path.isdir(tomo_path):
            for i,tomo_name in enumerate(tomo_list):
                if apix is None:
                    voxel_size = 1
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

                if len(tomo_file) == 1:
                    label = ['rlnIndex','rlnTomoName','rlnTiltFile','rlnPixelSize','rlnDefocus']
                    data.append([i, tomo_file[0], tilt_file, voxel_size, defocus])   
                else:
                    label = ['rlnIndex','rlnTomoReconstructedTomogramHalf1','rlnTomoReconstructedTomogramHalf2','rlnTiltFile','rlnPixelSize','rlnDefocus']
                    data.append([i, tomo_file[0], tomo_file[1], tilt_file, voxel_size, defocus]) 

        else:
            for i,tomo_name in enumerate(tomo_list):
                if tomo_path[-4:]=='.mrc' or tomo_path[-4:]=='.rec':
                    tomo_path = os.path.join(folder_name,tomo_name)
                    if apix is None:
                        volume, voxel_size = read_mrc(tomo_path)
                        if voxel_size == 0:
                            voxel_size = 1.0
                    else:
                        voxel_size = apix
                    label = ['rlnIndex','rlnTomoName','rlnPixelSize','rlnNumberSubtomo', 'rlnDefocus']
                    data.append([i, tomo_path, voxel_size,number_subtomos, defocus])  
        df = pd.DataFrame(data, columns = label)
        starfile.write(df,output_star)

    def extract_both(self,  star, subtomo_folder="subtomos", out_star = "subtomos.star", 
                between_tilts=False, cube_size=96, crop_size=128, 
                random = False):
        self.extract(star, subtomo_folder=subtomo_folder, out_star = out_star, 
                between_tilts=between_tilts, cube_size=cube_size, crop_size=crop_size, 
                tomo_type=0, random = random)

    def extract(self,  star, subtomo_folder="subtomos", out_star = "subtomos.star", 
                between_tilts=False, cube_size=96, crop_size=128, 
                use_denoised=False, use_deconv=False, random = True, apply_wedge = True):
        if use_deconv:
            tomo_type = 2
        elif use_denoised:
            tomo_type = 3
        else:
            tomo_type = 0

        if crop_size is None:
            crop_size = cube_size

        from IsoNet.utils.utils import mkfolder
        from IsoNet.preprocessing.cubes import extract_with_overlap
        from IsoNet.preprocessing.cubes import extract_subvolume, create_cube_seeds

        df = starfile.read(star)
        mkfolder(subtomo_folder, remove=True)
        particle_list = []
        for index, row in df.iterrows():
            tomo_index = row["rlnIndex"]
            tomo_folder = f"TS_{tomo_index:05d}"
            
            print(tomo_folder)
            mkfolder(os.path.join(subtomo_folder, tomo_folder))

            # wedge 
            wedge_path = os.path.join(subtomo_folder, tomo_folder, 'wedge.mrc')
            if "rlnTiltFile" in list(df.columns):
                self.psf(size=crop_size, tilt_file=row["rlnTiltFile"], output=wedge_path, between_tilts=between_tilts)
            else:
                print(wedge_path)
                self.psf(size=crop_size, output=wedge_path, between_tilts=between_tilts)
            if apply_wedge:
                wedge, vs = read_mrc(wedge_path)
            else:
                wedge = None

            # first half
            even_folder = os.path.join(subtomo_folder, tomo_folder, 'subtomo0')
            mkfolder(even_folder)
            tomo_name = row[types[tomo_type][0]]
            tomo = read_and_norm(tomo_name)

            # TODO mask
            mask = np.ones_like(tomo)

            if random:
                n_subtomo_per_tomo = row["rlnNumberSubtomo"]
                seeds=create_cube_seeds(tomo, n_subtomo_per_tomo, crop_size, mask)
                subtomos_names = extract_subvolume(tomo, seeds, crop_size, even_folder, prefix='', wedge=wedge)
            else:
                subtomos_names = extract_with_overlap(tomo, crop_size, cube_size, even_folder, prefix='', wedge=None)
            
            # second half
            if tomo_type < 1:
                odd_folder = os.path.join(subtomo_folder, tomo_folder, 'subtomo1')
                tomo_name = row[types[tomo_type][1]]
                mkfolder(odd_folder)
                tomo = read_and_norm(tomo_name)
                if random:
                    subtomos_names = extract_subvolume(tomo, seeds, crop_size, odd_folder)
                else:
                    subtomos_names = extract_with_overlap(tomo, crop_size, cube_size, odd_folder, prefix='', wedge=None)                

            for i in range(len(subtomos_names)):
                im_name1 = '{}/subvolume{}_{:0>6d}.mrc'.format(even_folder, '', i)
                if tomo_type < 1:#"rlnTomogram2Name" in list(df.columns):
                    im_name2 = '{}/subvolume{}_{:0>6d}.mrc'.format(odd_folder, '', i)
                    particle_list.append([im_name1,im_name2,wedge_path])
                    label = ["rlnParticleName","rlnParticle2Name","rlnWedgeName"]                    
                else:
                    particle_list.append([im_name1,wedge_path])
                    label = ["rlnParticleName","rlnWedgeName"]

        df = pd.DataFrame(particle_list, columns = label)
        starfile.write(df, out_star)
        if tomo_type < 1:
            extract_list = [df['rlnParticleName'].tolist(),df['rlnParticle2Name'].tolist()]
        else:
            extract_list = [df['rlnParticleName'].tolist()]   

        return  extract_list

    def deconv(self, star_file: str,
        deconv_folder:str="./deconv",
        do_half: bool = False,
        voltage: float=300.0,
        cs: float=2.7,
        snrfalloff: float=None,
        deconvstrength: float=None,
        highpassnyquist: float=0.02,
        chunk_size: int=None,
        overlap_rate: float= 0.25,
        ncpu:int=4,
        tomo_idx: str=None):
        """
        \nCTF deconvolution for the tomograms.\n
        isonet.py deconv star_file [--deconv_folder] [--snrfalloff] [--deconvstrength] [--highpassnyquist] [--overlap_rate] [--ncpu] [--tomo_idx]
        This step is recommended because it enhances low resolution information for a better contrast. No need to do deconvolution for phase plate data.
        :param deconv_folder: (./deconv) Folder created to save deconvoluted tomograms.
        :param star_file: (None) Star file for tomograms.
        :param voltage: (300.0) Acceleration voltage in kV.
        :param cs: (2.7) Spherical aberration in mm.
        :param snrfalloff: (1.0) SNR fall rate with the frequency. High values means losing more high frequency.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 1.0 will be used.
        :param deconvstrength: (1.0) Strength of the deconvolution.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 1.0 will be used.
        :param highpassnyquist: (0.02) Highpass filter for at very low frequency. We suggest to keep this default value.
        :param chunk_size: (None) When your computer has enough memory, please keep the chunk_size as the default value: None . Otherwise, you can let the program crop the tomogram into multiple chunks for multiprocessing and assembly them into one. The chunk_size defines the size of individual chunk. This option may induce artifacts along edges of chunks. When that happen, you may use larger overlap_rate.
        :param overlap_rate: (None) The overlapping rate for adjecent chunks.
        :param ncpu: (4) Number of cpus to use.
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
        """
        from IsoNet.utils.deconvolution import deconv_one
        import starfile
        from IsoNet.utils.utils import read_mrc, write_mrc
        from IsoNet.utils.utils import idx2list
        tomo_idx = idx2list(tomo_idx)

        if not os.path.isdir(deconv_folder):
            os.mkdir(deconv_folder)

        star = starfile.read(star_file)
        new_star = star.copy()

        if not 'rlnSnrFalloff' in new_star.columns:
            new_star['rlnSnrFalloff'] = 1
            new_star['rlnDeconvStrength'] = 1
            new_star['rlnDeconvTomoName'] = 'None'
            new_star['rlnDeconvTomo2Name'] = 'None'

        starfile.write(new_star,"test.star") 
        for i, it in star.iterrows():
            if tomo_idx is None or str(it.rlnIndex) in tomo_idx:
                if snrfalloff is not None:
                    new_star.loc[i,'rlnSnrFalloff']=snrfalloff
                if deconvstrength is not None:
                    new_star.loc[i,'rlnDeconvStrength']=deconvstrength
                
                tomo_file = it['rlnTomogramName']
                base_name = os.path.basename(tomo_file)
                deconv_tomo_name = '{}/{}'.format(deconv_folder,base_name)
                print(tomo_file,base_name)
                if 'rlnTomogram2Name' in new_star.columns and not do_half:
                    tomo_file2 = it['rlnTomogram2Name']
                    data2,voxel_size = read_mrc(tomo_file2)
                    data1,voxel_size  = read_mrc(tomo_file)
                    data = (data2 + data1)/2
                    write_mrc('tmp_full.mrc', data, voxel_size)
                    tomo_file = 'tmp_full.mrc'

                deconv_one(tomo_file,deconv_tomo_name,voltage=voltage,cs=cs,
                           defocus=new_star.loc[i,'rlnDefocus']/10000.0, 
                           pixel_size=new_star.loc[i,'rlnPixelSize'],
                           snrfalloff=new_star.loc[i,'rlnSnrFalloff'],
                            deconvstrength=new_star.loc[i,'rlnDeconvStrength'],highpassnyquist=highpassnyquist,
                            chunk_size=chunk_size,overlap_rate=overlap_rate,ncpu=ncpu)
                new_star.loc[i,'rlnDeconvTomoName']=deconv_tomo_name

                if 'rlnTomogram2Name' in new_star.columns and do_half:
                    tomo_file = it['rlnTomogram2Name']
                    base_name = os.path.basename(tomo_file)
                    deconv_tomo_name = '{}/{}'.format(deconv_folder,base_name)
                    deconv_one(tomo_file,deconv_tomo_name,voltage=voltage,cs=cs,
                           defocus=new_star.loc[i,'rlnDefocus']/10000.0, 
                           pixel_size=new_star.loc[i,'rlnPixelSize'],
                           snrfalloff=new_star.loc[i,'rlnSnrFalloff'],
                            deconvstrength=new_star.loc[i,'rlnDeconvStrength'],highpassnyquist=highpassnyquist,
                            chunk_size=chunk_size,overlap_rate=overlap_rate,ncpu=ncpu)
                    new_star.loc[i,'rlnDeconvTomo2Name']=deconv_tomo_name
        starfile.write(new_star,star_file) 
        logging.info('\n######Isonet done ctf deconvolve######\n')

        # except Exception:
        #     error_text = traceback.format_exc()
        #     f =open('log.txt','a+')
        #     f.write(error_text)
        #     f.close()
        #     logging.error(error_text)

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
