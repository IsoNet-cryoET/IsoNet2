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
from IsoNet.utils.fileio import read_mrc
from IsoNet.utils.utils import mkfolder
from IsoNet.utils.utils import parse_cpu, parse_gpu

# types ={
#     0: ["rlnTomoReconstructedTomogramHalf1", "rlnTomoReconstructedTomogramHalf2"],
#     1: ["rlnTomoName"],
#     2: ["rlnDeconvTomoName"],
#     3: ["rlnDenoisedTomoName"],
#     4: ["rlnCorrectedTomoName"],
# }
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
                     pixel_size = 10.0, 
                     defocus_folder="None",
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
        
        if full_folder == "None":
            count_folder = even_folder
        else:
            count_folder = full_folder
        num_tomo = len(os.listdir(count_folder))
        print("number of tomograms", num_tomo)


        data = []
        label = []
        def add_param(folder_name, param_name, default_val="None"):
            if folder_name != "None":
                # TODO check file extension
                files = sorted(os.listdir(full_folder))
                files = [f"{full_folder}/{item}" for item in files]
                assert len(files) == num_tomo
                data.append(files)
            else:
                data.append([default_val]*num_tomo)
            label.append(param_name)

        add_param(full_folder, 'rlnTomoName')
        add_param(even_folder, 'rlnTomoReconstructedTomogramHalf1')
        add_param(odd_folder, 'rlnTomoReconstructedTomogramHalf2')

        # deconv parameters
        add_param("None", 'rlnPixelSize',pixel_size)
        add_param(defocus_folder, 'rlnDefocus',0)
        add_param("None", "rlnSnrFalloff",1)
        add_param("None", "rlnDeconvStrength",1)
        add_param("None", "rlnDeconvTomoName","None")

        # mask parameters
        add_param("None", "rlnMaskBoundary","None")
        add_param("None", "rlnMaskDensityPercentage",50)
        add_param("None", "rlnMaskStdPercentage",50)
        add_param(mask_folder, "rlnMaskName","None")

        #add_param(tilt_file_folder, 'rlnTiltFile')
        #add_param(coordinate_folder, 'rlnBoxFile')
        
        add_param("None", 'rlnNumberSubtomo',number_subtomos)
        
        #TODO defocus file folder 
        #from IsoNet.utils.fileio import read_defocus_file
        #TODO decide the defocus file format from CTFFIND and GCTF
        
        data = list(map(list, zip(*data)))
        df = pd.DataFrame(data = data, columns = label)
        df.insert(0, 'rlnIndex', np.arange(num_tomo)+1)
        starfile.write(df,output_star)

    def deconv(self, star_file: str,
        deconv_folder:str="./deconv",
        input_column: str="rlnTomoName",
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

        for i, it in star.iterrows():
            if tomo_idx is None or str(it.rlnIndex) in tomo_idx:
                if snrfalloff is not None:
                    new_star.loc[i,'rlnSnrFalloff'] = snrfalloff
                if deconvstrength is not None:
                    new_star.loc[i,'rlnDeconvStrength'] = deconvstrength
                
                tomo_file = it[input_column]
                deconv_tomo_name = '{}/{}'.format(deconv_folder,os.path.basename(tomo_file))

                deconv_one(tomo_file,deconv_tomo_name,voltage=voltage,cs=cs,
                           defocus=new_star.loc[i,'rlnDefocus']/10000.0, 
                           pixel_size=new_star.loc[i,'rlnPixelSize'],
                           snrfalloff=new_star.loc[i,'rlnSnrFalloff'],
                            deconvstrength=new_star.loc[i,'rlnDeconvStrength'],highpassnyquist=highpassnyquist,
                            chunk_size=chunk_size,overlap_rate=overlap_rate,ncpu=ncpu)
                new_star.loc[i,'rlnDeconvTomoName']=deconv_tomo_name

        starfile.write(new_star,star_file) 
        logging.info('\n######Isonet done ctf deconvolve######\n')

        # except Exception:
        #     error_text = traceback.format_exc()
        #     f =open('log.txt','a+')
        #     f.write(error_text)
        #     f.close()
        #     logging.error(error_text)
            # with mrcfile.new(f"{output_dir}/corrected_{file_name}.mrc") as mrc:
            #     mrc.set_data(outData)

    def make_mask(self,star_file,
                input_column: str = "rlnDeconvTomoName",
                mask_folder: str = 'mask',
                patch_size: int=4,
                mask_boundary: str=None,
                density_percentage: int=None,
                std_percentage: int=None,
                use_deconv_tomo:bool=True,
                z_crop:float=None,
                tomo_idx=None):
        """
        \ngenerate a mask that include sample area and exclude "empty" area of the tomogram. The masks do not need to be precise. In general, the number of subtomograms (a value in star file) should be lesser if you masked out larger area. \n
        isonet.py make_mask star_file [--mask_folder] [--patch_size] [--density_percentage] [--std_percentage] [--use_deconv_tomo] [--tomo_idx]
        :param star_file: path to the tomogram or tomogram folder
        :param mask_folder: path and name of the mask to save as
        :param patch_size: (4) The size of the box from which the max-filter and std-filter are calculated.
        :param density_percentage: (50) The approximate percentage of pixels to keep based on their local pixel density.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 50 will be used.
        :param std_percentage: (50) The approximate percentage of pixels to keep based on their local standard deviation.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 50 will be used.
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
        :param z_crop: If exclude the top and bottom regions of tomograms along z axis. For example, "--z_crop 0.2" will mask out the top 20% and bottom 20% region along z axis.
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
        """
        from IsoNet.bin.make_mask import make_mask
        from IsoNet.utils.utils import idx2list

        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        logging.info('\n######Isonet starts making mask######\n')

        try:
            if not os.path.isdir(mask_folder):
                os.mkdir(mask_folder)
                
            star = starfile.read(star_file)
            new_star = star.copy()

            if not 'rlnMaskDensityPercentage' in new_star.columns:
                new_star['rlnMaskDensityPercentage'] = 50
                new_star['rlnMaskStdPercentage'] = 50
                new_star['rlnMaskName'] = 'None'

            if not input_column in new_star.columns:
                input_column = "rlnTomoName",

            tomo_idx = idx2list(tomo_idx)
            for i, it in star.iterrows():
                if tomo_idx is None or str(it.rlnIndex) in tomo_idx:
                    if density_percentage is not None:
                        new_star.loc[i,'rlnMaskDensityPercentage'] = density_percentage
                    if std_percentage is not None:
                        new_star.loc[i,'rlnMaskStdPercentage'] = std_percentage

                    tomo_file = it[input_column]
                    if tomo_file == "None":
                        logging.info(f"using rlnTomoName instead of {input_column}")
                        tomo_file = it["rlnTomoName"]
                    tomo_root_name = os.path.splitext(os.path.basename(tomo_file))[0]

                    if os.path.isfile(tomo_file):
                        logging.info('make_mask: {}| dir_to_save: {}| percentage: {}| window_scale: {}'.format(tomo_file,
                        mask_folder, it.rlnMaskDensityPercentage, patch_size))
                        
                        #if mask_boundary is None:
                        if "rlnMaskBoundary" in star.columns.tolist() and it.rlnMaskBoundary not in [None, "None"]:
                            mask_boundary = it.rlnMaskBoundary 
                        else:
                            mask_boundary = None
                              
                        mask_out_name = '{}/{}_mask.mrc'.format(mask_folder,tomo_root_name)
                        make_mask(tomo_file,
                                mask_out_name,
                                mask_boundary=mask_boundary,
                                side=patch_size,
                                density_percentage=it.rlnMaskDensityPercentage,
                                std_percentage=it.rlnMaskStdPercentage,
                                surface = z_crop)

                        new_star.loc[i,'rlnMaskName']=mask_out_name
            starfile.write(new_star,star_file)
            logging.info('\n######Isonet done making mask######\n')

        except Exception:
            error_text = traceback.format_exc()
            f =open('log.txt','a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)

    def extract(self,  
                star_file: str,
                input_column: str = "rlnDeconvTomoName",
                subtomo_folder="subtomos", 
                subtomo_star = "subtomos.star", 
                #between_tilts=False, 
                cube_size = 64,
                crop_size = None, 
                tomo_idx = None,
                uniform_extract=False):
        
        from IsoNet.utils.utils import mkfolder
        from IsoNet.utils.processing import normalize
        from IsoNet.preprocessing.cubes import extract_with_overlap
        from IsoNet.preprocessing.cubes import extract_subvolume, create_cube_seeds

        if crop_size is None:
            crop_size = cube_size + 16

        tomo_star = starfile.read(star_file)
        tomo_columns = tomo_star.columns.to_list()
        mkfolder(subtomo_folder, remove=True)
        particle_list = []
        for i, row in tomo_star.iterrows():
            if tomo_idx is None or str(row.rlnIndex) in tomo_idx: 

                # wedge 
                # wedge_path = os.path.join(subtomo_folder, tomo_folder, 'wedge.mrc')
                # if "rlnTiltFile" in list(df.columns):
                #     self.psf(size=crop_size, tilt_file=row["rlnTiltFile"], output=wedge_path, between_tilts=between_tilts)
                # else:
                #     print(wedge_path)
                #     self.psf(size=crop_size, output=wedge_path, between_tilts=between_tilts)
                # if apply_wedge:
                #     wedge, vs = read_mrc(wedge_path)
                # else:
                #     wedge = None

                tomo_name = row[input_column]
                tomo, _ = read_mrc(tomo_name)
                tomo = normalize(tomo)

                if "rlnMaskName" in tomo_columns:
                    mask_file = row["rlnMaskName"]
                    with mrcfile.open(mask_file,permissive=True) as mrc:
                        mask=mrc.data
                else:
                    mask = np.ones_like(tomo)
                count_start = len(particle_list)
                if uniform_extract:
                    #TODO uniform extract with mask
                    subtomos_names = extract_with_overlap(tomo, crop_size, cube_size, subtomo_folder, prefix='', wedge=None)
                else:
                    n_subtomo_per_tomo = row["rlnNumberSubtomo"]
                    seeds=create_cube_seeds(tomo, n_subtomo_per_tomo, crop_size, mask)
                    subtomos_names = extract_subvolume(tomo, seeds, crop_size, subtomo_folder, count_start, wedge=None)
                
                
                for i in range(len(subtomos_names)):
                    im_name = '{}/subvolume{}_{:0>6d}.mrc'.format(subtomo_folder, '', count_start+i)
                    particle_list.append([im_name, cube_size, crop_size])

        df = pd.DataFrame(particle_list, columns = ["rlnParticleName", "rlnCubeSize","rlnCropSize"])
        starfile.write(df, subtomo_star)
        #extract_list = [df['rlnParticleName'].tolist()]   

        return  #extract_list
    
    def refine(self,
        subtomo_star: str,
        gpuID: str = None,
        iterations: int = None,
        data_dir: str = None,
        pretrained_model: str = None,
        log_level: str = "info",
        output_dir: str='results',
        remove_intermediate: bool =False,
        select_subtomo_number: int = None,
        ncpus: int = 16,
        continue_from: str=None,
        epochs: int = 10,
        batch_size: int = None,
        steps_per_epoch: int = None,

        noise_level:  tuple=(0.05,0.10,0.15,0.20),
        noise_start_iter: tuple=(11,16,21,26),
        noise_mode: str = None,
        noise_dir: str = None,
        learning_rate: float = None,
        drop_out: float = 0.3,
        convs_per_depth: int = 3,
        kernel: tuple = (3,3,3),
        pool: tuple = None,
        unet_depth: int = 3,
        filter_base: int = None,
        batch_normalization: bool = True,
        normalize_percentile: bool = True,

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


        from IsoNet.utils.dict2attr import Arg
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   
        
        
        params = Arg(locals())
        
        from IsoNet.bin.refine import run
        run(params)

       


        logging.info("Finished")

    def refine_backup(self, 
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

    def predict(self, star_file: str, 
                model: str, 
                output_dir: str='./corrected_tomos', 
                gpuID: str = None, 
                cube_size:int=64,
                crop_size:int=96,
                input_column: str = "rlnDeconvTomoName",
                batch_size:int=None,
                normalize_percentile: bool=True,
                log_level: str="info", 
                tomo_idx=None):
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
        from IsoNet.utils.fileio import write_mrc
        import starfile
        import numpy as np
        from IsoNet.utils.processing import normalize

        mkfolder(output_dir, remove=False)
        network = Net(filter_base = 64,unet_depth=3, add_last=True)
        network.load(model)

        star = starfile.read(star_file)
        star["rlnCorrectedTomoName"] = None
        for index, tomo_row in star.iterrows():
            tomo, _ = read_mrc(tomo_row[input_column])
            tomo = normalize(tomo*-1,percentile=False)
            outData = network.predict_map(tomo, output_dir,cube_size=cube_size, crop_size=crop_size).astype(np.float32) #train based on init model and save new one as model_iter{num_iter}.h5
            file_base_name = os.path.basename(tomo_row[input_column])
            file_name, file_extension = os.path.splitext(file_base_name)
            out_file_name = f"{output_dir}/corrected_{file_name}.mrc"
            write_mrc(out_file_name, outData*-1)        
            star.at[index, "rlnCorrectedTomoName"] = out_file_name
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



    def extract_both(self,  star, subtomo_folder="subtomos", out_star = "subtomos.star", 
                between_tilts=False, cube_size=96, crop_size=128, 
                random = False):
        self.extract(star, subtomo_folder=subtomo_folder, out_star = out_star, 
                between_tilts=between_tilts, cube_size=cube_size, crop_size=crop_size, 
                tomo_type=0, random = random)





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
