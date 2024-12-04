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
from IsoNet.utils.fileio import read_mrc, write_mrc, create_folder
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

    def prepare_star(self, full: str="None",
                     even: str="None",
                     odd: str="None",
                     mask_folder: str='None',
                     coordinate_folder: str='None',
                     star_name: str='tomograms.star',
                     pixel_size = 'auto', 
                     defocus_folder: str="None",
                     cs: float=2.7,
                     voltage: float=300,
                     ac: float=0.1,
                     create_average: bool=True,
                     number_subtomos = 1000):
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
        if full == "None":
            count_folder = even
        else:
            count_folder = full
        num_tomo = len(os.listdir(count_folder))
        print("number of tomograms", num_tomo)

        data = []
        label = []

        def add_param(folder_name, param_name, default_val="None"):
            if folder_name != "None" and folder_name != None:
                # TODO check file extension
                files = sorted(os.listdir(folder_name))
                files = [f"{folder_name}/{item}" for item in files]
                assert len(files) == num_tomo
                data.append(files)
            else:
                if type(default_val) == list:
                    data.append(default_val)
                else:
                    data.append([default_val]*num_tomo)
            label.append(param_name)

        def create_average(even, odd, full = "sum_even_odd_tomograms"):

            even_files_names = sorted(os.listdir(even))
            even_files = [f"{even}/{item}" for item in even_files_names]
            odd_files = sorted(os.listdir(odd))
            odd_files = [f"{odd}/{item}" for item in odd_files]
            create_folder(full)
            for i in range(len(even_files)):
                tomo_even, voxel_size = read_mrc(even_files[i])
                tomo_odd, _ = read_mrc(odd_files[i])
                write_mrc(f'{full}/{os.path.splitext(even_files_names[i])[0]}_full.mrc',tomo_odd+tomo_even, voxel_size=voxel_size)

        if full == "None" and create_average:
            create_average(even, odd)
        
        # tomograms setup
        add_param(full, 'rlnTomoName')
        add_param(even, 'rlnTomoReconstructedTomogramHalf1')
        add_param(odd, 'rlnTomoReconstructedTomogramHalf2')

        # voxel_size
        if pixel_size in ["auto","None"]:
            voxel_size_list = []
            counted_files_names = sorted(os.listdir(count_folder))
            counted_files = [f"{count_folder}/{item}" for item in counted_files_names]        
            for i in range(len(counted_files)):
                _, apix = read_mrc(counted_files[i], inplace = True)
                voxel_size_list.append(apix)
            add_param("None", "rlnPixelSize", voxel_size_list)
        else:           
            add_param("None", 'rlnPixelSize', pixel_size)

        if defocus_folder in ['None', None]:
            add_param("None", "rlnDefocus", 0)
            add_param("None", "rlnVoltage", voltage)
            add_param("None", "rlnSphericalAberration", cs)
            add_param("None", "rlnAmplitudeContrast", ac)
        else:
            #TODO defocus file folder 
            #from IsoNet.utils.fileio import read_defocus_file
            #TODO decide the defocus file format from CTFFIND and GCTF
            print("read from CTFFIND result not implimented")

        add_param("None", "rlnSnrFalloff",0)
        add_param("None", "rlnDeconvStrength",1)
        # add_param("None", "rlnDeconvTomoName","None")

        # mask parameters
        add_param("None", "rlnMaskBoundary","None")
        add_param("None", "rlnMaskDensityPercentage",50)
        add_param("None", "rlnMaskStdPercentage",50)
        add_param(mask_folder, "rlnMaskName","None")

        # tilt angle parameters
        add_param("None", "rlnTiltMin",-60)
        add_param("None", "rlnTiltMax",60)

        # subtomogram coordinates
        add_param(coordinate_folder, 'rlnBoxFile', "None")
        if coordinate_folder not in ["None", None]:
            number_subtomos = "None"
            print("the number of subtomogram for each tomogram will be determined by the subtomogram coordinate files")
        add_param("None", 'rlnNumberSubtomo',number_subtomos)
        

        
        data = list(map(list, zip(*data)))
        df = pd.DataFrame(data = data, columns = label)
        df.insert(0, 'rlnIndex', np.arange(num_tomo)+1)
        starfile.write(df,star_name)

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

    def predict(self, star_file: str, 
                model: str, 
                model2: str=None,
                output_dir: str='./corrected_tomos', 
                gpuID: str = None, 
                input_column: str = "rlnDeconvTomoName",
                apply_mw_x1: bool=False, 
                correct_CTF: bool=False,
                isCTFflipped: bool=False,
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

        from IsoNet.models.network import Net,DuoNet
        from IsoNet.utils.fileio import write_mrc
        import starfile
        import numpy as np
        from IsoNet.utils.processing import normalize
        if model2 not in ['None', None]:
            network = DuoNet(pretrained_model1=model, pretrained_model2=model2,state='predict')
        else:
            network = Net(pretrained_model=model,state='predict')
        create_folder(output_dir, remove=False)
        cube_size = network.cube_size
        inner_cube_size = cube_size//3*2

        star = starfile.read(star_file)

        out_column = "rlnCorrectedTomoName"
        if network.method == 'n2n':
            out_column = "rlnDenoisedTomoName"
        if out_column not in star.columns:
            star[out_column] = None

        def normalize_and_predict(network_model, tomo_name, F_mask=None):
            tomo, _ = read_mrc(tomo_name)
            tomo = normalize(tomo*-1,percentile=False)
            outData = network_model.predict_map(tomo, output_dir, cube_size=inner_cube_size, crop_size=cube_size, \
                                          F_mask=F_mask)
            if len(outData) == 2:
                return [x.astype(np.float32)*-1 for x in outData]
            else:
                return outData.astype(np.float32) * -1
        
        def get_base_filename(tomo_name):
            file_base_name = os.path.basename(tomo_name)
            file_name, file_extension = os.path.splitext(file_base_name)
            file_name = f"{output_dir}/corrected_{network.method}_{network.arch}_{file_name}"
            return file_name
        
        if isCTFflipped == True and correct_CTF == True:
            print("Do not need to correct_CTF in prediction when tomogram is ctf phase-flipped")
            print("Setting correct_CTF to False here")
            print("This is expected, do not need to worry")
            correct_CTF = False

        for index, tomo_row in star.iterrows():
            # wedgevolume
            if apply_mw_x1:
                min_angle, max_angle = float(tomo_row['rlnTiltMin']), float(tomo_row['rlnTiltMax'])
                from IsoNet.utils.missing_wedge import mw3D
                F_mask = mw3D(cube_size, missingAngle=[90 + min_angle, 90 - max_angle])
            else:
                F_mask = None
            
            if correct_CTF:
                from IsoNet.utils.CTF import get_ctf_3d
                defocus = tomo_row['rlnDefocus']/10000.
                ctf3d = get_ctf_3d(angpix=tomo_row['rlnPixelSize'], voltage=tomo_row['rlnVoltage'], \
                                   cs=tomo_row['rlnSphericalAberration'], defocus=defocus, phaseflipped=False,\
                                   phaseshift=0, amplitude=tomo_row['rlnAmplitudeContrast'],length=cube_size)
                ctf3d = np.sign(ctf3d)
                if F_mask in [None,"None"]:
                    F_mask = ctf3d
                else:
                    F_mask = ctf3d*F_mask

            if network.method in ['regular','isonet2']:
                star = starfile.read(star_file)
                if not input_column in star.columns or star.iloc[0][input_column] in [None, "None"]:
                    print("using rlnTomoName instead of rlnDeconvTomoName")
                    input_column = "rlnTomoName"
                outData_full = normalize_and_predict(network, tomo_row[input_column],F_mask=F_mask)
                base_filename = get_base_filename(tomo_row[input_column])

            if network.method in ['n2n','isonet2-n2n']:
                base_filename = get_base_filename(tomo_row['rlnTomoReconstructedTomogramHalf1'])
                out_half1 = normalize_and_predict(network, tomo_row["rlnTomoReconstructedTomogramHalf1"],F_mask=F_mask)
                out_half2 = normalize_and_predict(network, tomo_row["rlnTomoReconstructedTomogramHalf2"],F_mask=F_mask)
                if model2 in ['None', None]:

                    outData_full = (out_half1 + out_half2) * (0.5)
                else:
                    
                    out_file_net1_half1 = f"{base_filename}_net1_half1.mrc"
                    out_file_net2_half1 = f"{base_filename}_net2_half1.mrc"
                    out_file_net1_half2 = f"{base_filename}_net1_half2.mrc"
                    out_file_net2_half2 = f"{base_filename}_net2_half2.mrc"
                    
                    write_mrc(out_file_net1_half1, out_half1[0])
                    write_mrc(out_file_net2_half1, out_half1[1])
                    write_mrc(out_file_net1_half2, out_half2[0])
                    write_mrc(out_file_net2_half2, out_half2[1])

                    outData_full = (out_half1[0] + out_half1[1] + out_half2[0] + out_half2[1])*0.25
            out_file_name = f"{base_filename}.mrc"
            write_mrc(out_file_name, outData_full)
            star.at[index, out_column] = out_file_name            
        starfile.write(star,star_file)

    def refine(self, 
                   star_file: str,
                   output_dir: str="isonet_maps",

                   gpuID: str=None,
                   ncpus: int=16, 

                   method: str="isonet2-n2n",
                   arch: str='unet-medium',
                   pretrained_model: str=None,
                   pretrained_model2: str=None,

                   cube_size: int=80,
                   epochs: int=50,

                   
                   input_column: str= 'rlnDeconvTomoName',
                   batch_size: int=None, 
                   acc_batches: int=1,
                   loss_func: str = "L2",
                   learning_rate: float=3e-4,
                   T_max: int=10,
                   learning_rate_min:float=3e-4,
                   random_rotation: bool=True, 
                   mw_weight: float=20,
                   apply_mw_x1: bool=True, 
                   compile_model: bool=False,
                   mixed_precision: bool=True,

                   correct_CTF: bool=False,
                   isCTFflipped: bool=False,

                   noise_level: float=0.2, 
                   noise_mode: str="ramp",

                   with_predict: bool=True,
                   split_halves: bool=False
                   ):
        '''
        method: n2n isonet2 isonet2-n2n
        arch: unet-default, unet-small, unet-medium, HSFormer, vtunet
        gamma: <=0 normal loss, >0 ddw loss, ddw default 2, 
        apply_mw_x1: apply missing wedge to subtomograms in the begining. True seems to be better.
        compile_model: improve the speed of training, sometime error
        mixed_precision: use mixed precision to reduce VRAM and increase speed
        loss_func: L2, Huber
        '''
        create_folder(output_dir,remove=False)

        ngpus, gpuID, gpuID_list=parse_gpu(gpuID)
        # print(ngpus, gpuID, gpuID_list)
        n_workers = ncpus//ngpus
        if n_workers == 0:
            n_workers = 1
        ncpus = ngpus*n_workers
        print(f"{n_workers} CPU cores per GPU, total {ncpus} CPUs")

        if batch_size is None:
            if ngpus == 1:
                batch_size = 4
            else:
                batch_size = 2 * len(gpuID_list)
        steps_per_epoch = 200000000

        print(f"method {method}")
        if method == "isonet2":
            star = starfile.read(star_file)
            if not input_column in star.columns or star.iloc[0][input_column] in [None, "None"]:
                print("using rlnTomoName instead of rlnDeconvTomoName")
                input_column = "rlnTomoName"
            num_noise_volume = 1000
            if noise_level > 0:
                print("generating noise folder")
                from IsoNet.utils.noise import make_noise_folder
                noise_dir = f"{output_dir}/noise_volumes"
                # Note: the angle for this noise generation is range(-90,90,3)
                make_noise_folder(noise_dir,noise_mode,cube_size,num_noise_volume,ncpus=ncpus)

        training_params = {
            "method":method,
            "input_column": input_column,
            "arch": arch,
            "ncpus": ncpus,
            "star_file":star_file,
            "output_dir":output_dir,
            "batch_size":batch_size,
            "acc_batches": acc_batches,
            "epochs": epochs,
            "steps_per_epoch":steps_per_epoch,
            "learning_rate":learning_rate,
            "cube_size": cube_size,
            "mw_weight": mw_weight,
            "random_rotation":random_rotation,
            'apply_mw_x1':apply_mw_x1,
            'mixed_precision':mixed_precision,
            'compile_model':compile_model,
            'T_max':T_max,
            'learning_rate_min':learning_rate_min,
            'loss_func':loss_func,
            'correct_CTF':correct_CTF,
            "isCTFflipped": isCTFflipped,
            "starting_epoch": 0,
            "noise_level": noise_level
        }
        if split_halves:
            from IsoNet.models.network import DuoNet
            network = DuoNet(method=method, arch=arch, cube_size=cube_size, pretrained_model1=pretrained_model, pretrained_model2 = pretrained_model2, state='train')
            network.train(training_params) #train based on init model and save new one as model_iter{num_iter}.h5            
        else:
            training_params['split'] = "full"
            from IsoNet.models.network import Net
            network = Net(method=method, arch=arch, cube_size=cube_size, pretrained_model=pretrained_model,state='train')
            network.train(training_params) #train based on init model and save new one as model_iter{num_iter}.h5
        if with_predict:
            if split_halves:
                model_file1 = f"{output_dir}/network_{arch}_{cube_size}_top.pt"
                model_file2 = f"{output_dir}/network_{arch}_{cube_size}_bottom.pt"
                self.predict(star_file=star_file, model=model_file1, model2=model_file2, output_dir=output_dir, gpuID=gpuID,\
                                             correct_CTF=correct_CTF,isCTFflipped=isCTFflipped) 
            else:
                model_file = f"{output_dir}/network_{arch}_{cube_size}_full.pt"
                self.predict(star_file=star_file, model=model_file, output_dir=output_dir, gpuID=gpuID, \
                             correct_CTF=correct_CTF,isCTFflipped=isCTFflipped) 
                #f"{training_params['output_dir']}/network_{training_params['arch']}_{training_params['method']}.pt"

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



    def extract_legacy(self,  
                star_file: str,
                input_column: str = "rlnDeconvTomoName",
                subtomo_folder="subtomos", 
                subtomo_star = "subtomos.star", 
                cube_size = 64,
                crop_size = None, 
                tomo_idx = None,
                uniform_extract=False):
        
        from IsoNet.utils.processing import normalize
        from IsoNet.preprocessing.cubes import extract_with_overlap
        from IsoNet.preprocessing.cubes import extract_subvolume, create_cube_seeds

        if crop_size is None:
            crop_size = cube_size + 16

        tomo_star = starfile.read(star_file)
        tomo_columns = tomo_star.columns.to_list()
        create_folder(subtomo_folder, remove=True)
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

                if "rlnMaskName" in tomo_columns and row["rlnMaskName"] not in [None, "None"]:
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
    
    def refine_legacy(self,
        subtomo_star: str,
        gpuID: str = None,
        iterations: int = None,
        data_dir: str = None,
        pretrained_model: str = None,
        log_level: str = "info",
        output_dir: str='results',
        remove_intermediate: bool =False,
        select_subtomo_number: int = None,
        ncpus: int = 8,
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
