#!/usr/bin/env python3
import fire
import logging
import os, sys
from IsoNet.utils.dict2attr import check_parse
from fire import core
import starfile
import mrcfile
import numpy as np
from IsoNet.utils.fileio import read_mrc, write_mrc, create_folder
from IsoNet.utils.utils import parse_params, process_gpuID
import tqdm
import pandas as pd
from IsoNet.utils.utils import idx2list, process_tomograms
from IsoNet.utils.deconvolution import deconv_one
from IsoNet.bin.make_mask import make_mask
import os
import sys
import logging
import numpy as np
import starfile
from IsoNet.models.network import Net, DuoNet
from IsoNet.utils.processing import normalize
from IsoNet.utils.missing_wedge import mw3D
from IsoNet.utils.CTF import get_ctf_3d
import shutil
from IsoNet.utils.plot_metrics import save_slices_and_spectrum

class ISONET:
    """
    ISONET: Train on tomograms to simultaneously perform missing wedge correction, CTF correction, and denoising.

    Usage: 
        isonet.py [command] -h

    Commands:   
        prepare_star      Generate a tomograms.star file from tomogram folders
        denoise           Train denoising model (noise2noise)
        deconv            CTF deconvolution for tomograms
        make_mask         Generate masks for tomograms
        refine            Train missing wedge correction model
        predict           Predict tomograms using trained model
    """

    def prepare_star(self, full: str="None",
                     even: str="None",
                     odd: str="None",
                     mask_folder: str='None',
                     coordinate_folder: str='None',
                     star_name: str='tomograms.star',
                     pixel_size = 'auto', 
                     defocus: list=[10000],
                     cs: float=2.7,
                     voltage: float=300,
                     ac: float=0.1,
                     tilt_min: float=-60,
                     tilt_max: float=60,
                     create_average: bool=False,
                     number_subtomos = 'auto'):
        """
        Generate a tomograms.star file from folder(s) containing tomogram files.

        This command generates a tomograms.star file from a folder containing only tomogram files (.mrc or .rec).
        If there is no even/odd separation, please specify full folder.
        If you have even and odd tomograms, please use the even_folder and odd_folder parameters.

        Args:
            full: Directory containing full tomogram(s). Usually 1-5 tomograms are sufficient.
            even: Directory containing even half tomograms.
            odd: Directory containing odd half tomograms.
            mask_folder: Directory containing mask files for tomograms.
            coordinate_folder: Directory containing coordinate files for subtomogram extraction.
            star_name: Output star file, similar to that from "relion". You can modify this file manually or with gui.
            pixel_size: Pixel size in angstroms. Usually you want to bin your tomograms to about 10A pixel size. Extreme deviations in pixel size are not recommended, since the target resolution on Z-axis should be about 30A.
            defocus: Defocus for zero tilt in angstroms. Can be a single value or a list of values for each tomogram.
            cs: Spherical aberration in mm.
            voltage: Acceleration voltage in kV.
            ac: Amplitude contrast.
            tilt_min: Minimum tilt angle.
            tilt_max: Maximum tilt angle.
            tilt_step: Tilt step size.
            create_average: Whether to create average tomograms from even/odd pairs.
            number_subtomos: Number of subtomograms to be extracted during training. You can directly modify this in the generated star file or with gui if you want different numbers extracted for different tomograms.
        """
        import starfile
        import pandas as pd
        count_folder = full if full not in ["None", None] else even
        num_tomo = len(os.listdir(count_folder))
        logging.info(f"Number of tomograms: {num_tomo}")
        
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
                    if len(default_val) == 1:
                        data.append(default_val * num_tomo)
                    else:
                        data.append(default_val)
                else:
                    data.append([default_val]*num_tomo)
            label.append(param_name)

        def create_average_func(even, odd, average = "averaged_tomos"):
            even_files_names = sorted(os.listdir(even))
            even_files = [f"{even}/{item}" for item in even_files_names]
            odd_files_names = sorted(os.listdir(odd))
            odd_files = [f"{odd}/{item}" for item in odd_files_names]
            create_folder(average)
            for i in tqdm.tqdm(range(len(even_files)), desc="Averaging even and odd tomograms", unit=" tomograms"):
                tomo_even, voxel_size = read_mrc(even_files[i])
                tomo_odd, _ = read_mrc(odd_files[i])
                write_mrc(f'{average}/{os.path.splitext(even_files_names[i])[0]}_full.mrc',tomo_odd+tomo_even, voxel_size=voxel_size)
            return average
        
        if full in ["None",None] and create_average:
            logging.info("Creating average from even/odd tomograms ")
            full = create_average_func(even, odd)
        
        # tomograms setup
        add_param(full, 'rlnTomoName')
        add_param(even, 'rlnTomoReconstructedTomogramHalf1')
        add_param(odd, 'rlnTomoReconstructedTomogramHalf2')


        # voxel_size
        if pixel_size in ["None", None, "auto"]:
            voxel_size_list = []
            counted_files_names = sorted(os.listdir(count_folder))
            counted_files = [f"{count_folder}/{item}" for item in counted_files_names]        
            for i in range(len(counted_files)):
                _, apix = read_mrc(counted_files[i], inplace = True)
                voxel_size_list.append(str(apix))
            add_param("None", "rlnPixelSize", voxel_size_list)
        else:           
            add_param("None", 'rlnPixelSize', pixel_size)

        # if defocus_folder in ['None', None]:
        add_param("None", "rlnDefocus", defocus)
        add_param("None", "rlnVoltage", voltage)
        add_param("None", "rlnSphericalAberration", cs)
        add_param("None", "rlnAmplitudeContrast", ac)

        # mask parameters
        add_param("None", "rlnMaskBoundary","None")
        add_param(mask_folder, "rlnMaskName","None")

        # tilt angle parameters
        add_param("None", "rlnTiltMin",tilt_min)
        add_param("None", "rlnTiltMax",tilt_max)

        # subtomogram coordinates
        add_param(coordinate_folder, 'rlnBoxFile', "None")
        if number_subtomos in [None, "None","auto"]:
            number_subtomos = int(3000/num_tomo)
        if coordinate_folder not in ["None", None]:
            number_subtomos = "None"
            logging.info("the number of subtomogram for each tomogram will be determined by the subtomogram coordinate files")
        add_param("None", 'rlnNumberSubtomo', number_subtomos)
        add_param(None, 'rlnCorrectedTomoName', "None")
        

        
        data = list(map(list, zip(*data)))
        df = pd.DataFrame(data = data, columns = label)
        df.insert(0, 'rlnIndex', np.arange(num_tomo)+1)
        starfile.write(df,star_name)

        #df.to_json('.to_node.json', orient='records', lines=True)  # orient='records' gives a list of dictionaries
        df.to_json('.to_node.json')  # orient='records' gives a list of dictionaries

    def star2json(self, star_file="tomograms.star", json_file='tomograms.json'):
        """
        Convert star file to JSON format.

        Args:
            star_file: Input star file to convert.
            json_file: Output JSON file name.
        """
        star = starfile.read(star_file)
        star.to_json(json_file)

    def json2star(self, json_file, star_name="tomograms.star"):
        """
        Convert JSON file to star format.

        Args:
            json_file: Input JSON file to convert.
            star_name: Output star file name.
        """
        df = pd.read_json(json_file)
        starfile.write(df,star_name)

    def deconv(self, star_file: str,
        output_dir:str="./deconv",
        input_column: str="rlnTomoName",
        snrfalloff: float=1,
        deconvstrength: float=1,
        highpassnyquist: float=0.02,
        chunk_size: int=None,
        overlap_rate: float= 0.25,
        ncpus:int=4,
        phaseflipped:bool=False,
        tomo_idx: str=None):
        """
        CTF deconvolution for tomograms.

        This step is recommended because it enhances low-resolution information for better contrast.
        This is unnecessary for phase plate data.

        Args:
            star_file: Star file for tomograms.
            output_dir: Folder to save deconvolved tomograms.
            input_column: Column name in star file to use as input tomograms.
            snrfalloff: SNR falloff rate with frequency. Higher values correspond to less high frequency data. If not set, the program will look for the parameter in the star file or use the default.
            deconvstrength: Strength of the deconvolution. If not set, the program will look for the parameter in the star file or use the default.
            highpassnyquist: Highpass filter at very low frequency. The default is usually sufficient.
            chunk_size: If memory is limited, set chunk_size to crop the tomogram into smaller chunks for processing. May induce artifacts at chunk edges; increase overlap_rate if needed.
            overlap_rate: Overlap rate for adjacent chunks.
            ncpus: Number of CPUs to use.
            phaseflipped: Whether input tomograms are phase flipped.
            tomo_idx: If set, process only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").
        """
        def deconv_row(i, row, new_star):
            tomo_file = row[input_column]
            deconv_tomo_name = os.path.join(output_dir, os.path.basename(tomo_file))

            common_kwargs = {
                "voltage":        row["rlnVoltage"],
                "cs":             row["rlnSphericalAberration"],
                "defocus":        row["rlnDefocus"] / 10000.0,
                "pixel_size":     row["rlnPixelSize"],
                "snrfalloff":     snrfalloff,
                "deconvstrength": deconvstrength,
                "highpassnyquist": highpassnyquist,
                "chunk_size":     chunk_size,
                "overlap_rate":   overlap_rate,
                "phaseflipped":   phaseflipped,
                "ncpu":           ncpus,
            }

            deconv_one(
                tomo_file,
                deconv_tomo_name,
                output_dir,
                **common_kwargs
            )

            new_star.at[i, "rlnDeconvTomoName"] = deconv_tomo_name

        process_tomograms(
            star_file,
            output_dir,
            tomo_idx,
            desc="CTF Deconv",
            row_processor=deconv_row
        )


    def make_mask(self,star_file,
                input_column: str = "rlnDeconvTomoName",
                output_dir: str = 'mask',
                patch_size: int=4,
                density_percentage: int=50,
                std_percentage: int=50,
                z_crop:float=0.2,
                tomo_idx=None):
        """
        Generate masks for subtomogram extraction that exclude "empty" areas of tomograms.

        The masks do not need to be precise. In general, more selective masks reduce the number of subtomograms needed.
    
        Args:
            star_file: Path to the star file containing tomogram information.
            input_column: Column name in star file to use as input tomograms.
            output_dir: Directory to save generated masks.
            patch_size: The size of the box from which the max-filter and std-filter are calculated.
            density_percentage: The approximate percentage of pixels to keep based on their local pixel density. If not set, will look for parameter in star file or use default.
            std_percentage: The approximate percentage of pixels to keep based on their local standard deviation. If not set, will look for parameter in star file or use default.
            z_crop: The percentage of tomogram data masked out along the z axis, split between the top and bottom regions. For example, 0.2 will mask out the top 10% and bottom 10% region along z axis.
            tomo_idx: If set, process only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").
        """

        def mask_row(i, row, new_star):
            if input_column in row and row[input_column] not in [None, "None"]:
                tomo = row[input_column]
            elif 'rlnTomoName' in row and row.rlnTomoName not in [None, "None"]:
                tomo = row["rlnTomoName"]
            else:
                tomo = row["rlnTomoReconstructedTomogramHalf1"]

            if 'rlnMaskBoundary' in row and row.rlnMaskBoundary not in [None, "None"]:
                boundary = row.rlnMaskBoundary
            else:
                boundary = None

            base = os.path.splitext(os.path.basename(tomo))[0]
            out_path = os.path.join(output_dir, f"{base}_mask.mrc")

            make_mask(
                tomo,
                out_path,
                mask_boundary=boundary,
                side=patch_size,
                density_percentage=density_percentage,
                std_percentage=std_percentage,
                surface=z_crop/2
            )
            new_star.at[i, 'rlnMaskName'] = out_path

        logging.info("Generating masks for tomograms")
        process_tomograms(
            star_file,
            output_dir,
            tomo_idx,
            desc="Make Mask",
            row_processor=mask_row
        )


    def predict(self, star_file: str, 
                model: str, 
                output_dir: str='./corrected_tomos', 
                gpuID: str = None, 
                input_column: str = "rlnDeconvTomoName",
                apply_mw_x1: bool=True, 
                isCTFflipped: bool=False,
                padding_factor: float=1.5,
                tomo_idx=None,
                output_prefix=""):
        """
        Predict tomograms using trained model.

        Args:
            star_file: Star file for tomograms.
            model: Path to trained network model (.pt file).
            output_dir: Directory to save output predicted tomograms.
            gpuID: GPU IDs to use during prediction (e.g., "0,1,2,3").
            input_column: Column name in star file to use as input tomograms.
            apply_mw_x1: Whether to apply missing wedge mask to input tomograms.
            isCTFflipped: Whether input tomograms are phase flipped.
            padding_factor: Padding factor for prediction cubes.
            tomo_idx: If set, process only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").
            output_prefix: Prefix to append to predicted MRC files.
        """
        ngpus, gpuID, gpuID_list = process_gpuID(gpuID)

        network = Net(pretrained_model=model, state='predict')

        cube_size = network.cube_size
        CTF_mode = network.CTF_mode
        if hasattr(network, 'do_phaseflip_input'):
            do_phaseflip_input = network.do_phaseflip_input
        else:
            logging.info("network do not have do_phaseflip_input")
            do_phaseflip_input = True

        all_tomo_paths = []
        def predict_row(i, row, new_star):
            # 1) Build missing‑wedge mask if requested
            F_mask = (
                mw3D(cube_size, missingAngle=[
                    90 + float(row.rlnTiltMin),
                    90 - float(row.rlnTiltMax)
                ]) if apply_mw_x1 else None
            )

            # 2) Optionally incorporate CTF into the mask
            if CTF_mode in ["wiener","phase_only","network"] and do_phaseflip_input:
                if isCTFflipped == False:
                    defocus = row.rlnDefocus / 10000.0
                    ctf3d = np.sign(
                        get_ctf_3d(
                            angpix=row.rlnPixelSize,
                            voltage=row.rlnVoltage,
                            cs=row.rlnSphericalAberration,
                            defocus=defocus,
                            phaseflipped=False,
                            phaseshift=0,
                            amplitude=row.rlnAmplitudeContrast,
                            length=cube_size
                        )
                    )
                    F_mask = ctf3d * F_mask if F_mask is not None else ctf3d

            # 3) Decide which tomo to feed in
            if network.method in ['regular', 'isonet2']:
                tomo_paths = [row.get(input_column) or row.rlnTomoName]
            else:
                tomo_paths = [
                    row.rlnTomoReconstructedTomogramHalf1,
                    row.rlnTomoReconstructedTomogramHalf2
                ]
            base = os.path.splitext(os.path.basename(tomo_paths[0]))[0]
            prefix = f"{output_dir}/{output_prefix}_{network.method}_{network.arch}_{base}"
            
            out_data = []
            for tomo_p in tomo_paths:
                tomo_vol, _ = read_mrc(tomo_p)
                # now we are using precentile again similar to isonet1
                if network.method =='isonet2':
                    tomo_vol = normalize(tomo_vol * -1, percentile=True)
                else:
                    Z = tomo_vol.shape[0]
                    tomo_vol = tomo_vol*-1
                    mean = np.mean(tomo_vol[Z//2-30:Z//2+30])
                    std = np.std(tomo_vol[Z//2-30:Z//2+30])
                    tomo_vol = (tomo_vol-mean)/std#normalize(tomo_vol * -1, percentile=False)
                out_data.append(network.predict_map(
                    tomo_vol, output_dir,
                    cube_size= int(cube_size / padding_factor+0.1),
                    crop_size=cube_size,
                    F_mask=F_mask
                ))

            out_data = sum(out_data) / len(out_data)

            out_file = f"{prefix}.mrc"
            write_mrc(out_file, out_data.astype(np.float32) * -1)
            all_tomo_paths.append(out_file)


            # 5) Update STAR
            column = "rlnDenoisedTomoName" if network.method == 'n2n' else "rlnCorrectedTomoName"
            new_star.at[i, column] = out_file   
            save_slices_and_spectrum(out_file,output_dir,'')
            logging.info(f" Predicted {[os.path.relpath(p) for p in tomo_paths]} → {out_file}")
            

        process_tomograms(
            star_path=star_file,
            output_dir=output_dir,
            idx_str=tomo_idx,
            desc="Predict Tomograms",
            row_processor=predict_row
        )
        return all_tomo_paths

    def denoise(self, 
                   star_file: str,
                   output_dir: str="denoise",

                   gpuID: str=None,
                   ncpus: int=16, 

                   arch: str='unet-medium',
                   pretrained_model: str=None,

                   cube_size: int=96,
                   epochs: int=50,

                   batch_size = "auto", 
                   loss_func: str = "L2",
                   save_interval: int=10,
                   learning_rate: float=3e-4,
                   learning_rate_min:float=3e-4,
                   mixed_precision: bool=True,

                   CTF_mode: str="None",
                   isCTFflipped: bool=False,
                   do_phaseflip_input: bool=True,
                   bfactor: float=0,
                   clip_first_peak_mode: float=1,
                   snrfalloff: float=0,
                   deconvstrength: float=1,
                   highpassnyquist:float=0.02,

                   with_preview: bool=True,
                   prev_tomo_idx:str=1
                   ):
        """
        Train denoising model using noise2noise approach.

        This method uses noise2noise training approach for denoising tomograms.

        Args:
            star_file: Star file for tomograms.
            output_dir: Directory to save trained model and results.
            gpuID: GPU IDs to use during training (e.g., "0,1,2,3").
            ncpus: Number of CPUs to use for data processing.
            arch: Model architecture (unet-large, unet-small, unet-medium, HSFormer, vtunet).
            pretrained_model: Path to pretrained model to continue training.
            cube_size: Size of training subvolumes.
            epochs: Number of training epochs.
            batch_size: Batch size for training. If none is provided, it will be calculated based on GPU availability.
            loss_func: Loss function to use (L2, Huber, L1).
            save_interval: Interval to save model checkpoints.
            learning_rate: Initial learning rate.
            learning_rate_min: Minimum learning rate for scheduler.
            mixed_precision: Use mixed precision to reduce VRAM and increase speed.
            CTF_mode: CTF correction mode (None, phase_only, wiener, network).
            isCTFflipped: Whether input tomograms are phase flipped.
            do_phaseflip_input: Whether to apply phase flip during training.
            bfactor: B-factor for filtering.
            clip_first_peak_mode: Mode for clipping first peak in CTF: 0 - no clipping, 1 - constant, 2 - negative sine function, 3 - cosine function.
            snrfalloff: SNR falloff parameter. A higher value means less high frequency data.
            deconvstrength: Deconvolution strength parameter. A higher value means stronger deconvolution.
            highpassnyquist: High-pass filter parameter. A higher value means more low frequency data is removed.
            with_preview: Whether to run prediction after training.
            prev_tomo_idx: If set, process only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").
        """
        acc_batches=1
        create_folder(output_dir,remove=False)
        batch_size, ngpus, ncpus = parse_params(batch_size, gpuID, ncpus)
        steps_per_epoch = 200000000

        training_params = {
            "method":'n2n',
            "input_column": None,
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
            "mw_weight": 0,
            'apply_mw_x1':True,
            'mixed_precision':mixed_precision,
            'compile_model':False,
            'T_max':save_interval,
            'learning_rate_min':learning_rate_min,
            'loss_func':loss_func,
            'CTF_mode':CTF_mode,
            "phaseflipped": isCTFflipped,
            "starting_epoch": 0,
            "noise_level": 0,
            "correct_between_tilts":False,
            "start_bt_size":128,
            'snrfalloff':snrfalloff,
            "deconvstrength": deconvstrength,
            "highpassnyquist":highpassnyquist,
            "do_phaseflip_input":do_phaseflip_input,
            "bfactor":bfactor,
            "clip_first_peak_mode":clip_first_peak_mode,
        }

        training_params['split'] = "full"
        from IsoNet.models.network import Net
        from IsoNet.utils.plot_metrics import save_slices_and_spectrum
        network = Net(method='n2n', arch=arch, cube_size=cube_size, pretrained_model=pretrained_model,state='train')
        network.prepare_train_dataset(training_params)
        if with_preview:
            new_epochs = save_interval
            training_params["epochs"] = new_epochs
            for step in range(save_interval, epochs+1, save_interval):
                logging.info(f"Training for {step-save_interval} to {step} epochs")
                network.train(training_params) #train based on init model and save new one as model_iter{num_iter}.h5
                model_file = f"{output_dir}/network_n2n_{arch}_{cube_size}_full.pt"
                shutil.copy(model_file, f"{output_dir}/network_n2n_{arch}_{cube_size}_epoch{step}_full.pt")
                all_tomo_paths = self.predict(star_file=star_file, model=model_file, output_dir=output_dir, gpuID=gpuID, \
                            isCTFflipped=isCTFflipped, tomo_idx=prev_tomo_idx,output_prefix=f"corrected_epochs{step}") 
                save_slices_and_spectrum(all_tomo_paths[0],output_dir,step)
        else:
            network.train(training_params) #train based on init model and save new one as model_iter{num_iter}.h5



    def refine(self, 
                   star_file: str,
                   output_dir: str="isonet_maps",

                   gpuID: str=None,
                   ncpus: int=16, 

                   method: str="None",
                   arch: str='unet-medium',
                   pretrained_model: str=None,

                   cube_size: int=96,
                   epochs: int=50,
                   
                   input_column: str= 'rlnDeconvTomoName',
                   batch_size: int=None, 
                   loss_func: str = "L2",
                   learning_rate: float=3e-4,
                   save_interval: int=10,
                   learning_rate_min:float=3e-4,
                   mw_weight: float=-1,
                   apply_mw_x1: bool=True, 
                   mixed_precision: bool=True,

                   CTF_mode: str="None",
                   clip_first_peak_mode: int=1,
                   bfactor: float=0,

                   isCTFflipped: bool=False,
                   do_phaseflip_input: bool=True,

                   noise_level: float=0, 
                   noise_mode: str="nofilter",

                   random_rot_weight: float=0.2,

                   with_preview: bool=True,
                   prev_tomo_idx:str=1,

                   snrfalloff: float=0,
                   deconvstrength: float=1,
                   highpassnyquist:float=0.02,
                   with_deconv: bool=False,
                   with_mask: bool=False,
                   mask_update_interval: int=0,
                   ):
        """
        Train missing wedge correction model for tomogram refinement.

        This is the main training method for missing wedge correction using IsoNet2 approach.
        Supports both single tomogram training (isonet2) and noise2noise training (isonet2-n2n).

        Args:
            star_file: Star file for tomograms.
            output_dir: Directory to save trained model and results.
            gpuID: GPU IDs to use during training (e.g., "0,1,2,3").
            ncpus: Number of CPUs to use for data processing.
            method: Training method (isonet2, isonet2-n2n). If None, will be auto-detected from star file.
            arch: Model architecture (unet-large, unet-small, unet-medium, HSFormer, vtunet).
            pretrained_model: Path to pretrained model to continue training. Previous method, arch, cube_size, CTF_mode, and metrics will be loaded.
            cube_size: Size of training cubes.
            epochs: Number of training epochs.
            input_column: Column name in star file to use as input tomograms.
            batch_size: Batch size for training. If none is provided, it will be calculated based on GPU availability.
            loss_func: Loss function to use (L2, Huber, L1).
            learning_rate: Initial learning rate.
            save_interval: Interval to save model checkpoints. Default is epochs/10.
            learning_rate_min: Minimum learning rate for scheduler.
            mw_weight: Weight for missing wedge loss. Higher values correspond to stronger emphasis on missing wedge regions. Disabled by default.
            apply_mw_x1: Whether to apply missing wedge to subtomograms at the beginning.
            mixed_precision: Whether to use mixed precision to reduce VRAM and increase speed.
            CTF_mode: CTF correction mode (None, phase_only, wiener, network).
            clip_first_peak_mode: Mode for clipping first peak in CTF: 0 - no clipping, 1 - constant, 2 - negative sine function, 3 - cosine function.
            bfactor: B-factor applied during training/prediction to boost high-frequency content; ideal values around 200 - 500.
            isCTFflipped: Whether input tomograms are phase flipped.
            do_phaseflip_input: Whether to apply phase flip during training.
            noise_level: Noise level to add during training.
            noise_mode: Noise generation mode (None, ramp, hamming).
            random_rot_weight: Percentage of rotations applied as random augmentation.
            with_preview: Whether to run prediction every save interval.
            prev_tomo_idx: If set, automatically predict only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").
            snrfalloff: SNR falloff parameter.
            deconvstrength: Deconvolution strength parameter.
            highpassnyquist: High-pass filter parameter.
            with_deconv: Only applies to method isonet2. Whether to run deconvolution preprocessing automatically.
            with_mask: Whether to generate masks automatically.
            num_mask_updates: Number of times to update masks based on corrected tomograms during training.
        """
        compile_model=False
        # there is some questions about this parameter, relate to the placement of the zerograd
        acc_batches=1
        correct_between_tilts: bool=False
        start_bt_size: int=128

        create_folder(output_dir,remove=False)
        batch_size, ngpus, ncpus = parse_params(batch_size, gpuID, ncpus, fit_ncpus_to_ngpus=True)
        steps_per_epoch = 200000000
        
        star = starfile.read(star_file)
        if method in ["None", None]:
            has_full = 'rlnTomoName' in star.columns and star.iloc[0]['rlnTomoName'] not in [None, "None"]
            has_split = 'rlnTomoReconstructedTomogramHalf1' in star.columns and star.iloc[0]['rlnTomoReconstructedTomogramHalf1'] not in [None, "None"]
            if has_full and has_split:
                raise ValueError("Both full and half tomograms are present in the star file. Please specify method as either 'isonet2' or 'isonet2-n2n'.")
            elif has_split:
                logging.info("Using method isonet2-n2n")
                method = 'isonet2-n2n'
            else:
                logging.info("Using method isonet2")
                method = 'isonet2'
        
        if method == "isonet2":
            star = starfile.read(star_file)
            if with_deconv:
                logging.info("Running deconvolution preprocessing")
                self.deconv(star_file=star_file,
                            output_dir=f"{output_dir}/deconv_tomos",
                            input_column="rlnTomoName",
                            snrfalloff=snrfalloff,
                            deconvstrength=deconvstrength,
                            highpassnyquist=highpassnyquist,
                            ncpus=ncpus)
                input_column = "rlnDeconvTomoName"
            if noise_level <= 0:
                logging.info("Your noise_level is 0, we recommend to increase noise_level for denoising during isonet2 training")
            else:
                num_noise_volume = 1000
                logging.info("Generating noise folder")
                from IsoNet.utils.noise import make_noise_folder
                noise_dir = f"{output_dir}/noise_volumes"
                # Note: the angle for this noise generation is range(-90,90,3)
                make_noise_folder(noise_dir,noise_mode,cube_size,num_noise_volume,ncpus=ncpus)
        
        if not input_column in star.columns or star.iloc[0][input_column] in [None, "None"]:
            if method == "isonet2":
                logging.info("using rlnTomoName instead of rlnDeconvTomoName")
                input_column = "rlnTomoName"
            elif method == "isonet2-n2n":
                logging.info("using rlnTomoReconstructedTomogramHalf1 instead of rlnDeconvTomoName")
                input_column = "rlnTomoReconstructedTomogramHalf1"

        if with_mask:
            self.make_mask(star_file=star_file,
                           input_column=input_column,
                           output_dir=f"{output_dir}/masks",
                           tomo_idx=None)

        if mw_weight > 0:
            logging.info("Enabling mw_weight")
            # logging.info("using masked loss seperating in and out of the missing wedge")

        training_params = {
            "method": method,
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
            'apply_mw_x1':apply_mw_x1,
            'mixed_precision':mixed_precision,
            'compile_model':compile_model,
            'T_max':save_interval,
            'learning_rate_min':learning_rate_min,
            'loss_func':loss_func,
            'CTF_mode':CTF_mode,
            "phaseflipped": isCTFflipped,
            "starting_epoch": 0,
            "noise_level": noise_level,
            "correct_between_tilts":correct_between_tilts,
            "start_bt_size":start_bt_size,
            'snrfalloff':snrfalloff,
            "deconvstrength": deconvstrength,
            "highpassnyquist":highpassnyquist,
            "random_rot_weight":random_rot_weight,
            'do_phaseflip_input':do_phaseflip_input,
            "clip_first_peak_mode":clip_first_peak_mode,
            "bfactor": bfactor
        }

        training_params['split'] = "full"
        from IsoNet.models.network import Net
        network = Net(method=method, arch=arch, cube_size=cube_size, pretrained_model=pretrained_model,state='train')
        network.prepare_train_dataset(training_params)
        if with_preview:
            new_epochs = save_interval
            training_params["epochs"] = new_epochs
            for step in range(save_interval, epochs+1, save_interval):
                logging.info(f"Training for {step-save_interval} to {step} epochs")
                network.train(training_params) #train based on init model and save new one as model_iter{num_iter}.h5
                model_file = f"{output_dir}/network_{method}_{arch}_{cube_size}_full.pt"
                shutil.copy(model_file, f"{output_dir}/network_{method}_{arch}_{cube_size}_epoch{step}_full.pt")
                if mask_update_interval == step // save_interval:
                    all_tomo_paths = self.predict(star_file=star_file, model=model_file, output_dir=output_dir, gpuID=gpuID, \
                                isCTFflipped=isCTFflipped, tomo_idx=None,output_prefix=f"corrected_epochs{step}")
                    logging.info(f"Updating masks based on the corrected tomograms at epoch {step}")
                    self.make_mask(star_file=star_file,
                           input_column="rlnCorrectedTomoName",
                           output_dir=f"{output_dir}/masks",
                           tomo_idx=None)
                else:
                    all_tomo_paths = self.predict(star_file=star_file, model=model_file, output_dir=output_dir, gpuID=gpuID, \
                                isCTFflipped=isCTFflipped, tomo_idx=prev_tomo_idx,output_prefix=f"corrected_epochs{step}")
                save_slices_and_spectrum(all_tomo_paths[0],output_dir,step)
        else:
            network.train(training_params) #train based on init model and save new one as model_iter{num_iter}.h5

    # def refine_v1(self,
    #     star_file: str,
    #     output_dir: str='results',

    #     gpuID: str = None,
    #     ncpus: int = 16,

    #     arch: str="unet-medium",
    #     pretrained_model: str = None,

    #     cube_size = 80,

    #     input_column: str= 'rlnDeconvTomoName',
    #     batch_size: int = None,
    #     loss_func: str= 'L2',
    #     learning_rate: float = 3e-4,
    #     T_max: int =  10,
    #     learning_rate_min: float = 3e-4,
    #     compile_model: bool = False,
    #     mixed_precision: bool = True,

    #     iterations: int = 30,
    #     continue_from: str=None,
    #     epochs: int = 10,

    #     noise_level:  tuple = (0.05,0.1,0.15,0.2),
    #     noise_mode: str = 'noFilter',
    #     noise_start_iter: tuple = (10,15,20,25),

    #     with_preview: bool = True,

    #     # temporarily fixed parameters
    #     normalize_percentile: bool = True,
    #     select_subtomo_number: int = None,
    #     noise_dir: str = None,
    #     split_halves: bool=False,
    #     data_dir: str = None,
    #     log_level: str = "info",
    #     remove_intermediate: bool =False,
    #     steps_per_epoch: int = None,
    #     tomo_idx = None,
    #     crop_size = None, 
    #     random_rotation: bool =  False,

    # ):

    #     """
    #     \n\n
    #     IsoNet.py map_refine half.mrc FSC3D.mrc --mask mask.mrc --limit_res 3.5 [--gpuID] [--ncpus] [--output_dir] [--fsc_file]...
    #     :param i: Input half map 1
    #     :param aniso_file: 3DFSC file
    #     :param mask: Filename of a user-provided mask
    #     :param independent: Independently process half1 and half2, this will disable the noise2noise-based denoising but will provide independent maps for gold-standard FSC
    #     :param gpuID: The ID of gpu to be used during the training.
    #     :param alpha: Ranging from 0 to inf. Weighting between the equivariance loss and consistency loss.
    #     :param beta: Ranging from 0 to inf. Weighting of the denoising. Large number means more denoising. 
    #     :param limit_res: Important! Resolution limit for IsoNet recovery. Information beyong this limit will not be modified.
    #     :param ncpus: Number of cpu.
    #     :param output_dir: The name of directory to save output maps
    #     :param pretrained_model: The neural network model with ".pt" to continue training or prediction. 
    #     :param reference: Retain the low resolution information from the reference in the IsoNet refine process.
    #     :param ref_resolution: The limit resolution to keep from the reference. Ususlly  10-20 A resolution. 
    #     :param epochs: Number of epochs.
    #     :param n_subvolume: Number of subvolumes 
    #     :param predict_crop_size: The size of subvolumes, should be larger then the cube_size
    #     :param cube_size: Size of cubes for training, should be divisible by 16, e.g. 32, 64, 80.
    #     :param batch_size: Size of the minibatch. If None, batch_size will be the max(2 * number_of_gpu,4). batch_size should be divisible by the number of gpu.
    #     :param acc_batches: If this value is set to 2 (or more), accumulate gradiant will be used to save memory consumption.  
    #     :param learning_rate: learning rate. Default learning rate is 3e-4 while previous IsoNet tomography used 3e-4 as learning rate
    #     """


    #     from IsoNet.utils.utils import parse_params
    #     from IsoNet.utils.dict2attr import load_args_from_json, filter_dict
    #     params=filter_dict(locals())
    #     if params['continue_from'] is not None:
    #         logging.info('\n######Isonet Continues Refining######\n')
    #         params = load_args_from_json(params.continue_from)

    #     params["batch_size"], params["ngpus"], params["ncpus"] = parse_params(batch_size, gpuID, ncpus)
    #     # params["crop_size"] = params.get("crop_size") or params["cube_size"] + 16
    #     params["crop_size"] = params.get("cube_size")

    #     params["data_dir"] = params.get("data_dir") or f'{params["output_dir"]}/data'
    #     params["steps_per_epoch"] = params.get("steps_per_epoch") or 200
    #     params["noise_dir"] = params.get("noise_dir") or f'{params["output_dir"]}/training_noise'
    #     params["log_level"] = params.get("log_level") or "info"

    #     if type(params["noise_level"]) not in [tuple, list]:
    #         params["noise_level"] = [params["noise_level"]]
    #     if type(params["noise_start_iter"]) not in [tuple, list]:
    #         params["noise_start_iter"] = [params["noise_start_iter"]]

    #     from IsoNet.bin.refine import run
    #     run(params)

    #     # if with_preview:
    #         # self.predict(star_file, params['])


    def simulate_noise_F(self, size=128, tilt_step=3, repeats=1000, ncpus=51):
        """
        Simulate Fourier domain noise statistics for tomographic reconstruction.

        Args:
            size: Volume size for simulation.
            tilt_step: Angular step between tilts in degrees.
            repeats: Number of noise realizations to average.
            ncpus: Number of CPU cores for parallel processing.
        """
        from IsoNet.utils.WBP import backprojection
        from IsoNet.utils.processing import normalize
        from joblib import Parallel, delayed
        #result = np.zeros((ctf2d.shape[1], ctf2d.shape[1], ctf2d.shape[1]), dtype=np.float32)
        angles = np.arange(-90,90,tilt_step)
        def simulate_one():
            noise = np.random.normal(size=(len(angles), size,size))
            out = backprojection(noise, angles, filter_name='ramp')
            F_result = np.fft.fftshift(np.fft.fftn(out))
            out = np.abs(F_result).astype(np.float32)
            #out = (np.real(F_result)).astype(np.float32)
            return out
        results = Parallel(n_jobs=ncpus)(delayed(simulate_one)() for _ in range(repeats))
        result = np.average(results, axis=0)
        result = np.average(result, axis=1)
        result = normalize(result, percentile = True, pmin=33.0, pmax=66.0, clip=True)
        out_name = f"simulated_F_{size}_{tilt_step}.mrc"
        with mrcfile.new(out_name, overwrite=True) as mrc:
             mrc.set_data(result)
        return result

    def FSC(self, t1, t2, m):
    # def postprocessing(self, t1, t2, b1, b2):
        """
        Combine half-maps for postprocessing and FSC calculation. Default with 2 inputs is cross-correlation.

        Args:
            t1: Path to first  half-map.
            t2: Path to second  half-map.
            m: Path to mask.
        """
        t1, _ = read_mrc(t1)
        t2, _ = read_mrc(t2)
        m, _ = read_mrc(m)
        # b1, _ = read_mrc(b1)
        # b2, _ = read_mrc(b2)

        # shape = t1.shape
        # mask_top = np.zeros_like(t1)
        # mask_top[:,:shape[1]//2,:] = 1
        # mask_bottom = 1 - mask_top

        # half1 = t1*mask_bottom+b1*mask_top
        # half2 = t2*mask_bottom+b2*mask_top

        from IsoNet.utils.FSC import FSC
        np.savetxt('FSC.txt',
                   FSC(t1,t2, m))
        # from IsoNet.utils.processing import FSC
        # logging.info(FSC(half1,half2))
        return 0
    
    def resize(self, star_file:str, apix: float=15, out_folder="tomograms_resized"):
        """
        Rescale tomograms to a given pixel size.

        Args:
            star_file: Star file containing tomogram information.
            apix: Target pixel size in Angstroms.
            out_folder: Output folder for rescaled tomograms.
        """
        '''
        This function rescale the tomograms to a given pixelsize
        '''
        import mrcfile
        import starfile
        import os
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
                logging.info("scaling: {}".format(output_tomo1_name))
                new_data = zoom(data, zoom_factor,order=3, prefilter=False)
                with mrcfile.new(output_tomo1_name,overwrite=True) as mrc:
                    mrc.set_data(new_data)
                    mrc.voxel_size = apix

            if old_tomo2_name is not None:
                output_tomo2_name = out_folder+'/'+tomo_folder+'/'+os.path.basename(old_tomo2_name)
                with mrcfile.open(old_tomo2_name, permissive=True) as mrc:
                    data = mrc.data
                logging.info("scaling: {}".format(output_tomo2_name))
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
        logging.info("scale_finished: {}".format(out_folder+'.star'))

    def postprocessing(self, t1, t2, b1, b2):
        t1, _ = read_mrc(t1)
        t2, _ = read_mrc(t2)
        b1, _ = read_mrc(b1)
        b2, _ = read_mrc(b2)

        shape = t1.shape
        mask_top = np.zeros_like(t1)
        mask_top[:,:shape[1]//2,:] = 1
        mask_bottom = 1 - mask_top

        half1 = t1*mask_bottom+b1*mask_top
        half2 = t2*mask_bottom+b2*mask_top

        from IsoNet.utils.processing import FSC
        print(FSC(half1,half2))
        return 0
    
    def powerlaw_filtering(self, 
                    h1: str,
                    o: str = "weighting.mrc",
                    mask: str=None, 
                    low_res: float=50,
                    ):
        """
        Apply power-law filtering to flatten Fourier amplitude within resolution range.

        This will sharpen the map by flattening Fourier amplitudes.

        Args:
            h1: Input map file to be filtered.
            o: Output filename for the filtered map.
            mask: Optional mask file to apply before filtering.
            low_res: Low resolution limit in Angstroms. Typically 10-50A. High resolution limit is typically the resolution at FSC=0.143.
        """
        import numpy as np
        import mrcfile
        from numpy.fft import fftshift, fftn, ifftn

        with mrcfile.open(h1,'r') as mrc:
            input_map = mrc.data
            nz,ny,nx = input_map.shape
            voxel_size = mrc.voxel_size.x
            logging.info(voxel_size)
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
        #logging.info(k)
        b = np.log(F_curve[limit_r_low]) - k * np.log(limit_r_low**-1)
        logging.info(b)
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
        """
        Generate point spread function (PSF) or missing wedge mask for tomographic reconstruction.

        Args:
            size: Volume size for PSF generation.
            tilt_file: Path to tilt angle file, or None to use default -60 to +60 range.
            w: Line width for between-tilts mode.
            output: Output filename for PSF volume.
            between_tilts: Generate mask between tilt angles vs standard wedge.
        """
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
   

    def check(self):
        """
        Check IsoNet installation and GPU performance.

        Usage:
            isonet.py check

        This command verifies that IsoNet is properly installed and tests GPU performance
        with mixed precision vs single precision training.
        """
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])

        from IsoNet.bin.predict import predict
        from IsoNet.bin.refine import run
        import skimage
        import tqdm
        logging.info('IsoNet --version 2.0 alpha installed')
        logging.info(f"checking gpu speed")
        from IsoNet.bin.verify import verify
        fp16, fp32 = verify()
        logging.info(f"time for mixed/half precsion and single precision are {fp16} and {fp32}. ")
        logging.info(f"The first number should be much smaller than the second one, if not please check whether cudnn, cuda, and pytorch versions match.")

    def gui(self):
        """
        Launch the graphical user interface for IsoNet.

        Usage:
            isonet.py gui

        This opens the IsoNet GUI application for interactive use.
        """
        import IsoNet.gui.Isonet_star_app as app
        app.main()

def Display(lines, out):
    """
    Display formatted text output.
    
    Args:
        lines: List of text lines to display.
        out: Output stream to write to.
    """
    text = "\n".join(lines) + "\n"
    out.write(text)

def main():
    """
    Main entry point for IsoNet command-line interface.
    
    Configures logging and launches Fire CLI framework to handle command dispatch.
    """
    core.Display = Display
    logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)
    if len(sys.argv) > 1:
       check_parse(sys.argv[1:])
    fire.Fire(ISONET)


if __name__ == "__main__":
    exit(main())
