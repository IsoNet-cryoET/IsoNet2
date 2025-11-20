<!-- # IsoNet2 Tutorial

**IsoNet2** is a deep-learning software package for simultaneous missing wedge correction, denoising, and CTF correction in cryo-electron tomography reconstructions using a deep neural network trained on information from the original tomogram(s). Compared to IsoNet1, IsoNet2 produces tomograms with higher resolution and less noise in roughly a tenth of the time. The software requires tomograms as input. Paired tomograms for Noise2Noise training can be split by either frame or tilt.

**IsoNet2** contains six modules: **prepare star**, **CTF deconvolve**, **generate mask**, **denoise**, **refine**, and **predict**. All commands in IsoNet operate on **.star** text files which record paths of data and relevant parameters. For detailed descriptions of each module please refer to the individual tasks. Users can choose to utilize IsoNet through either GUI or command-lines.

# 1. Installation and System Requirements
The following tutorial is written for assuming absolutely no experience with Anaconda or Linux environments.

Software Requirements: This Linux installation of IsoNet requires [`CUDA Version >= 12.0`](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html) and [Anaconda](https://www.anaconda.com/download). A more comprehensive guide for installing CUDA can be found in the [TomoPy UI docs](https://tomopyui.readthedocs.io/en/latest/install.html).


Hardware Requirements: ***Nvidia GTX 1080Ti or newer, with at least 8 GB VRAM ?***

You can check your `CUDA` version using `nvidia-smi`, which should produce something similar to below:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.261.03             Driver Version: 535.261.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |

...

|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+

```
Clone this repository:
```
git clone https://github.com/procyontao/IsoNet2.git
cd IsoNet2
```

Run `./install.sh`. This creates an Conda environment (installing requirements) using the included **isonet2_environment.yml** file and updates your environment variables (allowing you to call isonet.py) by running `source isonet2.bashrc`. You may append this command to your .bashrc file.


Installation should take 5-10 minutes. Upon successful installation, running the command `isonet.py --help` should display the following help message.
```
INFO: Showing help with the command 'isonet.py -- --help'.

NAME
    isonet.py - ISONET: Train on tomograms and restore missing-wedge.

...
```
# 2. Tutorial

The following outlines the example IsoNet2 command-line and GUI workflow in Linux for two datasets. More in-depth explanations of each parameter can be found under ***4. IsoNet Modules.*** A video tutorial can be found in the following [Google Drive.](https://drive.google.com/drive/u/1/folders/1P9sxSSJIWPs7hIGey3I38u3B2OvmiCAC)

## 2.1 Ribosome dataset (split tomogram)

This [dataset](https://drive.google.com/drive/u/0/folders/1mmvTDvi-rjDcCF1aSBZLCxQ5ulWI1sRl) contains 5 tomograms from EMPIAR-10985 with frame-based EVN/ODD split.

### 2.1.1 Prepare Tomograms and Starfile

In a new working directory, download the **tomograms_split** folder. This contains even and odd half tomograms.  

Output for `ls -1 tomograms_split/EVN`:
```
9x9_ts_01_sort_EVN_Vol-rotx.mrc
9x9_ts_02_sort_EVN_Vol-rotx.mrc
9x9_ts_03_sort_EVN_Vol-rotx.mrc
9x9_ts_04_sort_EVN_Vol-rotx.mrc
9x9_ts_05_sort_EVN_Vol-rotx.mrc
```
Output for `ls -1 tomograms_split/ODD`:
```
9x9_ts_01_sort_ODD_Vol-rotx.mrc
9x9_ts_02_sort_ODD_Vol-rotx.mrc
9x9_ts_03_sort_ODD_Vol-rotx.mrc
9x9_ts_04_sort_ODD_Vol-rotx.mrc
9x9_ts_05_sort_ODD_Vol-rotx.mrc
```

![Fig. 1](figures/Fig1.png, "Tomograms 1-5")
  
Prepare the starfile.   
**number_subtomos** defines how many subtomograms are extracted per tomogram per epoch. By default, it is calculated using *1000/num_tomo* to extract 50,000 total subtomograms over 50 epochs.   
**defocus** is the approximate defocus in _Å_ calculated for the 0-degree tilt images. This is not necessary for already-CTF-corrected or phase plate data. You may enter a single value to be used for every tomogram or a list of values to be applied to their respective tomograms. You may also use your default text editor (or our ***GUI***) to open **tomograms.star** and manually enter the defocus value for each tomogram in the **rlnDefocus** column.
```
isonet.py prepare_star --even tomograms_split/EVN --odd tomograms_split/ODD --star_name tomograms.star --pixel_size 5.35 --defocus "[25928.79, 25048.72, 25785.23, 26376.26, 26910.0]"
```

Output for `cat tomograms.star`:

```
# Created by the starfile Python package (version 0.5.6) at 13:43:23 on 17/11/2025


data_

loop_
_rlnIndex #1
_rlnTomoName #2
_rlnTomoReconstructedTomogramHalf1 #3
_rlnTomoReconstructedTomogramHalf2 #4
_rlnPixelSize #5
_rlnDefocus #6
_rlnVoltage #7
_rlnSphericalAberration #8
_rlnAmplitudeContrast #9
_rlnMaskBoundary #10
_rlnMaskName #11
_rlnTiltMin #12
_rlnTiltMax #13
_rlnBoxFile #14
_rlnNumberSubtomo #15
1       None    tomograms_split/EVN/9x9_ts_01_sort_EVN_Vol-rotx.mrc     tomograms_split/ODD/9x9_ts_01_sort_ODD_Vol-rotx.mrc     5.400000        25928.790000    300   2.700000 0.100000        None    None    -60     60      None    1000
2       None    tomograms_split/EVN/9x9_ts_02_sort_EVN_Vol-rotx.mrc     tomograms_split/ODD/9x9_ts_02_sort_ODD_Vol-rotx.mrc     5.400000        25048.720000    300   2.700000 0.100000        None    None    -60     60      None    1000
3       None    tomograms_split/EVN/9x9_ts_03_sort_EVN_Vol-rotx.mrc     tomograms_split/ODD/9x9_ts_03_sort_ODD_Vol-rotx.mrc     5.400000        25785.230000    300   2.700000 0.100000        None    None    -60     60      None    1000
4       None    tomograms_split/EVN/9x9_ts_04_sort_EVN_Vol-rotx.mrc     tomograms_split/ODD/9x9_ts_04_sort_ODD_Vol-rotx.mrc     5.400000        26376.260000    300   2.700000 0.100000        None    None    -60     60      None    1000
5       None    tomograms_split/EVN/9x9_ts_05_sort_EVN_Vol-rotx.mrc     tomograms_split/ODD/9x9_ts_05_sort_ODD_Vol-rotx.mrc     5.400000        26910.000000    300   2.700000 0.100000        None    None    -60     60      None    1000
```
### 2.1.2 Refine

Train IsoNet to reconstruct missing wedge and denoise subtomograms.

```
isonet.py refine tomograms.star --output_dir isonet_maps --gpuID <ids> --epochs 50
```

### 2.1.3. Predict

Apply the trained model to the original tomograms to recover missing wedge regions:

```
isonet.py predict tomograms.star isonet_maps/network_isonet2-n2n_unet-medium_96_full.pt
```

## 2.2 HIV dataset with full tomograms as input

### 2.1.1 Prepare Star

Collect all tomogram files (.mrc or .rec) into a folder.

Generate a STAR file:

```
isonet.py prepare_star <folder> --outputstar <tomograms.star> --pixelsize <value> --numbersubtomos <number>
```

Manually edit with Vim or other editors:

IMAGE

Add defocus for each tomogram in the fourth column (in Å). Adjust rlnNumberSubtomo for each tomogram as needed.

### 2.1.2 Deconv

Perform CTF deconvolution (optional, skip for phase plate tomograms or refinements using network-based deconvolution):
isonet.py deconv <tomograms.star> --deconvfolder <output_folder> --snrfalloff <value> --deconvstrength <value>
Other options: --highpassnyquist, --chunksize, --overlap_rate, --ncpu
2.3. Make Mask
Generate a mask for each tomogram to exclude empty regions:
isonet.py make_mask <tomograms.star> --maskfolder <output_folder> --patchsize <value> --densitypercentage <val> --stdpercentage <val> --zcrop <val>

### 2.1.4 Refine
Train IsoNet to reconstruct missing wedge and denoise subtomograms:
isonet.py refine <subtomo.star> --outputdir <results_folder> --gpuID <ids> --iterations <number> --noiselevel <list> --noisestartiter <list> --method <isonet2-n2n/isonet2> --arch <network type>
2.5. Predict
Apply the trained model to the original tomograms to recover missing wedge regions:
isonet.py predict <tomograms.star> <model_file.h5> --gpuID <ids> --outputdir <output_folder> --inputcolumn <column name>

## 2.3 FAQ

**Q: When using my own data, the fitting and/or validation loss is very low or even close to zero. Is this a problem? What should I do about it?**\
  A: Low losses can be due to overall small voxel values in the tomogram and may cause instabilities during model fitting. If you observe very low losses (e.g. `1e-3` to `1e-9`) in the first epoch of model fitting, try standardizing your tomograms such that they have zero mean and unit variance before. You can do that manually, or by setting `standardize_full_tomos: true` in the `shared` field of your config.

- **Q: How to speed up the model fitting process?**  
  A: There is a number of things you can try: 
    - **Smaller model:** You can try using a smaller U-Net. While this will reduce the expressiveness of the model, we have found that using a U-Net with 32 channels in the first layer provides similar results to a U-Net with the default 64 channels, but is signficantly faster to train. You can modify the number of channels by adjusting `chans` in the `unet_params_dict` argument.
    - **Manual early stopping:** While in general, you should fit the model until the fitting and/or validation losses converge or until the validation starts to increase, you can try to stop earlier. We found that DeepDeWedge often produces good results that do not change much anymore even if the fitting and/or validation losses are still decreasing. Therefore, we recommend to occasinally check the output of `ddw refine-tomogram` during fitting to see if the results are already satisfactory. However, be aware that reconstructions may still improve as long as the losses are decreasing.
    - **Faster dataloading:** If you notice that the GPU utilization fluctuates a lot during model fitting, you can increase the number of CPU workers for data loading by adjusting `num_workers`.
    - **Larger batches:** If you have a fast GPU with a lot of memory, you can try increasing the batch size by adjusting the `batch_size`.


- **Q: How large should the sub-tomograms for model fitting and tomogram refinement be?**\
  A: We have found that larger sub-tomograms give better results up to a point (see the Appendix of our paper).
  In most of our experiments, we used sub-tomograms of size 96x96x96 voxels, and we recommend not going below 64x64x64 voxels. \
  **Note**: The size of the sub-tomograms must be divisible by 2^`num_downsample_layers`, where `num_downsample_layers` is the number of downsample layers in the U-Net, e.g., for a U-Net with 3 downsample layers, the size of the sub-tomograms must be divisible by 8.

- **Q: How many sub-tomograms should I use for model fitting?**\
  A: So far, we have seen good results when fitting the default U-Net on at least 150 sub-tomograms of size 96x96x96 voxels. The smaller the sub-tomograms, the more sub-tomograms you should use, but we have not yet found a clear rule of thumb. You can increase/decrease the number of sub-tomograms by decreasing/increasing the three values in the `subtomo_extraction_strides` argument used in `ddw prepare-data`.



# 3. Example (EMPIAR Ribosome Dataset)

dataset: https://www.ebi.ac.uk/empiar/EMPIAR-10985/

## 3.1 Ribosome tomograms split into even and odd frames

mw correction + ctf + denoise

prepare_star + refine

isonet.py prepare_star --even tomoset/EVN --odd tomoset/ODD --pixel_size 5.35 --number_subtomos 800

isonet.py refine tomograms.star -o 01_test --CTF_mode network

isonet.py refine tomograms.star -o 01_test --CTF_mode wiener >> 01_test/log.txt 2>&1 &

First, create a folder for your project. Inside your folder, create a subfolder (in this case the subfolder's name will be 'tomoset') and move all tomograms into the new subfolder.

Next, we will use the prepare_star function to create a STAR file containing paths and parameters of all the tomograms in the dataset that will be used later. It can handle single tomograms as well as split even and odd tomograms. For this tutorial, our dataset contains an even and odd separation so we will use the --even and --odd parameters. We will also define our --pixel_size to be 5.35 and --number_subtomos to be 800.

isonet.py prepare_star --even tomoset/EVN --odd tomoset/ODD --pixel_size 5.35 --number_subtomos 800

After preparing our star file, we will use the refine function
3.2 Ribosome tomograms without split
mw correction + ctf+ denoise without EVN/ODD
prepare_star + refine

isonet.py prepare_star --even tomoset/EVN --odd tomoset/ODD --pixel_size 5.35 --number_subtomos 800 --create_average True

MODIFY STAR FILE WITH CORRECT DEFOCUS (ask)

isonet.py deconv tomograms.star

isonet.py make_mask tomograms.star

isonet.py refine tomograms.star -o 01_test --noise_level 0.2 >> 01_test/log.txt 2>&1 &
3.3 denoise with n2n + ctf → cryocare + ctf

prepare_star + denoise
isonet.py prepare_star --even tomoset/EVN --odd tomoset/ODD --pixel_size 5.35 --number_subtomos 800

#1.
isonet.py deconv tomograms.star

isonet.py make_mask tomograms.star

isonet.py denoise tomograms.star -o 03_test >> 03_test/log.txt 2>&1 &

#2.
isonet.py denoise tomograms.star -o 03_test --CTF_mode network >> 03_test/log.txt 2>&1 &
#3.
isonet.py denoise tomograms.star -o 03_test --CTF_mode wiener >> 03_test/log.txt 2>&1 &

# 4. IsoNet Modules

## prepare_star
Generate a tomograms.star file that lists tomogram file paths and acquisition metadata used by all downstream IsoNet commands. The function can accept either a single set of full tomograms or paired even/odd half tomograms for noise2noise workflows.

### Key parameters

+ full — Directory with full tomogram files; use for single-map training (isonet2).

+ even — Directory with even-half tomograms; use with odd for noise2noise (isonet2-n2n).

+ odd — Directory with odd-half tomograms; used together with even.

+ tilt_min — Minimum tilt angle in degrees; default **-60**. Override if your tilt range is different.
+ tilt_max — Maximum tilt angle in degrees; default **60**. Override if your tilt range is different.

+ tilt_step — Tilt step size in degrees; default **3**.

+ pixel_size — Pixel size (Å). Defaults to "auto" (reads from tomograms) but you can set a target value (commonly ~10 Å for typical IsoNet runs).

+ create_average — If True and no full provided, create full tomograms by summing the provided even and odd folders; useful for producing a single full tomogram from two halves.

+ number_subtomos — Number of subtomograms to extract per tomogram (written to rlnNumberSubtomo). For IsoNet2, increasing this is analogous to increasing training exposure and can improve results at the cost of runtime and memory.

+ mask_folder — Optional directory with masks; entries are recorded in rlnMaskName.

+ coordinate_folder — Optional directory with subtomogram coordinate files; if provided, the number of subtomograms is taken from the coordinate files and overrides number_subtomos.

+ cs, voltage, ac, rlnDefocus — Microscope parameters (spherical aberration mm, acceleration voltage kV, amplitude contrast, defocus in the STAR default units). Set only if different from defaults.

### Practical notes
> Use even and odd when you plan to use noise2noise training; use full for single-map training.
If tilt range differs from ±60°, supply tilt_min and tilt_max so the code records the correct missing-wedge geometry.
Inspect and edit the generated STAR if you need tomogram-specific subtomogram counts or have pregenerated mask/defocus entries.

## deconv
CTF deconvolution preprocessing that enhances low-resolution contrast and recovers information attenuated by the microscope contrast transfer function. Recommended for non–phase-plate data; skip for phase-plate data or if intending to use network-based CTF deconvolution.

### Key parameters
+ star_file — Input STAR listing tomograms and acquisition metadata.

+ input_column — STAR column used for input tomogram paths (default **rlnTomoName**).

+ output_dir — Folder to write deconvolved tomograms (rlnDeconvTomoName entries point here).

+ snrfalloff — Controls frequency-dependent SNR attenuation applied during deconvolution; default **1.0**. Larger values reduce high-frequency contribution more aggressively and can stabilize deconvolution on noisy data; smaller values preserve more high-frequency content but risk amplifying noise.

+ deconvstrength — Scalar multiplier for deconvolution strength; default **1.0**. Increasing this emphasizes correction and low-frequency recovery but can introduce ringing/artifacts if set too high.

+ highpassnyquist — Fraction of the Nyquist used as a very-low-frequency high-pass cutoff; default **0.02**. Use to remove large-scale intensity gradients and drift; usually left at default.

+ chunk_size — If set, tomograms are processed in smaller cubic chunks to reduce memory usage. Useful for very large tomograms or limited RAM/VRAM. May create edge artifacts if chunks are too small.

+ overlap_rate — Fractional overlap between adjacent chunks when chunking (default **0.25**). Larger overlaps reduce edge artifacts at cost of extra computation.

+ ncpus — Number of CPU workers for CPU-bound parts of deconvolution; increase on multi-core systems.

+ phaseflipped — If True, input is assumed already phase-flipped; otherwise the function uses defocus and CTF info to apply phase handling.

### Practical notes
> Inspect deconvolved outputs visually for ringing or other artifacts after changing snrfalloff or deconvstrength.
Use chunking plus a moderate overlap_rate (0.25–0.5) when memory is limited.

## make_mask
Generate masks to prioritize regions of interest. Masks improve sampling efficiency and training stability.
### Key parameters
+ star_file — Input STAR listing tomograms.

+ input_column — STAR column to read tomograms from (default **rlnDeconvTomoName**; falls back to **rlnTomoName** if absent).

+ output_dir — Folder to save mask MRCs; rlnMaskName is updated in the STAR.

+ patch_size — Local patch size used for max/std local filters (default **4**). Larger values smooth detection of specimen regions; default works for typical pixel sizes.

+ density_percentage — Percentage of voxels retained based on local density ranking (default **50**). Lower values create stricter masks (keep fewer voxels).

+ std_percentage — Percentage retained based on local standard-deviation ranking (default **50**). Lower values emphasize textured regions.

+ z_crop — Fraction of tomogram Z to crop from both ends (default **0.2** masks out top and bottom 10% each). Use to avoid sampling low-quality reconstruction edges.

+ tomo_idx — Limit mask generation to a subset of STAR entries (e.g., "1,3,5-7").

### Practical notes
Defaults are suitable for most datasets; tune density/std percentages for very sparse specimens or dense, crowded volumes.
If automatic masks miss specimen regions, edit boundaries in the STAR or provide manual masks.

## denoise and refine
Both functions are training entry points. Use denoise for pure noise-to-noise (n2n) training workflows; use refine for IsoNet2 missing-wedge correction (IsoNet2) or IsoNet2-n2n combined modes. Many parameters are shared between them.

### Key parameters

+ method (refine only) — "isonet2" for single-map missing-wedge correction, "isonet2-n2n" for noise2noise when even/odd halves are present. If omitted, the code auto-detects the method from the STAR columns.

+ arch — Network architecture string (e.g., unet-small, unet-medium, unet-large, HSFormer, vtunet). Determines model capacity and VRAM requirements.

+ cube_size — Size in voxels of training subvolumes (default **96**). Must be compatible with the network (divisible by the network downsampling factors).

+ epochs — Number of training epochs; longer for harder recovery, but watch validation.

+ batch_size — Number of subtomograms per optimization step; if None, this is automatically derived from the available GPUs. Batch size per GPU matters for gradient stability.

+ acc_batches — Number of gradient accumulation steps to emulate larger batches when memory is limited.

+ mixed_precision — If True, uses float16/mixed precision to reduce VRAM and speed up training.

+ CTF_mode — CTF handling mode: "None", "phase_only", "wiener", or "network".
    + "network" applies a CTF-shaped filter to network inputs so the network learns to restore it.
    + "wiener" configures the pipeline to emulate a Wiener-filtered target.

+ clip_first_peak_mode — Controls attenuation of overrepresented very-low-frequency CTF peak: 
    + 0 none
    + 1 constant clip
    + 2 negative sine
    + 3 cosine

+ bfactor — B-factor applied during training/prediction to boost high-frequency content; ideal values around 200 - 500.

+ noise_level — For plain isonet2 (non-n2n) training, supply >0 to enable denoising capability (adds artificial noise during training).

+ noise_mode — Controls filter applied when generating synthetic noise (used with noise_level).

+ split_halves — If True, train separate top/bottom or even/odd networks (DuoNet) and save separate checkpoints.

+ snrfalloff, deconvstrength, highpassnyquist — parameters for CTF deconvolution 
    + forwarded to deconvolution if you refine with --with_deconv True
    + used to calculate Wiener filter for network-based deconvolution

+ with_deconv — If True (refine only), run deconvolution and mask creation before training and set input_column to the deconvolved tomograms.

+ with_predict — If True, run prediction using the final checkpoint(s) after training.

### Practical notes
> Choose arch, cube_size, and batch_size to fit your GPU memory; larger architectures and cubes improve fidelity but increase resource needs.
Enable mixed_precision to save VRAM and speed up training if your GPU and drivers support it.
Use split_halves to obtain gold-standard half-map workflows (independent models for even/odd).
If you request deconvolution within refine, tune snrfalloff and deconvstrength there to avoid overcorrection artifacts.

## predict

Apply a trained IsoNet model to tomograms to produce denoised or missing-wedge–corrected volumes. Prediction utilizes the model's saved cube size and CTF handling options, but allows for runtime adjustments.

### Key parameters
+ star_file — Input STAR describing tomograms to predict.

+ model — Path to trained model checkpoint (.pt) for single-model prediction.

+ model2 — Optional second checkpoint for DuoNet (dual-model) prediction.

+ output_dir — Folder to save predicted tomograms; outputs are recorded in the STAR as 
rlnCorrectedTomoName or rlnDenoisedTomoName depending on method.

+ gpuID — GPU IDs string (e.g., "0" or "0,1"); use multiple GPUs when available for speed.

+ input_column — STAR column used for input tomogram paths (default **rlnDeconvTomoName**).

+ apply_mw_x1 — If True (default), build and apply the missing-wedge mask to cubic inputs before prediction.

+ phaseflipped — Declare if input tomograms are already phase-flipped; affects CTF handling.

+ do_phaseflip_input — Whether to apply phase-flip handling to inputs for CTF-aware modes (default **True**).

+ padding_factor — Cubic padding factor used during tiling to reduce edge effects (default **1.5**); larger padding reduces seams but increases computation.

+ tomo_idx — Process a subset of STAR entries by index.

### Practical notes
> Match prediction cube/crop sizes and padding to the network’s training settings (these come from the model object).
When using CTF-aware models, ensure phaseflipped and STAR defocus/CTF fields are correct. -->