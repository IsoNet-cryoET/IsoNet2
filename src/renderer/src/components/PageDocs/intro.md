# IsoNet Tutorial

## 0. Introduction

**IsoNet2** is a deep-learning software package for missing wedge correction, Noise2Noise denoising, and CTF correction in cryo-electron tomography reconstructions. All these functions are achieved by training a single neural network using information learned from the original tomogram(s). Compared with IsoNet1, IsoNet2 training results in higher resolution and less noise, and it is about also 10x faster than IsoNet1.

The software takes tomograms as input. Paired tomograms for Noise2Noise training can be split by either frame or tilt.

**IsoNet2** contains modules: prepare star, CTF deconvolve, generate mask, denoise, refine, and predict. All commands in IsoNet operate on **.star** text files which include description and path of data and parameters. Including pixel size, defocus value for 0-degree tilt, and the tilt range are all recommended for training preparation. Users can choose to utilize IsoNet through either GUI or command-lines.

## 1. Installation and System Requirements

IsoNet runs on Linux and requires CUDA-capable GPUs.
Recommended: Nvidia GTX 1080Ti or newer, with at least 8 GB VRAM per GPU.README.md

Dependencies (via Anaconda environment):

    • Python ≥3.6
    • torch (PyTorch)
    • cudatoolkit and cudnn
    • tqdm, matplotlib, scipy, numpy, scikit-image, mrcfile, fire

Installation steps:

• Create a conda environment, install dependencies using install.sh or pip.

• Install cudatoolkit/cudnn compatible with the GPU.

• Install PyTorch from https://pytorch.org.

• Update environment variables via .bashrc or source-env.sh:

```{.bash language="bash"}
export PATH=<PATH_TO_ISONET2_FOLDER>/IsoNet/bin:$PATH
export PYTHONPATH=<PATH_TO_ISONET2_FOLDER>:$PYTHONPATH
```

• Start IsoNet with isonet.py.

Verified environments:
cuda 11.8, cudnn 8.5, PyTorch 2.0.1

cuda 11.3, cudnn 8.2, PyTorch 1.13.1.README.md

## 2. Examples

The following two examples describe the basic steps of running the program and generating corrected tomograms. The dataset and a video tutorial can be found here https://drive.google.com/drive/folders/1DXjIsz6-EiQm7mBMuMHHwdZErZ_bXAgp

### 2.1 Ribosome dataset with EVN/ODD split

This Ribosome dataset contains 5 tomograms (1,2,3,4,5) from EMPIAR-10985 with EVN/ODD split.

![fig1](figures/figxx.png)

#### 2.1.1 Prepare Tomograms and Starfile

#### 2.1.2 Refine

#### 2.1.3. Predict

#### 2.1.4. Optionally make mask

This is useful if tomgoram has sparse sample and we can generated masked from correctedtomog or denoised tomograms

#### 2.1.5. Optionally denoise

First, create a folder for your project. In this folder, create subfolder (in this case subfolder name is tomograms/EVN and tomgrams/ODD) and move all tomogram (with suffix .mrc or .rec) to the corresponding subfolder.

```{.bash language="bash"}
mkdir tomograms
mkdir tomograms/EVN
mkdir tomograms/ODD
mv *EVN.mrc tomograms/EVN
mv *ODD.mrc tomograms/ODD
```

**_edit this_**
Then run the following command in your project folder to generate a
tomogram.star file. The number_subtomos defined how many tomograms are used for training in each epoch for a certain tomogram. Typically number_subtomos per tomograms times number of tomograms times number of epochs to train is about 100000 for adequate training. For example, for 500 number_subtomos and 4 tomograms, we recommend training to 50 epochs (500x4x50 == 100000)

```{.bash language="bash"}
isonet.py prepare_star -e tomograms/EVN -o tomograms/ODD --output_star tomograms.star --pixelsize <value> --number_subtomos 500
```

Please use your favorite text editor, such as vi or gedit, to open the **tomograms.star**, and enter one defocus value for each tomogram in the rlnDefocus column. This value should be the approximate defocus value calculated for the 0-degree tilt images in **Å**. Also modify the min and max tilt angles in the corresponding colums.

After editing, your star file should look as follows. Note, defocus value is only for CTF deconvolution/correction, if you want to skip CTF relevant correction or the tomograms are acquired with a phase plate, leave this column as default 0.

#### 2.1.2 Refine

Train IsoNet to reconstruct missing wedge and denoise subtomograms:

```{.bash language="bash"}
isonet.py refine tomograms.star --outputdir <results_folder> --gpuID <ids> --epochs <number>
```

#### 2.5. Predict

Apply the trained model to the original tomograms to recover missing wedge regions:

```{.bash language="bash"}
isonet.py predict tomograms.star <model_file.h5> --gpuID <ids> --outputdir <output_folder>
```

### 2.2 HIV dataset with full tomograms as input

#### 2.1.1 Prepare Star

Collect all tomogram files (.mrc or .rec) into a folder.

Generate a STAR file:

```{.bash language="bash"}
isonet.py prepare_star <folder> --outputstar <tomograms.star> --pixelsize <value> --numbersubtomos <number>
```

Manually edit with Vim or other editors:

IMAGE

Add defocus for each tomogram in the fourth column (in Å). Adjust rlnNumberSubtomo for each tomogram as needed.

#### 2.1.2 Deconv

Perform CTF deconvolution (optional, skip for phase plate tomograms or refinements using network-based deconvolution):
isonet.py deconv <tomograms.star> --deconvfolder <output_folder> --snrfalloff <value> --deconvstrength <value>
Other options: --highpassnyquist, --chunksize, --overlap_rate, --ncpu
2.3. Make Mask
Generate a mask for each tomogram to exclude empty regions:
isonet.py make_mask <tomograms.star> --maskfolder <output_folder> --patchsize <value> --densitypercentage <val> --stdpercentage <val> --zcrop <val>

2.4 Refine
Train IsoNet to reconstruct missing wedge and denoise subtomograms:
isonet.py refine <subtomo.star> --outputdir <results_folder> --gpuID <ids> --iterations <number> --noiselevel <list> --noisestartiter <list> --method <isonet2-n2n/isonet2> --arch <network type>
2.5. Predict
Apply the trained model to the original tomograms to recover missing wedge regions:
isonet.py predict <tomograms.star> <model_file.h5> --gpuID <ids> --outputdir <output_folder> --inputcolumn <column name>

## 3. Example (EMPIAR Ribosome Dataset)

dataset: https://www.ebi.ac.uk/empiar/EMPIAR-10985/

### 3.1 Ribosome tomograms split into even and odd frames

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

4. IsoNet commands

prepare_star
Generate a tomograms.star file that lists tomogram file paths and acquisition metadata used by all downstream IsoNet commands. The function can accept either a single set of full tomograms or paired even/odd half tomograms for noise2noise workflows.
Key parameters
full — Directory with full tomogram files; use for single-map training (isonet2).
even — Directory with even-half tomograms; use with odd for noise2noise (isonet2-n2n).
odd — Directory with odd-half tomograms; used together with even.
tilt_min — Minimum tilt angle in degrees; default -60. Override if your tilt range is different.
tilt_max — Maximum tilt angle in degrees; default 60. Override if your tilt range is different.
tilt_step — Tilt step size in degrees; default 3.
pixel_size — Pixel size (Å). Defaults to "auto" (reads from tomograms) but you can set a target value (commonly ~10 Å for typical IsoNet runs).
create_average — If True and no full provided, create full tomograms by summing the provided even and odd folders; useful for producing a single full tomogram from two halves.
number_subtomos — Number of subtomograms to extract per tomogram (written to rlnNumberSubtomo). For IsoNet2, increasing this is analogous to increasing training exposure and can improve results at the cost of runtime and memory.
mask_folder — Optional directory with masks; entries are recorded in rlnMaskName.
coordinate_folder — Optional directory with subtomogram coordinate files; if provided, the number of subtomograms is taken from the coordinate files and overrides number_subtomos.
cs, voltage, ac, rlnDefocus — Microscope parameters (spherical aberration mm, acceleration voltage kV, amplitude contrast, defocus in the STAR default units). Set only if different from defaults.
Practical notes
Use even and odd when you plan noise2noise training; use full for single-map training.
If tilt range differs from ±60°, supply tilt_min and tilt_max so the code records the correct missing-wedge geometry.
Inspect and edit the generated STAR if you need tomogram-specific subtomogram counts or different mask/defocus entries.

deconv
CTF deconvolution preprocessing that enhances low-resolution contrast and recovers information attenuated by the microscope contrast transfer function. Recommended for non–phase-plate data; skip for phase-plate data or if intending to use network-based CTF deconvolution.
Key parameters
star_file — Input STAR listing tomograms and acquisition metadata.
input_column — STAR column used for input tomogram paths (default rlnTomoName).
output_dir — Folder to write deconvolved tomograms (rlnDeconvTomoName entries point here).
snrfalloff — Controls frequency-dependent SNR attenuation applied during deconvolution; default 1.0. Larger values reduce high-frequency contribution more aggressively and can stabilize deconvolution on noisy data; smaller values preserve more high-frequency content but risk amplifying noise.
deconvstrength — Scalar multiplier for deconvolution strength; default 1.0. Increasing this emphasizes correction and low-frequency recovery but can introduce ringing/artifacts if set too high.
highpassnyquist — Fraction of the Nyquist used as a very-low-frequency high-pass cutoff; default 0.02. Use to remove large-scale intensity gradients and drift; usually left at default.
chunk_size — If set, tomograms are processed in smaller cubic chunks to reduce memory usage. Useful for very large tomograms or limited RAM/VRAM. May create edge artifacts if chunks are too small.
overlap_rate — Fractional overlap between adjacent chunks when chunking (default 0.25). Larger overlaps reduce edge artifacts at cost of extra computation.
ncpus — Number of CPU workers for CPU-bound parts of deconvolution; increase on multi-core systems.
phaseflipped — If True, input is assumed already phase-flipped; otherwise the function uses defocus and CTF info to apply phase handling.
Practical notes
Inspect deconvolved outputs visually for ringing or other artifacts after changing snrfalloff or deconvstrength.
Use chunking plus a moderate overlap_rate (0.25–0.5) when memory is limited.

make_mask
Generate masks to prioritize regions of interest. Masks improve sampling efficiency and training stability.
Key parameters
star_file — Input STAR listing tomograms.
input_column — STAR column to read tomograms from (default rlnDeconvTomoName; falls back to rlnTomoName if absent).
output_dir — Folder to save mask MRCs; rlnMaskName is updated in the STAR.
patch_size — Local patch size used for max/std local filters (default 4). Larger values smooth detection of specimen regions; default works for typical pixel sizes.
density_percentage — Percentage of voxels retained based on local density ranking (default 50). Lower values create stricter masks (keep fewer voxels).
std_percentage — Percentage retained based on local standard-deviation ranking (default 50). Lower values emphasize textured regions.
z_crop — Fraction of tomogram Z to crop from both ends (default 0.2 masks out top and bottom 10% each). Use to avoid sampling low-quality reconstruction edges.
tomo_idx — Limit mask generation to a subset of STAR entries (e.g., "1,3,5-7").
Practical notes
Defaults are suitable for most datasets; tune density/std percentages for very sparse specimens or dense, crowded volumes.
If automatic masks miss specimen regions, edit boundaries in the STAR or provide manual masks.

denoise and refine
Both functions are training entry points. Use denoise for pure noise-to-noise (n2n) training workflows; use refine for IsoNet2 missing-wedge correction (IsoNet2) or IsoNet2-n2n combined modes. Many parameters are shared between them.
Key parameters
method (refine only) — "isonet2" for single-map missing-wedge correction, "isonet2-n2n" for noise2noise when even/odd halves are present. If omitted, the code auto-detects the method from the STAR columns.
arch — Network architecture string (e.g., unet-small, unet-medium, unet-large, HSFormer, vtunet). Determines model capacity and VRAM requirements.
cube_size — Size in voxels of training subvolumes (default 96). Must be compatible with the network (divisible by the network downsampling factors).
epochs — Number of training epochs; longer for harder recovery, but watch validation.
batch_size — Number of subtomograms per optimization step; if None, this is automatically derived from the available GPUs. Batch size per GPU matters for gradient stability.
acc_batches — Number of gradient accumulation steps to emulate larger batches when memory is limited.
mixed_precision — If True, uses float16/mixed precision to reduce VRAM and speed up training.
CTF_mode — CTF handling mode: "None", "phase_only", "wiener", or "network". Choose according to your reconstruction pipeline and whether you want the network to handle CTF effects.
"network" applies a CTF-shaped filter to network inputs so the network learns to restore it.
"wiener" configures the pipeline to emulate a Wiener-filtered target.
clip_first_peak_mode — Controls attenuation of overrepresented very-low-frequency CTF peak: 0 none, 1 constant clip, 2 negative sine, 3 cosine.
bfactor — B-factor applied during training/prediction to boost high-frequency content; ideal values around 200 - 500.
noise_level — For plain isonet2 (non-n2n) training, supply >0 to enable denoising capability (adds synthetic noise during training). For noise2noise (isonet2-n2n or denoise), this is typically not required.
noise_mode — Controls filter applied when generating synthetic noise (used with noise_level).
split_halves — If True, train separate top/bottom or even/odd networks (DuoNet) and save separate checkpoints.
snrfalloff, deconvstrength, highpassnyquist — parameters for CTF deconvolution
forwarded to deconvolution if you refine with --with_deconv True
used to calculate Wiener filter for network-based deconvolution
with_deconv — If True (refine only), run deconvolution and mask creation before training and set input_column to the deconvolved tomograms.
with_predict — If True, run prediction using the final checkpoint(s) after training.
Practical notes
Choose arch, cube_size, and batch_size to fit your GPU memory; larger architectures and cubes improve fidelity but increase resource needs.
Enable mixed_precision to save VRAM and speed up training if your GPU and drivers support it.
Use split_halves to obtain gold-standard half-map workflows (independent models for even/odd).
If you request deconvolution within refine, tune snrfalloff and deconvstrength there to avoid overcorrection artifacts.

predict
Apply a trained IsoNet model to tomograms to produce denoised or missing-wedge–corrected volumes. Prediction utilizes the model's saved cube size and CTF handling options, but allows for runtime adjustments.
Key parameters
star_file — Input STAR describing tomograms to predict.
model — Path to trained model checkpoint (.pt) for single-model prediction.
model2 — Optional second checkpoint for DuoNet (dual-model) prediction.
output_dir — Folder to save predicted tomograms; outputs are recorded in the STAR as rlnCorrectedTomoName or rlnDenoisedTomoName depending on method.
gpuID — GPU IDs string (e.g., "0" or "0,1"); use multiple GPUs when available for speed.
input_column — STAR column used for input tomogram paths (default rlnDeconvTomoName).
apply_mw_x1 — If True (default), build and apply the missing-wedge mask to cubic inputs before prediction.
phaseflipped — Declare if input tomograms are already phase-flipped; affects CTF handling.
do_phaseflip_input — Whether to apply phase-flip handling to inputs for CTF-aware modes (default True).
padding_factor — Cubic padding factor used during tiling to reduce edge effects (default 1.5); larger padding reduces seams but increases computation.
tomo_idx — Process a subset of STAR entries by index.
Practical notes
Match prediction cube/crop sizes and padding to the network’s training settings (these come from the model object).
When using CTF-aware models, ensure phaseflipped and STAR defocus/CTF fields are correct.

```

```
