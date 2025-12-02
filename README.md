# IsoNet2 Tutorial

**IsoNet2** is a deep-learning software package for simultaneous missing wedge correction, denoising, and CTF correction in cryo-electron tomography reconstructions using a deep neural network trained on information from the original tomogram(s). Compared to IsoNet1, IsoNet2 produces tomograms with higher resolution and less noise in roughly a tenth of the time. The software requires tomograms as input. Paired tomograms for Noise2Noise training can be split by either frame or tilt.

**IsoNet2** contains six modules: **prepare star**, **CTF deconvolve**, **generate mask**, **denoise**, **refine**, and **predict**. All commands in IsoNet operate on **.star** text files which record paths of data and relevant parameters. For detailed descriptions of each module please refer to the individual tasks. Users can choose to utilize IsoNet through either GUI or command-lines.

# 1. Installation and System Requirements
The following tutorial is written for assuming absolutely no experience with Anaconda or Linux environments.


Software Requirements: This Linux installation of IsoNet requires CUDA Version >= 12.0 and Conda.


Hardware Requirements: ***Nvidia GTX 1080Ti or newer, with at least 8 GB VRAM ?***


## [Installing Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
+ Download the installer:

  + [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installer for Linux.

  + [Anaconda Distribution](https://www.anaconda.com/download) installer for Linux.

  + [Miniforge](https://conda-forge.org/download/) installer for Linux.

+ Verify your installer [hashes](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#hash-verification).

+ In your terminal window, run `bash <conda-installer-name>-latest-Linux-x86_64.sh`

  `conda-installer-name` will be one of "Miniconda3", "Anaconda", or "Miniforge3".

+ Follow the prompts on the installer screens. If you are unsure about any setting, accept the defaults. You can change them later.

+ Close and then re-open your terminal window.

+ After successful installation, running the command conda list should display a list of installed packages.

## [Installing CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
+ Find information on your GPU for [Linux](https://itsfoss.com/check-graphics-card-linux/).

+ Look up your GPU to make sure it is supported. If its [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) is below ***XXX***, you will not be able to run IsoNet2.

+ Make sure you have the latest graphics card driver version for your GPU from the [NVIDIA driver page](https://www.nvidia.com/en-us/drivers/).

+ Reference your Toolkit Driver Version in Table 3 on the [CUDA Toolkit Docs.](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) to see that it has a corresponding `CUDA Version >= 12.0 (Toolkit Driver Version >=525.60.13)`.

+ Select the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) version that matches what you read in Table 3 and follow the install instructions.

After successfully installing, you can check your `CUDA` version using `nvidia-smi`, which should produce something similar to below:
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
## Installing IsoNet2

Clone this repository:
```
git clone https://github.com/procyontao/IsoNet2.git
cd IsoNet2
```

Run `bash install.sh`. This creates an Conda environment (installing requirements) using the included **isonet2_environment.yml** file and updates your environment variables (allowing you to call isonet.py) by running `source isonet2.bashrc`. You may append this command to your .bashrc file so you don't have to re-source it every time you wish to use IsoNet.


Installation should take 5-10 minutes. Upon successful installation, running the command `isonet.py --help` should display the following help message.
```
INFO: Showing help with the command 'isonet.py -- --help'.

NAME
    isonet.py - ISONET: Train on tomograms and restore missing-wedge.

...
```
# 2. Tutorial

The following outlines the example IsoNet2 GUI and command-line workflow in Linux. More in-depth explanations of each parameter can be found under ***4. IsoNet Modules.*** A video tutorial can be found in the following [Google Drive.](https://drive.google.com/drive/u/1/folders/1P9sxSSJIWPs7hIGey3I38u3B2OvmiCAC)

This [dataset](https://drive.google.com/drive/folders/1JQb5YBr83JM9FWFrxfe0mP8fTgXKJU00?usp=sharing) contains 5 HIV tomograms from EMPIAR-10164 with frame-based EVN/ODD split.

## 2.0 Download Tomograms

In a new working directory, download the **tomograms_split** folder. This contains even and odd half tomograms.

Output for `ls -1 tomograms_split/EVN`:
```
TS_01_EVN.mrc
TS_03_EVN.mrc
TS_43_EVN.mrc
TS_45_EVN.mrc
TS_54_EVN.mrc
```
Output for `ls -1 tomograms_split/ODD`:
```
TS_01_ODD.mrc
TS_03_ODD.mrc
TS_43_ODD.mrc
TS_45_ODD.mrc
TS_54_ODD.mrc
```

![](./IsoNet/tutorial/figures/Fig1.png)

## 2.1 GUI

The IsoNet2 GUI provides intuitive, detailed, and organized process management, making it ideal for running multiple jobs in queue while maintaining granular control of each step of the process.

### 2.1.0 Launch GUI

Launch the GUI by entering `IsoNet2 -no-sandbox &` into your terminal.

### 2.1.1 Prepare Star

Open the **Prepare** tab and select **Even/Odd Input**:

![](./IsoNet/tutorial/figures/GUI/01OpenPrepare.png)

Identify your **even** and **odd** data directories. 

![](./IsoNet/tutorial/figures/GUI/02SelectEvenOdd.png)

Disable **create average** for Noise2Noise training. Enable this to create a full tomogram from two half tomograms for non-Noise2Noise training. Adjust the **pixel size in Å** and **number subtomograms per tomo** as needed. For the purpose of this quick tutorial, we will decrease the number of subtomograms. The other parameters are related to your physical electron microscope and are used later for network-based deconvolution. They are left at default for this tutorial, as are the **tilt min** and **tilt max**. **Show command** provides the `isonet.py` command if you prefer to run it directly in your terminal.

![](./IsoNet/tutorial/figures/GUI/03ModifyParameters.png)

**Run** your job. The starfile should automatically display. If you ran the command in your terminal, **load the starfile** from your working directory. Adjust the **rlnDefocus** column with the approximate defocus in Å at 0° for each tomogram.

![](./IsoNet/tutorial/figures/GUI/04ModifyDefocus.png)

### 2.1.2 Create Mask

Open the **Create Mask** tab and select your **Input Column**. For even/odd split datasets, choose **rlnTomoReconstructedTomogramHalf1**

![](./IsoNet/tutorial/figures/GUI/05OpenMask.png)

**Submit** your job. Clicking on your job in the new column will display your progress. This output is saved to **./make_mask/jobID/log.txt**.

![](./IsoNet/tutorial/figures/GUI/06RunMask.png)

### 2.1.3 Refine

Open the **Refine** tab. Keep **Even/Odd Input** enabled and adjust **mw weight** and **gpuID** as needed. **mw weight** determines how heavily the network prioritizes missing wedge correction over denoising. Here the ratio is 200:1.

![](./IsoNet/tutorial/figures/GUI/07OpenRefine.png)

 Click **Submit (In Queue)** for this job. Clicking on your job in the new column will display your progress. This output is saved to **./refine/jobID/log.txt**. A graphical representation of the loss is saved to **./refine/jobID_refine/loss_full.png**

![](./IsoNet/tutorial/figures/GUI/08RunRefine.png)

To practice queuing jobs, open the **Refine** tab again, using the same parameters as before. Scroll down and select an arbitrary **CTF_mode** to differentiate it from the first job. Click **Submit (In Queue)** to queue your job. You will see a second marker appear in the refine column. 
> Clicking **Submit (Run Immediately)** will bypass the queue, causing both jobs to run simultaneously. This will increase training time for both jobs.

![](./IsoNet/tutorial/figures/GUI/09QueueRefine.png)

Open the **Jobs Viewer** tab. Here we can see the our jobs' statuses and IDs. You can remove the second job.

![](./IsoNet/tutorial/figures/GUI/10JobsViewer.png)

### 2.1.4 Predict

After training is complete, open the **Predict** tab. Adjust **gpuID** as needed. Specifying a tomo index will limit prediction to the corresponding tomograms. We will leave it as 'all' here.

![](./IsoNet/tutorial/figures/GUI/11OpenPredict.png)

Select the completed model file from your **./refine/jobID_refine** directory. **Submit** your job. Clicking on your job in the new column will display your progress. This output is saved to **./predict/jobID/log.txt**.

![](./IsoNet/tutorial/figures/GUI/12SelectModel.png)

Wait for prediction to finish. View your corrected tomograms in **./predict/jobID_predict**.

![](./IsoNet/tutorial/figures/Fig2.png)

## 2.2 Command Line

The IsoNet2 CLI requires more hands-on data management and familiarity with parameters, but may be ideal for rapid preliminary fine-tuning of different parameters in the pipeline. 

### 2.2.1 Prepare Star
Prepare the starfile.
You may enter a single defocus value to be used for every tomogram or a list of values to be applied to their respective tomograms. You may also use your default text editor or the GUI to open **tomograms.star** and manually enter the defocus values.
```
isonet.py prepare_star --even tomograms_split/EVN --odd tomograms_split/ODD --star_name tomograms.star --pixel_size 5.4 --defocus "[39057,14817,25241,29776,15463]" --number_subtomos 200
```

Output for `cat tomograms.star`:

```
# Created by the starfile Python package (version 0.5.6) at 13:43:23 on 17/11/2025


data_

loop_
_rlnIndex #1

...

5       None    tomograms_split/EVN/TS_54_EVN.mrc       tomograms_split/ODD/TS_54_ODD.mrc       5.400000     15463   300     2.700000        0.100000        None    None    -60     60      None    200 None

```
### 2.2.2 Refine

Train IsoNet to reconstruct missing wedge and denoise subtomograms. **with_mask** automatically generates the masks for each tomogram before refinement. **mw_weight** determines how heavily the network prioritizes missing wedge correction over denoising. Here the ratio is 200:1. We recommend directing the output to a logfile so you can review the loss values, but this can also be done graphically by viewing **./isonet_maps/loss_full.png**.

```
isonet.py refine tomograms.star --with_mask True --gpuID <ids> --mw_weight 200 > refine_log.txt
```

### 2.2.3. Predict

After training, apply the trained model to the original tomograms to recover missing wedge regions:

```
isonet.py predict tomograms.star isonet_maps/network_isonet2-n2n_unet-medium_96_full.pt --gpuID <ids>
```

# 3. FAQs
## Q: When should I use even/odd split versus full tomograms?
Use even/odd split tomograms when you want to perform Noise2Noise (n2n) training, which is generally recommended as it provides better denoising. Use full tomograms for single-map training (isonet2 method) when paired halves are not available, as with older datasets.
## Q: How do I determine the correct defocus values for my tomograms?
You need to provide the approximate defocus in Ångströms at 0° tilt for each tomogram. These values come from your data collection software or can be estimated from CTF fitting.
## Q: How many subtomograms should I extract per tomogram/epochs should I train for?
The default is 1000 subtomograms per tomogram per epoch. Increasing this value is analogous to increasing the number of training epochs, as subtomograms are extracted during training (as opposed to before, in IsoNet1). Because IsoNet2 does not currently use a specialized learning rate scheduler, it is okay to keep the default and simply halt training when the loss has converged.
## Q: How can I reduce memory usage during training?
+ Enable mixed_precision for float16 training
+ Reduce batch_size
+ Choose a smaller network architecture
+ Reduce cube_size
+ Use chunk_size with overlap_rate for processing large tomograms
## Q: When should I use CTF deconvolution?
CTF deconvolution preprocessing is recommended for non-phase-plate data to enhance low-resolution contrast. Skip this step for phase-plate data or if you plan to use network-based CTF deconvolution during training.
## Q: How do I know if my deconvolution settings are too aggressive?
A: Inspect deconvolved outputs visually for ringing artifacts or excessive noise amplification. If present, reduce deconvstrength or increase snrfalloff to be more conservative.
## Q: When should I create masks?
Masks prioritize regions of interest (specimen areas) during training, which improves sampling efficiency and training stability by focusing the network on relevant areas rather than empty space. **We recommend always creating a mask.**
## Q: My masks are missing specimen regions. What can I do?
A: You can manually edit the mask boundaries in the .star file, adjust the density_percentage and std_percentage parameters to be less strict (higher values), or provide your own manual masks through the mask_folder parameter.

# 4. IsoNet Modules

## prepare_star
Generate a tomograms.star file that lists tomogram file paths and acquisition metadata used by all downstream IsoNet commands. The function can accept either a single set of full tomograms or paired even/odd half tomograms for noise2noise workflows.

### Key parameters

+ full — Directory with full tomogram files; use for single-map training (isonet2).

+ even — Directory with even-half tomograms; use with odd for noise2noise (isonet2-n2n).

+ odd — Directory with odd-half tomograms; used together with even.

+ mask_folder — Optional directory with masks; entries are recorded in rlnMaskName.

+ coordinate_folder — Optional directory with subtomogram coordinate files; if provided, the number of subtomograms is taken from the coordinate files and overrides number_subtomos.

+ star_name — Name of starfile.

+ pixel_size — Pixel size (Å). Defaults to "auto" (reads from tomograms) but you can set a target value (commonly ~10 Å for typical IsoNet runs).

+ cs, voltage, ac, rlnDefocus — Microscope parameters (spherical aberration mm, acceleration voltage kV, amplitude contrast, defocus in the STAR default units). Set only if different from defaults.

+ tilt_min — Minimum tilt angle in degrees; default **-60**. Override if your tilt range is different.

+ tilt_max — Maximum tilt angle in degrees; default **60**. Override if your tilt range is different.

+ tilt_step — Tilt step size in degrees; default **3**.

+ create_average — If True and no full provided, create full tomograms by summing the provided even and odd folders; useful for producing a single full tomogram from two halves.

+ number_subtomos — Number of subtomograms to extract per tomogram (written to rlnNumberSubtomo). For IsoNet2, increasing this is analogous to increasing training exposure and can improve results at the cost of runtime and memory.


### Practical notes
> Use even and odd when you plan to use noise2noise training; use full for single-map training.
If tilt range differs from ±60°, supply tilt_min and tilt_max so the code records the correct missing-wedge geometry.
Inspect and edit the generated STAR if you need tomogram-specific subtomogram counts or have pregenerated mask/defocus entries.

## deconv
CTF deconvolution preprocessing that enhances low-resolution contrast and recovers information attenuated by the microscope contrast transfer function. Recommended for non–phase-plate data; skip for phase-plate data or if intending to use network-based CTF deconvolution.

### Key parameters
+ star_file — Input STAR listing tomograms and acquisition metadata.

+ output_dir — Folder to write deconvolved tomograms (rlnDeconvTomoName entries point here).

+ input_column — STAR column used for input tomogram paths (default **rlnTomoName**).

+ snrfalloff — Controls frequency-dependent SNR attenuation applied during deconvolution; default **1.0**. Larger values reduce high-frequency contribution more aggressively and can stabilize deconvolution on noisy data; smaller values preserve more high-frequency content but risk amplifying noise.

+ deconvstrength — Scalar multiplier for deconvolution strength; default **1.0**. Increasing this emphasizes correction and low-frequency recovery but can introduce ringing/artifacts if set too high.

+ highpassnyquist — Fraction of the Nyquist used as a very-low-frequency high-pass cutoff; default **0.02**. Use to remove large-scale intensity gradients and drift; usually left at default.

+ chunk_size — If set, tomograms are processed in smaller cubic chunks to reduce memory usage. Useful for very large tomograms or limited RAM/VRAM. May create edge artifacts if chunks are too small.

+ overlap_rate — Fractional overlap between adjacent chunks when chunking (default **0.25**). Larger overlaps reduce edge artifacts at cost of extra computation.

+ ncpus — Number of CPU workers for CPU-bound parts of deconvolution; increase on multi-core systems.

+ phaseflipped — If True, input is assumed already phase-flipped; otherwise the function uses defocus and CTF info to apply phase handling.

+ tomo_idx: If set, process only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").

### Practical notes
> Inspect deconvolved outputs visually for ringing or other artifacts after changing snrfalloff or deconvstrength.
Use chunking plus a moderate overlap_rate (0.25–0.5) when memory is limited.

## make_mask
Generate masks to prioritize regions of interest. Masks improve sampling efficiency and training stability.
### Key parameters
+ star_file — Input STAR listing tomograms.

+ input_column — STAR column to read tomograms from (default **rlnDeconvTomoName**; falls back to **rlnTomoName** or **rlnTomoReconstructedTomogramHalf1** if absent).

+ output_dir — Folder to save mask MRCs; rlnMaskName is updated in the STAR.

+ patch_size — Local patch size used for max/std local filters (default **4**). Larger values smooth detection of specimen regions; default works for typical pixel sizes.

+ density_percentage — Percentage of voxels retained based on local density ranking (default **50**). Lower values create stricter masks (keep fewer voxels).

+ std_percentage — Percentage retained based on local standard-deviation ranking (default **50**). Lower values emphasize textured regions.

+ z_crop — Fraction of tomogram Z to crop from both ends (default **0.2** masks out top and bottom 10% each). Use to avoid sampling low-quality reconstruction edges.

+ tomo_idx — If set, process only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").

### Practical notes
> Defaults are suitable for most datasets; tune density/std percentages for very sparse specimens or dense, crowded volumes.
If automatic masks miss specimen regions, edit boundaries in the STAR or provide manual masks.

## denoise and refine
Both functions are training entry points. Use denoise for pure noise-to-noise (n2n) training workflows; use refine for IsoNet2 missing-wedge correction (IsoNet2) or IsoNet2-n2n combined modes. Many parameters are shared between them.

### Key parameters

+ method (*refine only*) — "isonet2" for single-map missing-wedge correction, "isonet2-n2n" for noise2noise when even/odd halves are present. If omitted, the code auto-detects the method from the STAR columns.

+ input_column (*refine only*) — Column name in star file to use as input tomograms.

+ mw_weight (*refine only*) — Weight for missing wedge loss. Higher values correspond to stronger emphasis on missing 
wedge regions. Disabled by default.

+ apply_mw_x1 (*refine only*) — Whether to apply missing wedge to subtomograms at the beginning.

+ clip_first_peak_mode (*refine only*) — Controls attenuation of overrepresented very-low-frequency CTF peak: 
  + 0 none
  + 1 constant clip
  + 2 negative sine
  + 3 cosine

+ noise_level (*refine single-map only*) — Adds artificial noise during training.

+ noise_mode (*refine single-map only*) — Controls filter applied when generating synthetic noise (None, ramp, hamming).

+ random_rot_weight (*refine only*) — Percentage of rotations applied as random augmentation.

+ with_deconv (*refine only*) — Whether to deconvolve automatically.

+ with_mask (*refine only*) — Whether to generate masks automatically.

+ num_mask_updates (*refine only*) — Number of times to update masks based on predicted tomograms during training. Increases training time significantly.
        
+ star_file — Star file for tomograms.

+ output_dir — Directory to save trained model and results.

+ gpuID — GPU IDs to use during training (e.g., "0,1,2,3").

+ ncpus — Number of CPUs to use for data processing.

+ arch — Network architecture string (e.g., unet-small, unet-medium, unet-large, HSFormer, vtunet). Determines model capacity and VRAM requirements.

+ pretrained_model — Path to pretrained model to continue training. Previous method, arch, cube_size, CTF_mode, and metrics will be loaded.
            
+ cube_size — Size in voxels of training subvolumes (default **96**). Must be compatible with the network (divisible by the network downsampling factors).

+ epochs — Number of training epochs.

+ batch_size — Number of subtomograms per optimization step; if None, this is automatically derived from the available GPUs. Batch size per GPU matters for gradient stability.

+ loss_func — Loss function to use (L2, Huber, L1).

+ learning_rate — Initial learning rate.

+ save_interval — Interval to save model checkpoints. Default is epochs/10.

+ learning_rate_min — Minimum learning rate for scheduler.

+ mixed_precision — If True, uses float16/mixed precision to reduce VRAM and speed up training.

+ CTF_mode — CTF handling mode: "None", "phase_only", "wiener", or "network".
  + "None": No CTF correction
  + "phase_only": Phase-only correction
  + "network": Applies CTF-shaped filter to network input
  + "wiener": Applies Wiener filter to network target

+ bfactor — B-factor applied during training/prediction to boost high-frequency content; ideal values around 200 - 500.

+ isCTFflipped — Whether input tomograms are phase flipped.

+ do_phaseflip_input — Whether to apply phase flip during training.
            
+ with_predict — If True, run prediction using the final checkpoint(s) after training.

+ pred_tomo_idx — If set, automatically predict only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").
            
+ snrfalloff, deconvstrength, highpassnyquist — parameters for CTF deconvolution 
  + forwarded to deconvolution for `--with_deconv True`
  + used to calculate Wiener filter for network-based deconvolution

### Practical notes
> Choose arch, cube_size, and batch_size to fit your GPU memory; larger architectures and cubes improve fidelity but increase resource needs.
Enable mixed_precision to save VRAM and speed up training if your GPU and drivers support it.

## predict

Apply a trained IsoNet model to tomograms to produce denoised or missing-wedge–corrected volumes. Prediction utilizes the model's saved cube size and CTF handling options, but allows for runtime adjustments.

### Key parameters
+ star_file — Input STAR describing tomograms to predict.

+ model — Path to trained model checkpoint (.pt) for single-model prediction.

+ output_dir — Folder to save predicted tomograms; outputs are recorded in the STAR as 
rlnCorrectedTomoName or rlnDenoisedTomoName depending on method.

+ gpuID — GPU IDs string (e.g., "0" or "0,1"); use multiple GPUs when available for speed.

+ input_column — STAR column used for input tomogram paths (default **rlnDeconvTomoName**).

+ apply_mw_x1 — If True (default), build and apply the missing-wedge mask to cubic inputs before prediction.

+ isCTFflipped — Declare if input tomograms are already phase-flipped; affects CTF handling.

+ padding_factor — Cubic padding factor used during tiling to reduce edge effects (default **1.5**); larger padding reduces seams but increases computation.

+ tomo_idx — Process a subset of STAR entries by index.

+ output_prefix — Prefix to append to predicted MRC files.

### Practical notes
> Match prediction cube/crop sizes and padding to the network’s training settings (these come from the model object).
When using CTF-aware models, ensure phaseflipped and STAR defocus/CTF fields are correct.