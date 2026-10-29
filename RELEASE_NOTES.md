# IsoNet2 v2.1.0

This release includes IsoNet2 source code and the current Linux x86-64 GUI, built from the same release tag. The GUI is distributed as `isoapp.AppImage` and follows the IsoNet2 release version.

## Downloads and installation

- Download **Source code (zip)** or **Source code (tar.gz)** and extract it.
- Download **isoapp.AppImage** and place it in the extracted IsoNet2 directory.
- Download **isoapp.AppImage.sha256** to verify the GUI download with `sha256sum -c isoapp.AppImage.sha256`.
- Run `bash install.sh` from the source directory to create the Conda environment and install IsoNet2. The installer downloads this release's GUI into `build/`.
- Run `source isonet2.bashrc`, then use `IsoNet2` to launch the GUI or `isonet.py -h` for the CLI.

The AppImage provides the GUI; Python, CUDA, and the IsoNet2 backend still require the environment described in the README. Only a Linux x86-64 GUI binary is provided.

## Changes since v2.0.0-beta

- Integrated GUI source and packaging, improved STAR loading, and updated the tutorial.
- Added FSC loss and detailed documentation for architectures, losses, CTF modes, and training controls.
- Improved GPU selection, DDP process-group handling, and MRC voxel-size preservation.
- Made the averaged-tomogram output directory configurable.
- Combined paired-half inputs in model forward passes during refinement for consistent BatchNorm statistics.
- Added augmentation noise before CTF filtering in paired-half missing-wedge filling.
- Removed unused model implementations. SCUNet remains experimental, without a quality or stability guarantee, and is available only through the CLI.

## Changes to existing commands

**`CTF_mode=network` is now the default for refine and denoise**, including the GUI. We strongly recommend this mode for CTF correction. To disable explicit CTF correction, specify `--CTF_mode None`.

Recommended settings for network CTF correction are:

```bash
isonet.py refine tomograms.star --CTF_mode network --loss_func L2 --clip_first_peak_mode 1
isonet.py refine tomograms.star --CTF_mode network --loss_func FSC --clip_first_peak_mode 0
```

L2 with clip mode 1 is the default. Selecting FSC does not automatically change the clip mode; set mode 0 explicitly.

`apply_mw_x1` was unused during training and has been removed from refine and denoise, including their GUI forms. Remove it from existing training commands. It remains available in predict; automatic previews retain the enabled masking behavior.

Training behavior has changed since the beta release, including paired-half batching. Identical parameters do not imply identical training results. Experimental colored-noise augmentation is not included in this release.

Full change history: https://github.com/IsoNet-cryoET/IsoNet2/compare/IsoNet-2.0.0-beta...IsoNet-2.1.0
