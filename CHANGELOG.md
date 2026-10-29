# Changelog

## v2.1.0 — 2026-10-29

- Make `CTF_mode=network` the default for refine and denoise in both the CLI and GUI.
- Recommend L2 loss with `clip_first_peak_mode=1`, or FSC loss with `clip_first_peak_mode=0`, for network CTF correction.
- Add FSC loss and expand documentation for architectures, loss functions, CTF handling, and training settings.
- Batch paired-half inputs in the same forward pass during refinement for consistent BatchNorm statistics.
- Add augmentation noise before CTF filtering when filling the missing wedge during paired-half refinement.
- Remove the unused `apply_mw_x1` option from refine and denoise; retain it in predict.
- Make the averaged-tomogram output directory configurable and correct the tutorial checkpoint cube size.
- Improve GPU ID handling, DDP lifecycle management, MRC voxel-size preservation, and STAR loading.
- Remove unused model implementations and expose SCUNet as experimental CLI-only architectures.
- Ship the current integrated GUI as `isoapp.AppImage`, with a SHA256 checksum. The GUI follows the IsoNet2 release and has no separate public version.
- Add a tag-triggered workflow to build and publish release assets from the tagged source.

## Changes since 2025-12-12

### GUI Integration and packaging

- Integrated the IsoApp GUI into this repository under `IsoApp/`.
- Built and packaged the GUI successfully with `npm run build:linux`.
- Added a faster and safer STAR loading path in `IsoApp/src/main/handlers/other.js` and restored prepare-page refresh flow.

### Training and loss updates

- Added FSC loss support in `IsoNet/models/masked_loss.py` and wired it into training.
- Added a PyTorch CUDA kernel cache fallback in `IsoNet/__init__.py` to avoid cache-directory warnings.
- Updated training output and plots to display `n2n_loss` and `mw_loss` instead of `inside_loss` and `outside_loss`.

### Stability and behavior fixes

- Improved GPU ID validation and scheduler handling in the training workflow.
- Fixed MRC header voxel size preservation during I/O.
- Resolved DDP process group lifecycle issues during prediction.

### Miscellaneous updates

- Updated `.gitignore` to better ignore local, editor, and temporary files.
- The AppImage crash issue appears to be environment-specific; the packaged GUI build is successful from this repo.

## Notes

- The IsoApp GUI is integrated into this repository under `IsoApp/`, not a separate repository.
- Use `npm ci` followed by `npm run build:linux` in `IsoApp/` to reproduce the current GUI package.
