# Changelog

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
