import numpy as np
import os
from torch.utils.data.dataset import Dataset
from IsoNet.utils.fileio import read_mrc
import starfile
import mrcfile
from tqdm import tqdm
import random
from IsoNet.utils.missing_wedge import mw3D

def normalize_percentage(volume, percentile=4, lower_bound = None, upper_bound=None):
    # original_shape = tensor.shape
    
    # batch_size = tensor.size(0)
    # flattened_tensor = tensor.reshape(1, -1)

    factor = percentile/100.
    # lower_bound_subtomo = np.quantile(volume, factor, dim=1, keepdim=True)
    # upper_bound_subtomo = np.quantile(volume, 1-factor, dim=1, keepdim=True)
    lower_bound_subtomo = np.percentile(volume,factor,axis=None,keepdims=True)
    upper_bound_subtomo = np.percentile(volume,1-factor,axis=None,keepdims=True)
    if lower_bound is None: 
        normalized_volume = (volume - lower_bound_subtomo) / (upper_bound_subtomo - lower_bound_subtomo)
    else:
        normalized_volume = (volume - lower_bound) / (upper_bound - lower_bound)
    
    return normalized_volume, lower_bound_subtomo, upper_bound_subtomo

# class Train_sets_regular(Dataset):
#     def __init__(self, paths, shuffle=True):
#         super(Train_sets_regular, self).__init__()
#         path_all = []
#         for dir in ["train_x", "train_y"]:
#             p = f"{paths}/{dir}/"
#             path_all.append(sorted([p+f for f in os.listdir(p)]))

#         zipped_path = list(map(list, zip(*path_all)))
#         if shuffle:
#             np.random.shuffle(zipped_path)
#         self.path_all = zipped_path

#     def __getitem__(self, idx):
#         results = []
#         for i,p in enumerate(self.path_all[idx]):
#             x, _ = read_mrc(p)
#             x = x[np.newaxis,:,:,:] 
#             x = torch.as_tensor(x.copy())
#             results.append(x)
#         return results
    
#     def __len__(self):
#         return len(self.path_all)



class Train_sets_n2n(Dataset):
    """
    Dataset class to load tomograms and provide subvolumes for n2n and spisonet methods.
    """

    def __init__(self, tomo_star, method="n2n", cube_size=64, input_column = "rlnTomoName", 
                 split="full", noise_dir=None,correct_between_tilts=False, start_bt_size=48,
                 snrfalloff=0, deconvstrength=1, highpassnyquist=0.02, clip_first_peak_mode=0, bfactor = 0):
        self.star = starfile.read(tomo_star)
        self.method = method
        self.n_tomos = len(self.star)
        self.input_column = input_column
        self.cube_size = cube_size
        self.split = split
        self.clip_first_peak_mode = clip_first_peak_mode

        self.n_samples_per_tomo = []
        self.tomo_paths_odd = []
        self.tomo_paths_even = []
        self.tomo_paths_gt = []

        self.tomo_paths = []
        self.coords = []
        self.mean = []
        self.std = []

        self.upperbounds = []
        self.lowerbounds = []

        self.mw_list = []
        self.wiener_list = []
        self.CTF_list = []
        self.start_bt_size=start_bt_size
        self.correct_between_tilts = correct_between_tilts
        self.snrfalloff=snrfalloff
        self.deconvstrength=deconvstrength
        self.highpassnyquist=highpassnyquist
        self.has_groundtruth = False
        self.bfactor = bfactor

        self._initialize_data()
        self.length = sum([coords.shape[0] for coords in self.coords])
        self.cumulative_samples = np.cumsum(self.n_samples_per_tomo)
        self.noise_dir = noise_dir
        if noise_dir != None:
            noise_files = os.listdir(noise_dir)
            self.noise_files = [os.path.join(noise_dir, file) for file in noise_files]

    def _initialize_data(self):
        """Initialize paths, mean, std, and coordinates from the starfile."""
        column_name_list = self.star.columns.tolist()

        # Initialize tqdm progress bar
        for _, row in tqdm(self.star.iterrows(), total=len(self.star), desc="Preprocess tomograms", ncols=100):
            if 'rlnGroundTruth' in row and row['rlnGroundTruth'] not in [None, "None"]:
                self.has_groundtruth = True
            mask = self._load_statistics_and_mask(row, column_name_list)
            if 'rlnBoxFile' not in row or row['rlnBoxFile'] in [None, "None"]:
                n_samples = row['rlnNumberSubtomo']
                if self.split in ["top", "bottom"]:
                    n_samples = n_samples // 2
                self.n_samples_per_tomo.append(n_samples)
                coords = self.create_random_coords(mask.shape, mask, n_samples)
            else:
                coords = np.loadtxt(row['rlnBoxFile'], dtype=int)[:, [2, 1, 0]]
                self.n_samples_per_tomo.append(len(coords))
            self.coords.append(coords)

            min_angle, max_angle, tilt_step = row['rlnTiltMin'], row['rlnTiltMax'], 3
            if not self.correct_between_tilts:
                tilt_step = None
            if tilt_step not in ["None", None]:
                start_dim = self.start_bt_size/tilt_step
            else:
                start_dim = 100000
            self.mw_list.append(self._compute_missing_wedge(self.cube_size, min_angle, max_angle, tilt_step, start_dim))
            CTF_vol, wiener_vol = self._compute_CTF_vol(row)
            self.wiener_list.append(wiener_vol)
            self.CTF_list.append(CTF_vol)


    def _load_statistics_and_mask(self, row, column_name_list):
        """Load tomogram data and corresponding mask."""
        even_column = 'rlnTomoReconstructedTomogramHalf1'
        odd_column = 'rlnTomoReconstructedTomogramHalf2'

        self.tomo_paths_even.append(row[even_column])
        if self.method in ['isonet2-n2n','n2n']:
            self.tomo_paths_odd.append(row[odd_column])
        if self.has_groundtruth:
            self.tomo_paths_gt.append(row['rlnGroundTruth'])

        # with mrcfile.mmap(row[even_column], mode='r', permissive=True) as tomo_even:
        #     tomo_shape = tomo_even.data.shape
        with mrcfile.mmap(row[even_column], mode='r', permissive=True) as tomo_even, \
             mrcfile.mmap(row[odd_column], mode='r', permissive=True) as tomo_odd:
            tomo_shape = tomo_even.data.shape
            Z = tomo_shape[0]

            # _, lower_evn, upper_evn = normalize_percentage(tomo_even.data)
            # _, lower_odd, upper_odd = normalize_percentage(tomo_odd.data)

            # _, upper_evn, lower_evn = normalize_percentage(tomo_even.data[Z//2-30:Z//2+30])
            # _, upper_odd, lower_odd = normalize_percentage(tomo_odd.data[Z//2-30:Z//2+30])

            mean = [np.mean(tomo_even.data[Z//2-30:Z//2+30]), np.mean(tomo_odd.data[Z//2-30:Z//2+30])]
            std = [np.std(tomo_even.data[Z//2-30:Z//2+30]), np.std(tomo_odd.data[Z//2-30:Z//2+30])]

        self.mean.append(mean)
        self.std.append(std)

        # self.upperbounds.append([upper_evn, upper_odd])
        # self.lowerbounds.append([lower_evn, lower_odd])
        if "rlnMaskName" not in column_name_list or row.get("rlnMaskName") in [None, "None"]:
            mask = np.ones(tomo_shape, dtype=np.float32)
        else:
            mask, _ = read_mrc(row["rlnMaskName"])
            mask = mask.copy()
        return mask

    def create_random_coords(self, shape, mask, n_samples):
        """
        Create random coordinates within permissible regions for subvolume extraction.
        """
        z_max, y_max, x_max = shape
        half_size = self.cube_size // 2
        
        mask[:half_size,:,:] = 0
        mask[z_max-half_size:z_max,:,:] = 0
        mask[:, :half_size, :] = 0
        mask[:, y_max-half_size:, :] = 0
        mask[:,:,:half_size] = 0
        mask[:,:,x_max-half_size:x_max] = 0

        half_y = y_max // 2
        if self.split == "top":
            mask[:,half_y:y_max,:] = 0
        elif self.split == "bottom":
            mask[:,0:half_y,:] = 0

        # Flatten the mask and randomly sample indices

        valid_indices = np.flatnonzero(mask)  # Get indices of non-zero elements
        if len(valid_indices) < n_samples:
            raise ValueError("Not enough valid positions in the mask to sample.")
        sampled_indices = valid_indices[np.random.randint(0, len(valid_indices), n_samples)]
        rand_coords = np.array(np.unravel_index(sampled_indices, shape)).T
        return rand_coords

        # valid_inds = np.where(mask)
        # sample_inds = np.random.choice(len(valid_inds[0]), n_samples, replace=(len(valid_inds[0]) < n_samples))
        # rand_inds = [v[sample_inds] for v in valid_inds]
        # return np.stack(rand_inds, -1)

    def _compute_missing_wedge(self, cube_size, min_angle, max_angle, tilt_step, start_dim):
        """Compute the missing wedge mask for given tilt angles."""
        from IsoNet.utils.missing_wedge import mw3D
        mw = mw3D(cube_size, missingAngle=[90 + min_angle, 90 - max_angle], tilt_step=tilt_step, start_dim=start_dim)
        return mw

    def _compute_CTF_vol(self, row):
        """Compute the missing wedge mask for given tilt angles."""
        # defocus in Anstron convert to um
        defocus = row['rlnDefocus']/10000.
        from IsoNet.utils.CTF import get_wiener_3d
        from IsoNet.utils.CTF_new import get_ctf3d
        ctf3d = get_ctf3d(angpix=row['rlnPixelSize'], voltage=row['rlnVoltage'], cs=row['rlnSphericalAberration'], defocus=defocus,\
                                    phaseshift=0, amplitude=row['rlnAmplitudeContrast'],bfactor=self.bfactor, \
                                        shape=[self.cube_size,self.cube_size,self.cube_size], clip_first_peak_mode=self.clip_first_peak_mode)
        wiener3d = get_wiener_3d(angpix=row['rlnPixelSize'], voltage=row['rlnVoltage'], cs=row['rlnSphericalAberration'], defocus=defocus,\
                                  snrfalloff=self.snrfalloff, deconvstrength=self.deconvstrength, highpassnyquist=self.highpassnyquist, \
                                    phaseflipped=False, phaseshift=0, amplitude=row['rlnAmplitudeContrast'], length=self.cube_size)
        return ctf3d, wiener3d

    def random_swap(self, x, y):
        if np.random.rand() > 0.5:
            return y, x
        return x, y

    def load_and_normalize(self, tomo_paths, tomo_index, z, y, x, eo_idx, invert=True):
        """Load and normalize a subvolume from a tomogram."""
        half_size = self.cube_size // 2
        with mrcfile.mmap(tomo_paths[tomo_index], mode='r', permissive=True) as tomo:
            subvolume = tomo.data[z-half_size:z+half_size, y-half_size:y+half_size, x-half_size:x+half_size]
        
        # if invert:
        #     #return 1 - (subvolume - self.lowerbounds[tomo_index][eo_idx]) / (self.upperbounds[tomo_index][eo_idx]- self.lowerbounds[tomo_index][eo_idx])
        #     return (self.upperbounds[tomo_index][eo_idx] - subvolume) / (self.upperbounds[tomo_index][eo_idx]- self.lowerbounds[tomo_index][eo_idx])

        # else:
        #     return (subvolume - self.lowerbounds[tomo_index][eo_idx]) / (self.upperbounds[tomo_index][eo_idx]- self.lowerbounds[tomo_index][eo_idx])
        if invert:
            return (self.mean[tomo_index][eo_idx] - subvolume) / self.std[tomo_index][eo_idx]
        else:
            return (subvolume - self.mean[tomo_index][eo_idx]) / self.std[tomo_index][eo_idx]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return a sample of data at a given index."""
        #tomo_index, coord_index = divmod(idx, self.n_samples_per_tomo)
        tomo_index = np.searchsorted(self.cumulative_samples, idx, side='right')
        coord_index = idx - (self.cumulative_samples[tomo_index - 1] if tomo_index > 0 else 0)

        z, y, x = self.coords[tomo_index][coord_index]
        even_subvolume = self.load_and_normalize(self.tomo_paths_even, tomo_index, z, y, x, eo_idx=0)
        if self.method in ['isonet2-n2n','n2n']:
            odd_subvolume = self.load_and_normalize(self.tomo_paths_odd, tomo_index, z, y, x, eo_idx=1)
            x1_volume, x2_volume = self.random_swap(
                np.array(even_subvolume, dtype=np.float32)[np.newaxis, ...], 
                np.array(odd_subvolume, dtype=np.float32)[np.newaxis, ...]
            )
        else:
            x1_volume = np.array(even_subvolume, dtype=np.float32)[np.newaxis, ...]
            x2_volume = None

        if self.noise_dir != None:
            noise_file = random.choice(self.noise_files)
            noise_volume, _ = read_mrc(noise_file)
            #Noise along y axis is indenpedent, so that the y axis can be permutated.
            noise_volume = np.transpose(noise_volume, axes=(1,0,2))
            noise_volume = np.random.permutation(noise_volume)
            noise_volume = np.transpose(noise_volume, axes=(1,0,2))
        else:
            noise_volume = np.array([0], dtype=np.float32)

        if self.has_groundtruth:
            gt_subvolume = self.load_and_normalize(self.tomo_paths_gt, tomo_index, z, y, x, eo_idx=1)[np.newaxis, ...]
        else:
            gt_subvolume = np.array([0], dtype=np.float32)
        return x1_volume, x2_volume, gt_subvolume, self.mw_list[tomo_index][np.newaxis, ...], \
            self.CTF_list[tomo_index][np.newaxis, ...], self.wiener_list[tomo_index][np.newaxis, ...], noise_volume[np.newaxis, ...]        

if __name__ == '__main__':
    from IsoNet.utils.missing_wedge import mw3D
    mw = mw3D(128, missingAngle=[90 + (-64), 90 - 42],spherical=False, tilt_step=3, start_dim=10000)
    from IsoNet.utils.fileio import write_mrc
    write_mrc('wedge4.mrc',mw)