import numpy as np
import torch
import os
from torch.utils.data.dataset import Dataset
from IsoNet.utils.fileio import read_mrc
import starfile
import mrcfile
from tqdm import tqdm
import random

class Train_sets_regular(Dataset):
    def __init__(self, paths, shuffle=True):
        super(Train_sets_regular, self).__init__()
        path_all = []
        for dir in ["train_x", "train_y"]:
            p = f"{paths}/{dir}/"
            path_all.append(sorted([p+f for f in os.listdir(p)]))

        zipped_path = list(map(list, zip(*path_all)))
        if shuffle:
            np.random.shuffle(zipped_path)
        self.path_all = zipped_path

    def __getitem__(self, idx):
        results = []
        for i,p in enumerate(self.path_all[idx]):
            x, _ = read_mrc(p)
            x = x[np.newaxis,:,:,:] 
            x = torch.as_tensor(x.copy())
            results.append(x)
        return results
    
    def __len__(self):
        return len(self.path_all)



class Train_sets_n2n(Dataset):
    """
    Dataset class to load tomograms and provide subvolumes for n2n and spisonet methods.
    """

    def __init__(self, tomo_star, method="n2n", cube_size=64, input_column = "rlnTomoName", 
                 split="full", noise_dir=None,correct_between_tilts=False, start_bt_size=48,
                 snrfalloff=0, deconvstrength=1, highpassnyquist=0.02):
        self.star = starfile.read(tomo_star)
        self.method = method
        self.n_tomos = len(self.star)
        self.input_column = input_column
        self.cube_size = cube_size
        self.split = split

        self.n_samples_per_tomo = []
        self.tomo_paths_odd = []
        self.tomo_paths_even = []
        self.tomo_paths = []
        self.coords = []
        self.mean = []
        self.std = []
        self.mw_list = []
        self.wiener_list = []
        self.CTF_list = []
        self.start_bt_size=start_bt_size
        self.correct_between_tilts = correct_between_tilts
        self.snrfalloff=snrfalloff
        self.deconvstrength=deconvstrength
        self.highpassnyquist=highpassnyquist


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
            n_samples = row['rlnNumberSubtomo']
            if self.split in ["top", "bottom"]:
                n_samples = n_samples // 2
            self.n_samples_per_tomo.append(n_samples)
            mask = self._load_statistics_and_mask(row, column_name_list)
            if 'rlnBoxFile' not in row or row['rlnBoxFile'] in [None, "None"]:
                coords = self.create_random_coords(mask.shape, mask, n_samples)
            else:
                coords = np.loadtxt(row['rlnBoxFile'])
            self.coords.append(coords)

            min_angle, max_angle, tilt_step = row['rlnTiltMin'], row['rlnTiltMax'], row['rlnTiltStep']
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
        if self.method in ['isonet2-n2n','n2n']:
            even_column = 'rlnTomoReconstructedTomogramHalf1'
            odd_column = 'rlnTomoReconstructedTomogramHalf2'
        else:
            even_column = self.input_column
            odd_column = self.input_column

        self.tomo_paths_even.append(row[even_column])
        self.tomo_paths_odd.append(row[odd_column])

        with mrcfile.mmap(row[even_column], mode='r', permissive=True) as tomo_even, \
             mrcfile.mmap(row[odd_column], mode='r', permissive=True) as tomo_odd:
            tomo_shape = tomo_even.data.shape
            Z = tomo_shape[0]
            mean = [np.mean(tomo_even.data[Z//2-16:Z//2+16]), np.mean(tomo_odd.data[Z//2-16:Z//2+16])]
            std = [np.std(tomo_even.data[Z//2-16:Z//2+16]), np.std(tomo_odd.data[Z//2-16:Z//2+16])]

        self.mean.append(mean)
        self.std.append(std)
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
                                    phaseshift=0, amplitude=row['rlnAmplitudeContrast'],bfactor=0, \
                                        shape=[self.cube_size,self.cube_size,self.cube_size], clip_first_peak=True)
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
        odd_subvolume = self.load_and_normalize(self.tomo_paths_odd, tomo_index, z, y, x, eo_idx=1)

        x, y = self.random_swap(
            np.array(even_subvolume, dtype=np.float32)[np.newaxis, ...], 
            np.array(odd_subvolume, dtype=np.float32)[np.newaxis, ...]
        )

        if self.noise_dir != None:
            noise_file = random.choice(self.noise_files)
            noise_volume, _ = read_mrc(noise_file)
            #Noise along y axis is indenpedent, so that the y axis can be permutated.
            noise_volume = np.transpose(noise_volume, axes=(1,0,2))
            noise_volume = np.random.permutation(noise_volume)
            noise_volume = np.transpose(noise_volume, axes=(1,0,2))
        else:
            noise_volume = np.array([0], dtype=np.float32)

        return x, y, self.mw_list[tomo_index][np.newaxis, ...], \
                self.CTF_list[tomo_index][np.newaxis, ...], self.wiener_list[tomo_index][np.newaxis, ...], noise_volume[np.newaxis, ...]






































# import os
# import numpy as np
# import torch
# from torch.utils.data.dataset import Dataset
# import mrcfile
# from IsoNet.preprocessing.img_processing import normalize
# import random



# import starfile

# class IsoNet_Dataset(Dataset):
#     # this is a class similar to cryocare dataset
#     def __init__(self, tomo_star, use_n2n=True, use_deconv=True):
#         self.star = starfile.read(tomo_star)

#         column_name_list = self.star.columns.tolist()
#         if "rlnTomoReconstructedTomogramHalf1" not in column_name_list:   
#             use_n2n = False

#         if use_n2n or ("rlnDeconvTomoName" not in column_name_list) or (use_deconv == False):
#             use_deconv = False

#         for i, item in enumerate(self.star.iterrows()):
#             tomo=item[0]
#             if not use_n2n:
#                 reference_tomo = tomo['rlnTomoName']
#             else:
#                 reference_tomo = tomo['rlnTomoReconstructedTomogramHalf1']

#             #with mrcfile.open('')
#             if "rlnMaskName" not in column_name_list:
#                 mask = np.ones_like(tomo)
#             else:
#                 mask = tomo['mask_name']
#                 self.generate_coordinate(self, mask, )
        

#     def generate_coordinates(self):
#         pass

#     def create_random_coords(self, z, y, x, mask, n_samples):
#         # Inspired by isonet preprocessing.cubes:create_cube_seeds()
        
#         # Get permissible locations based on extraction_shape and sample_shape
#         slices = tuple([slice(z[0],z[1]-self.sample_shape[2]),
#                        slice(y[0],y[1]-self.sample_shape[1]),
#                        slice(x[0],x[1]-self.sample_shape[0])])
        
#         # Get intersect with mask-allowed values                       
#         valid_inds = np.where(mask[slices])
        
#         valid_inds = [v + s.start for s, v in zip(slices, valid_inds)]
        
#         sample_inds = np.random.choice(len(valid_inds[0]),
#                                        n_samples,
#                                        replace=len(valid_inds[0]) < n_samples)
        
#         rand_inds = [v[sample_inds] for v in valid_inds]
        

#         return np.stack([rand_inds[0],rand_inds[1], rand_inds[2]], -1)

                
    
#     def __len__(self):
#         return len(self.star)
    
#     def __getitem__(self, idx):
#         tomo_index, coord_index = idx//self.n_samples_per_tomo
#         pass

# if __name__ == "__main__":

#     dataset = IsoNet_Dataset('tomograms.star')

# class Train_sets(Dataset):
#     def __init__(self, data_star):
#         self.star = starfile.read(data_star)
#         if 'rlnParticle2Name' in self.star.columns:
#             self.n2n = True

# class Train_sets_backup(Dataset):
#     def __init__(self, data_dir, max_length = None, shuffle=True, beta=0.5, prefix = "train"):
#         super(Train_sets, self).__init__()
#         self.beta=beta
#         self.path_all = []
#         for d in  [prefix+"_x1", prefix+"_y1", prefix+"_x2", prefix+"_y2"]:
#             p = '{}/{}/'.format(data_dir, d)
#             self.path_all.append(sorted([p+f for f in os.listdir(p)]))
#         # shuffle=False
#         # if shuffle:
#         #     zipped_path = list(zip(self.path_all[0],self.path_all[1]))
#         #     np.random.shuffle(zipped_path)
#         #     self.path_all[0], self.path_all[1] = zip(*zipped_path)
#         #if max_length is not None:
#         #    if max_length < len(self.path_all):


#     def __getitem__(self, idx):

#         with mrcfile.open(self.path_all[0][idx]) as mrc:
#             x1 = mrc.data[np.newaxis,:,:,:]
#         with mrcfile.open(self.path_all[1][idx]) as mrc:
#             y1 = mrc.data[np.newaxis,:,:,:]
#         with mrcfile.open(self.path_all[2][idx]) as mrc:
#             x2 = mrc.data[np.newaxis,:,:,:]
#         with mrcfile.open(self.path_all[3][idx]) as mrc:
#             y2 = mrc.data[np.newaxis,:,:,:]

#         random_number = random.random()
#         random_number2 = random.random()
#         # x = y1
#         # y = y2
#         if random_number<self.beta:
#             if random_number2>0.5:
#                 x = x1
#                 y = y2
#             else:
#                 x = x2
#                 y = y1
#         else:
#             if random_number2>0.5:
#                 x = x1
#                 y = y1
#             else:
#                 x = x2
#                 y = y2
#         rx = torch.as_tensor(x.copy())
#         ry = torch.as_tensor(y.copy())
#         return rx, ry
#     def __len__(self):
#         return len(self.star)
    
#     def __getitem__(self,idx):
#         particle = self.star.loc[idx]
#         p1_name = particle['rlnParticleName']
#         if self.n2n:
#             p2_name = particle['rlnParticle2Name']
#         else:
#             p2_name = p1_name

#         with mrcfile.open(p1_name) as mrc:
#             rx = mrc.data[np.newaxis,:,:,:]
#         with mrcfile.open(p2_name) as mrc:
#             ry = mrc.data[np.newaxis,:,:,:]

#         rx = torch.as_tensor(rx.copy())
#         ry = torch.as_tensor(ry.copy())
#         wedge_name = particle['rlnWedgeName']

#         with mrcfile.open(wedge_name) as mrc:
#             wedge = mrc.data[:,:,:]        
#         wedge = torch.as_tensor(wedge.copy())

#         prob = np.random.rand()
#         if prob >= 0.5:
#             return rx,ry,wedge
#         if prob < 0.5:
#             return ry,rx,wedge


# class Train_sets_sp(Dataset):
#     def __init__(self, data_dir, max_length = None, shuffle=True, prefix = "train"):
#         super(Train_sets_sp, self).__init__()
#         # self.path_all = []
#         p = '{}/'.format(data_dir)
#         self.path_all = sorted([p+f for f in os.listdir(p)])

#         # if shuffle:
#         #     zipped_path = list(zip(self.path_all[0],self.path_all[1]))
#         #     np.random.shuffle(zipped_path)
#         #     self.path_all[0], self.path_all[1] = zip(*zipped_path)
#         # print(self.path_all)
#         #if max_length is not None:
#         #    if max_length < len(self.path_all):


#     def __getitem__(self, idx):
#         with mrcfile.open(self.path_all[idx]) as mrc:
#             rx = mrc.data[np.newaxis,:,:,:]
#         rx = torch.as_tensor(rx.copy())
#         return rx

#     def __len__(self):
#         return len(self.path_all)

# class Train_sets_sp_n2n(Dataset):
#     def __init__(self, data_dir, max_length = None, shuffle=True, prefix = "train"):
#         super(Train_sets_sp_n2n, self).__init__()
#         # self.path_all = []
#         p1 = '{}/'.format(data_dir[0])
#         p2 = '{}/'.format(data_dir[1])

#         self.path_all1 = sorted([p1+f for f in os.listdir(p1)])
#         self.path_all2 = sorted([p2+f for f in os.listdir(p2)])

#     def __getitem__(self, idx):
#         with mrcfile.open(self.path_all1[idx]) as mrc:
#             rx = mrc.data[np.newaxis,:,:,:]
#         rx = torch.as_tensor(rx.copy())

#         with mrcfile.open(self.path_all2[idx]) as mrc:
#             ry = mrc.data[np.newaxis,:,:,:]
#         ry = torch.as_tensor(ry.copy())
#         prob = np.random.rand()
#         if prob>=0.5:
#             return rx,ry
#         if prob<0.5:
#             return ry,rx

#     def __len__(self):
#         return len(self.path_all1)

# class Train_sets(Dataset):
#     def __init__(self, data_dir, max_length = None, shuffle=True, prefix = "train"):
#         super(Train_sets, self).__init__()
#         self.path_all = []
#         for d in  [prefix+"_x", prefix+"_y"]:
#             p = '{}/{}/'.format(data_dir, d)
#             self.path_all.append(sorted([p+f for f in os.listdir(p)]))

#         # if shuffle:
#         #     zipped_path = list(zip(self.path_all[0],self.path_all[1]))
#         #     np.random.shuffle(zipped_path)
#         #     self.path_all[0], self.path_all[1] = zip(*zipped_path)
#         # print(self.path_all)
#         #if max_length is not None:
#         #    if max_length < len(self.path_all):


#     def __getitem__(self, idx):
#         with mrcfile.open(self.path_all[0][idx]) as mrc:
#             #print(self.path_all[0][idx])
#             rx = mrc.data[np.newaxis,:,:,:]
#             # rx = mrc.data[:,:,:,np.newaxis]
#         with mrcfile.open(self.path_all[1][idx]) as mrc:
#             #print(self.path_all[1][idx])
#             ry = mrc.data[np.newaxis,:,:,:]
#             # ry = mrc.data[:,:,:,np.newaxis]
#         rx = torch.as_tensor(rx.copy())
#         ry = torch.as_tensor(ry.copy())
#         return rx, ry

#     def __len__(self):
#         return len(self.path_all[0])

# class Predict_sets(Dataset):
#     def __init__(self, mrc_list, inverted=True):
#         super(Predict_sets, self).__init__()
#         self.mrc_list=mrc_list
#         self.inverted = inverted

#     def __getitem__(self, idx):
#         with mrcfile.open(self.mrc_list[idx]) as mrc:
#             rx = mrc.data[np.newaxis,:,:,:].copy()
#         # rx = mrcfile.open(self.mrc_list[idx]).data[:,:,:,np.newaxis]
#         if self.inverted:
#             #rx=normalize(-rx, percentile = True)
#             rx=-rx
#         return rx

# #     def __len__(self):
# #         return len(self.mrc_list)



# # def get_datasets(data_dir, max_length = None):
# #     train_dataset = Train_sets(data_dir, max_length, prefix="train")
# #     val_dataset = Train_sets(data_dir, max_length, prefix="test")
# #     return train_dataset, val_dataset#, bench_dataset