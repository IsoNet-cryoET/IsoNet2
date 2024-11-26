import numpy as np
import torch
import os
from torch.utils.data.dataset import Dataset
from IsoNet.utils.fileio import read_mrc
import starfile
import mrcfile

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

    def __init__(self, tomo_star, method="n2n", cube_size=64, input_column = "rlnTomoName", isCTFflipped=False):
        self.star = starfile.read(tomo_star)
        self.method = method
        self.n_samples_per_tomo = []
        self.sample_shape = [cube_size, cube_size, cube_size]
        self.cube_size = cube_size

        # Initialize paths, statistics, and coordinates
        self.tomo_paths_odd = []
        self.tomo_paths_even = []
        self.tomo_paths = []
        self.coords = []
        self.mean = []
        self.std = []
        self.mw_list = []
        self.wiener_list = []
        self.CTF_list = []
        self.n_tomos = len(self.star)
        self.input_column = input_column
        self.isCTFflipped = isCTFflipped

        # Initialize data from starfile
        self._initialize_data()
        # Compute total dataset length
        self.length = sum([coords.shape[0] for coords in self.coords])

    def _initialize_data(self):
        """Initialize paths, mean, std, and coordinates from the starfile."""
        column_name_list = self.star.columns.tolist()

        for _, row in self.star.iterrows():
            self.n_samples_per_tomo=row['rlnNumberSubtomo']

            mask = self._load_statistics_and_mask(row, column_name_list)
            if row['rlnBoxFile'] in [None, "None"]:
                coords = self.create_random_coords(mask.shape, mask, self.n_samples_per_tomo)
            else:
                coords = np.loadtxt(row['rlnBoxFile'])
            self.coords.append(coords)

            # if self.method in ['isonet2','isonet2-n2n']:
            # compute for all the modes
            min_angle, max_angle = row['rlnTiltMin'], row['rlnTiltMax']
            self.mw_list.append(self._compute_missing_wedge(self.sample_shape[0], min_angle, max_angle))
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

        #if self.method in ['isonet2', 'n2n']:
        self.tomo_paths_even.append(row[even_column])
        self.tomo_paths_odd.append(row[odd_column])
        tomo_data =  mrcfile.mmap(row[even_column], mode='r', permissive=True).data
        tomo_data2 = mrcfile.mmap(row[odd_column], mode='r', permissive=True).data
        Z = tomo_data.data.shape[0]
        mean = [np.mean(tomo_data[Z//2-16:Z//2+16]), np.mean(tomo_data2[Z//2-16:Z//2+16])]
        std = [np.std(tomo_data[Z//2-16:Z//2+16]), np.std(tomo_data2[Z//2-16:Z//2+16])]
        # else:
        #     self.tomo_paths.append(row['rlnTomoName'])
        #     #tomo_data, _ = read_mrc(row['rlnTomoName'])
        #     tomo_data =  mrcfile.mmap(row['rlnTomoName'], mode='r', permissive=True).data
        #     Z = tomo_data.data.shape[0]
        #     mean = [np.mean(tomo_data[Z//2-16:Z//2+16])]
        #     std = [np.std(tomo_data[Z//2-16:Z//2+16])]           

        self.mean.append(mean)
        self.std.append(std)
        mask, _ = np.ones_like(tomo_data), 1 if "rlnMaskName" not in column_name_list or row['rlnMaskName']=="None" or \
                    row['rlnMaskName']==None else read_mrc(row['rlnMaskName'])
        return mask.copy()

    def create_random_coords(self, shape, mask, n_samples):
        """
        Create random coordinates within permissible regions for subvolume extraction.
        """
        size = self.sample_shape[0]
        z_max, y_max, x_max = shape
        half_size = size // 2 + 1 
        mask[0:half_size,:,:] = 0
        mask[z_max-half_size:z_max,:,:] = 0
        mask[:,0:half_size,:] = 0
        mask[:,y_max-half_size:y_max,:] = 0
        mask[:,:,0:half_size] = 0
        mask[:,:,x_max-half_size:x_max] = 0


        valid_inds = np.where(mask)

        sample_inds = np.random.choice(len(valid_inds[0]), n_samples, replace=(len(valid_inds[0]) < n_samples))
        rand_inds = [v[sample_inds] for v in valid_inds]
        rand_inds = [v - half_size for v in rand_inds]

        return np.stack(rand_inds, -1)

    def _compute_missing_wedge(self, cube_size, min_angle, max_angle):
        """Compute the missing wedge mask for given tilt angles."""
        from IsoNet.utils.missing_wedge import mw3D
        mw = mw3D(cube_size, missingAngle=[90 + min_angle, 90 - max_angle])
        return mw

    def _compute_CTF_vol(self, row):
        """Compute the missing wedge mask for given tilt angles."""
        # defocus in Anstron convert to um
        defocus = row['rlnDefocus']/10000.
        from IsoNet.utils.CTF import get_wiener_3d,get_ctf_3d
        ctf3d = get_ctf_3d(angpix=row['rlnPixelSize'], voltage=row['rlnVoltage'], cs=row['rlnSphericalAberration'], defocus=defocus,\
                                    phaseflipped=self.isCTFflipped, phaseshift=0, amplitude=row['rlnAmplitudeContrast'],length=self.cube_size)
        wiener3d = get_wiener_3d(angpix=row['rlnPixelSize'], voltage=row['rlnVoltage'], cs=row['rlnSphericalAberration'], defocus=defocus,\
                                  snrfalloff=row['rlnSnrFalloff'], deconvstrength=row['rlnDeconvStrength'], highpassnyquist=0.02, \
                                    phaseflipped=self.isCTFflipped, phaseshift=0, amplitude=row['rlnAmplitudeContrast'], length=self.cube_size)
        return ctf3d, wiener3d

    def random_swap(self, x, y):
        if np.random.rand() > 0.5:
            return y, x
        return x, y

    def load_and_normalize(self, tomo_paths, tomo_index, z, y, x, eo_idx):
        """Load and normalize a subvolume from a tomogram."""
        #print(tomo_paths[tomo_index],z, y, x)
        tomo = mrcfile.mmap(tomo_paths[tomo_index], mode='r', permissive=True)
        subvolume = tomo.data[z:z + self.sample_shape[0], y:y + self.sample_shape[1], x:x + self.sample_shape[2]]
        # the output inverted the contrast
        return (self.mean[tomo_index][eo_idx] - subvolume) / self.std[tomo_index][eo_idx]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return a sample of data at a given index."""
        tomo_index, coord_index = divmod(idx, self.n_samples_per_tomo)
        z, y, x = self.coords[tomo_index][coord_index]

        # if self.method in ['n2n', 'isonet2','isonet2-n2n']:
        even_subvolume = self.load_and_normalize(self.tomo_paths_even, tomo_index, z, y, x, eo_idx=0)
        odd_subvolume = self.load_and_normalize(self.tomo_paths_odd, tomo_index, z, y, x, eo_idx=1)

        x, y = self.random_swap(
            np.array(even_subvolume, dtype=np.float32)[np.newaxis, ...], 
            np.array(odd_subvolume, dtype=np.float32)[np.newaxis, ...]
        )

        # if self.method in ['isonet2','isonet2-n2n']:
        return x, y, self.mw_list[tomo_index][np.newaxis, ...], \
                self.CTF_list[tomo_index][np.newaxis, ...], self.wiener_list[tomo_index][np.newaxis, ...]
        # return x, y
        
        # elif self.method == 'spisonet-single':
        #     even_subvolume = self.load_and_normalize(self.tomo_paths, tomo_index, z, y, x, eo_idx=0)
        #     x = np.array(even_subvolume, dtype=np.float32)[np.newaxis, ...]
        #     return x, self.mw_list[tomo_index][np.newaxis, ...]

    def close(self):
        for even, odd in zip(self.tomo_paths_even, self.tomo_paths_odd):
            even.close()
            odd.close()







































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