import numpy as np
import torch
import os
import torch.nn as nn
from tqdm import tqdm
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
    # this is a class similar to cryocare dataset
    def __init__(self, tomo_star, method="n2n", cube_size=64, n_samples=1000):
        self.star = starfile.read(tomo_star)
        self.method = method
        self.n_samples_per_tomo = n_samples
        self.sample_shape = [cube_size,cube_size,cube_size]

        self.tomo_paths_odd = []
        self.tomo_paths_even = []
        self.tomo_paths = []
        self.coords = []
        self.mean = []
        self.std = []
        self.mw_list = []

        column_name_list = self.star.columns.tolist()

        input_column = "rlnTomoName"
        #self.tomos_odd = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_odd]
        #self.tomos_even = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_even]
        for i, item in  enumerate(self.star.iterrows()):
            row = item[1]
            if method == 'spisonet' or method == 'n2n':
                self.tomo_paths_odd.append(row['rlnTomoReconstructedTomogramHalf1'])
                self.tomo_paths_even.append(row['rlnTomoReconstructedTomogramHalf2'])
                tomo_data, pixel_size = read_mrc(row['rlnTomoReconstructedTomogramHalf1'])
            elif method == 'spisonet-single':
                self.tomo_paths.append(row[input_column])
                tomo_data, pixel_size = read_mrc(row[input_column])

            # TODO whether only depend of tomo half1
            self.mean.append(np.mean(tomo_data))
            self.std.append(np.std(tomo_data))

            if "rlnMaskName" not in column_name_list or row['rlnMaskName'] == None or row['rlnMaskName'] == "None":
                mask = np.ones_like(tomo_data)
            else:
                mask = read_mrc(row['mask_name'])

            [z,y,x] = tomo_data.shape
            coords = self.create_random_coords([0,z], [0, y], [0, x], mask, n_samples)
            self.coords.append(coords)

            if method == 'spisonet' or method == 'spisonet-single':
                min_angle = row['rlnTiltMin']
                max_angle = row['rlnTiltMax']
                from IsoNet.utils.missing_wedge import mw3D
                mw = mw3D(cube_size, missingAngle=[90+min_angle, 90-max_angle])
                self.mw_list.append(np.fft.fftshift(mw))

        self.length = sum([c.shape[0] for c in self.coords])
        # self.tomos_odd = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_odd]
        # self.tomos_even = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_even]
        self.n_tomos = len(self.tomo_paths_odd)
        
    def _initialize_data(self):
        """Initialize paths, mean, std, and coordinates from the starfile."""
        column_name_list = self.star.columns.tolist()
        input_column = "rlnTomoName"

        for _, row in self.star.iterrows():
            tomo_data, mask = self._load_tomogram_and_mask(row, column_name_list)

            self.mean.append(np.mean(tomo_data))
            self.std.append(np.std(tomo_data))

            coords = self.create_random_coords(tomo_data.shape, mask, self.n_samples_per_tomo)
            self.coords.append(coords)

            if self.method in ['spisonet', 'spisonet-single']:
                min_angle, max_angle = row['rlnTiltMin'], row['rlnTiltMax']
                self.mw_list.append(self._compute_missing_wedge(self.sample_shape[0], min_angle, max_angle))


    def create_random_coords(self, z, y, x, mask, n_samples):
        #TODO need careful analysis on this. To understand whether it start with the center or corner

        # Inspired by isonet preprocessing.cubes:create_cube_seeds()
        
        # Get permissible locations based on extraction_shape and sample_shape
        slices = tuple([slice(z[0],z[1]-self.sample_shape[2]),
                       slice(y[0],y[1]-self.sample_shape[1]),
                       slice(x[0],x[1]-self.sample_shape[0])])
        
        # Get intersect with mask-allowed values                       
        valid_inds = np.where(mask[slices])
        
        valid_inds = [v + s.start for s, v in zip(slices, valid_inds)]
        
        sample_inds = np.random.choice(len(valid_inds[0]),
                                       n_samples,
                                       replace=len(valid_inds[0]) < n_samples)
        
        rand_inds = [v[sample_inds] for v in valid_inds]
        

        return np.stack([rand_inds[0],rand_inds[1], rand_inds[2]], -1)

                
    
    def augment(self, x, y):
        # self.tilt_axis = None
        # if self.tilt_axis is not None:
        #     if self.sample_shape[0] == self.sample_shape[1] and \
        #             self.sample_shape[0] == self.sample_shape[2]:
        #         rot_k = np.random.randint(0, 4, 1)

        #         x[...,0] = np.rot90(x[...,0], k=rot_k, axes=self.rot_axes)
        #         y[...,0] = np.rot90(y[...,0], k=rot_k, axes=self.rot_axes)


        if np.random.rand() > 0.5:
            return y, x
        else:
            return x, y

    def __len__(self):
        return self.length
    
    def load_and_normalize(self, tomo_path, tomo_index, z, y, x):
            tomo = mrcfile.mmap(tomo_path[tomo_index], mode='r', permissive=True)
            subvolume = tomo.data[z:z + self.sample_shape[0],
                                y:y + self.sample_shape[1],
                                x:x + self.sample_shape[2]]
            return (subvolume - self.mean[tomo_index]) / self.std[tomo_index]
    
    def __getitem__(self, idx):
        tomo_index, coord_index = idx // self.n_samples_per_tomo, idx % self.n_samples_per_tomo
        z, y, x = self.coords[tomo_index][coord_index]

        if self.method == 'n2n' or self.method == 'spisonet':

            even_subvolume = self.load_and_normalize(self.tomo_paths_even, tomo_index, z, y, x)
            odd_subvolume = self.load_and_normalize(self.tomo_paths_even, tomo_index, z, y, x)

            x, y = self.augment(
                np.array(even_subvolume, dtype=np.float32)[np.newaxis, ...], 
                np.array(odd_subvolume, dtype=np.float32)[np.newaxis, ...]
            )

            if self.method == 'spisonet':
                return x,y,self.mw_list[tomo_index]
            return x,y
            
        elif self.method == 'spisonet-single':
            even_subvolume = self.load_and_normalize(self.tomo_paths, tomo_index, z, y, x)

            x = np.array(even_subvolume, dtype=np.float32)[ np.newaxis,...]

            return x, self.mw_list[tomo_index]

    def close(self):
        for even, odd in zip(self.tomos_even, self.tomos_odd):
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