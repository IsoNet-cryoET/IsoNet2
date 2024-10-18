import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import mrcfile
from IsoNet.preprocessing.img_processing import normalize

import starfile

class IsoNet_Dataset(Dataset):
    # this is a class similar to cryocare dataset
    def __init__(self, tomo_star, use_n2n=True, use_deconv=True):
        self.star = starfile.read(tomo_star)

        column_name_list = self.star.columns.tolist()
        if "rlnTomoReconstructedTomogramHalf1" not in column_name_list:   
            use_n2n = False

        if use_n2n or ("rlnDeconvTomoName" not in column_name_list) or (use_deconv == False):
            use_deconv = False

        for i, item in enumerate(self.star.iterrows()):
            tomo=item[0]
            if not use_n2n:
                reference_tomo = tomo['rlnTomoName']
            else:
                reference_tomo = tomo['rlnTomoReconstructedTomogramHalf1']

            #with mrcfile.open('')
            if "rlnMaskName" not in column_name_list:
                mask = np.ones_like(tomo)
            else:
                mask = tomo['mask_name']
                self.generate_coordinate(self, mask, )
        

    def generate_coordinates(self):
        pass

    def create_random_coords(self, z, y, x, mask, n_samples):
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

                
    
    def __len__(self):
        return len(self.star)
    
    def __getitem__(self, idx):
        tomo_index, coord_index = idx//self.n_samples_per_tomo
        pass

if __name__ == "__main__":

    dataset = IsoNet_Dataset('tomograms.star')

# class Train_sets(Dataset):
#     def __init__(self, data_star):
#         self.star = starfile.read(data_star)
#         if 'rlnParticle2Name' in self.star.columns:
#             self.n2n = True

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
#             rx=normalize(-rx, percentile = True)

#         return rx

#     def __len__(self):
#         return len(self.mrc_list)



# def get_datasets(data_dir, max_length = None):
#     train_dataset = Train_sets(data_dir, max_length, prefix="train")
#     val_dataset = Train_sets(data_dir, max_length, prefix="test")
#     return train_dataset, val_dataset#, bench_dataset