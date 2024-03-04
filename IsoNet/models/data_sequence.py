import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import mrcfile
from IsoNet.preprocessing.img_processing import normalize

class Train_sets_sp(Dataset):
    def __init__(self, data_dir, max_length = None, shuffle=True, prefix = "train"):
        super(Train_sets_sp, self).__init__()
        # self.path_all = []
        p = '{}/'.format(data_dir)
        self.path_all = sorted([p+f for f in os.listdir(p)])

        # if shuffle:
        #     zipped_path = list(zip(self.path_all[0],self.path_all[1]))
        #     np.random.shuffle(zipped_path)
        #     self.path_all[0], self.path_all[1] = zip(*zipped_path)
        # print(self.path_all)
        #if max_length is not None:
        #    if max_length < len(self.path_all):


    def __getitem__(self, idx):
        with mrcfile.open(self.path_all[idx]) as mrc:
            rx = mrc.data[np.newaxis,:,:,:]
        rx = torch.as_tensor(rx.copy())
        return rx

    def __len__(self):
        return len(self.path_all)

class Train_sets_sp_n2n(Dataset):
    def __init__(self, data_dir, max_length = None, shuffle=True, prefix = "train"):
        super(Train_sets_sp_n2n, self).__init__()
        # self.path_all = []
        p1 = '{}/'.format(data_dir[0])
        p2 = '{}/'.format(data_dir[1])

        self.path_all1 = sorted([p1+f for f in os.listdir(p1)])
        self.path_all2 = sorted([p2+f for f in os.listdir(p2)])

    def __getitem__(self, idx):
        with mrcfile.open(self.path_all1[idx]) as mrc:
            rx = mrc.data[np.newaxis,:,:,:]
        rx = torch.as_tensor(rx.copy())

        with mrcfile.open(self.path_all2[idx]) as mrc:
            ry = mrc.data[np.newaxis,:,:,:]
        ry = torch.as_tensor(ry.copy())
        prob = np.random.rand()
        if prob>=0.5:
            return rx,ry
        if prob<0.5:
            return ry,rx

    def __len__(self):
        return len(self.path_all1)

class Train_sets(Dataset):
    def __init__(self, data_dir, max_length = None, shuffle=True, prefix = "train"):
        super(Train_sets, self).__init__()
        self.path_all = []
        for d in  [prefix+"_x", prefix+"_y"]:
            p = '{}/{}/'.format(data_dir, d)
            self.path_all.append(sorted([p+f for f in os.listdir(p)]))

        # if shuffle:
        #     zipped_path = list(zip(self.path_all[0],self.path_all[1]))
        #     np.random.shuffle(zipped_path)
        #     self.path_all[0], self.path_all[1] = zip(*zipped_path)
        # print(self.path_all)
        #if max_length is not None:
        #    if max_length < len(self.path_all):


    def __getitem__(self, idx):
        with mrcfile.open(self.path_all[0][idx]) as mrc:
            #print(self.path_all[0][idx])
            rx = mrc.data[np.newaxis,:,:,:]
            # rx = mrc.data[:,:,:,np.newaxis]
        with mrcfile.open(self.path_all[1][idx]) as mrc:
            #print(self.path_all[1][idx])
            ry = mrc.data[np.newaxis,:,:,:]
            # ry = mrc.data[:,:,:,np.newaxis]
        rx = torch.as_tensor(rx.copy())
        ry = torch.as_tensor(ry.copy())
        return rx, ry

    def __len__(self):
        return len(self.path_all[0])

class Predict_sets(Dataset):
    def __init__(self, mrc_list, inverted=True):
        super(Predict_sets, self).__init__()
        self.mrc_list=mrc_list
        self.inverted = inverted

    def __getitem__(self, idx):
        with mrcfile.open(self.mrc_list[idx]) as mrc:
            rx = mrc.data[np.newaxis,:,:,:].copy()
        # rx = mrcfile.open(self.mrc_list[idx]).data[:,:,:,np.newaxis]
        if self.inverted:
            rx=normalize(-rx, percentile = True)

        return rx

    def __len__(self):
        return len(self.mrc_list)



def get_datasets(data_dir, max_length = None):
    train_dataset = Train_sets(data_dir, max_length, prefix="train")
    val_dataset = Train_sets(data_dir, max_length, prefix="test")
    return train_dataset, val_dataset#, bench_dataset