import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import mrcfile
from IsoNet.preprocessing.img_processing import normalize

class Train_sets(Dataset):
    def __init__(self, data_dir, max_length = None, shuffle=True, prefix = "train"):
        super(Train_sets, self).__init__()
        self.path_all = []
        for d in  [prefix+"_x", prefix+"_y"]:
            p = '{}/{}/'.format(data_dir, d)
            self.path_all.append(sorted([p+f for f in os.listdir(p)]))

        if shuffle:
            zipped_path = list(zip(self.path_all[0],self.path_all[1]))
            np.random.shuffle(zipped_path)
            self.path_all[0], self.path_all[1] = zip(*zipped_path)
        # print(self.path_all)
        #if max_length is not None:
        #    if max_length < len(self.path_all):


    def __getitem__(self, idx):
        with mrcfile.open(self.path_all[0][idx]) as mrc:
            rx = mrc.data[np.newaxis,:,:,:]
        with mrcfile.open(self.path_all[1][idx]) as mrc:
            ry = mrc.data[np.newaxis,:,:,:]
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