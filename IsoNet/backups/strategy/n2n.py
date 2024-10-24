import numpy as np
import torch
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from IsoNet.utils.fileio import read_mrc
import starfile
import mrcfile
class Train_sets(Dataset):
    # this is a class similar to cryocare dataset
    def __init__(self, tomo_star, cube_size=64):
        self.star = starfile.read(tomo_star)
        n_samples = 1000
        self.n_samples_per_tomo = n_samples
        self.sample_shape = [cube_size,cube_size,cube_size]
        column_name_list = self.star.columns.tolist()

        self.tomo_paths_odd = []
        self.tomo_paths_even = []
        self.coords = []
        self.mean = []
        self.std = []
        #self.tomos_odd = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_odd]
        #self.tomos_even = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_even]
        for i, item in  enumerate(self.star.iterrows()):
            row = item[1]
            self.tomo_paths_odd.append(row['rlnTomoReconstructedTomogramHalf1'])
            self.tomo_paths_even.append(row['rlnTomoReconstructedTomogramHalf2'])
            tomo_data, pixel_size = read_mrc(row['rlnTomoReconstructedTomogramHalf1'])
            self.mean.append(np.mean(tomo_data))
            self.std.append(np.std(tomo_data))
            # TODO whether only depend of tomo half1
            if "rlnMaskName" not in column_name_list or row['rlnMaskName'] == None or row['rlnMaskName'] == "None":
                mask = np.ones_like(tomo_data)
            else:
                mask = read_mrc(row['mask_name'])
            [z,y,x] = tomo_data.shape
            coords = self.create_random_coords([0,z], [0, y], [0, x], mask, n_samples)
            self.coords.append(coords)
        self.length = sum([c.shape[0] for c in self.coords])
        # self.tomos_odd = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_odd]
        # self.tomos_even = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_even]
        self.n_tomos = len(self.tomo_paths_odd)
        


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
    
    def __getitem__(self, idx):
        tomo_index, coord_index = idx // self.n_samples_per_tomo, idx % self.n_samples_per_tomo
        z, y, x = self.coords[tomo_index][coord_index]
        even_tomo = mrcfile.mmap(self.tomo_paths_even[tomo_index], mode='r', permissive=True)
        odd_tomo = mrcfile.mmap(self.tomo_paths_odd[tomo_index], mode='r', permissive=True)

        even_subvolume = even_tomo.data[z:z + self.sample_shape[0],
                         y:y + self.sample_shape[1],
                         x:x + self.sample_shape[2]]
        even_subvolume = (even_subvolume - self.mean[tomo_index]) / self.std[tomo_index]

        odd_subvolume = odd_tomo.data[z:z + self.sample_shape[0],
                        y:y + self.sample_shape[1],
                        x:x + self.sample_shape[2]]
        odd_subvolume = (odd_subvolume - self.mean[tomo_index]) / self.std[tomo_index]

        return self.augment(np.array(even_subvolume)[ np.newaxis,...], np.array(odd_subvolume)[np.newaxis,...])

    def close(self):
        for even, odd in zip(self.tomos_even, self.tomos_odd):
            even.close()
            odd.close()

def ddp_train(rank, world_size, port_number, model, training_params):
    #data_path, batch_size, acc_batches, epochs, steps_per_epoch, learning_rate, mixed_precision, model_path
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_number
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    batch_size = training_params['batch_size'] // training_params['acc_batches']
    batch_size_gpu = batch_size // world_size

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    model = DDP(model, device_ids=[rank])
    # if torch.__version__ >= "2.0.0":
    #     GPU_capability = torch.cuda.get_device_capability()
    #     if GPU_capability[0] >= 7:
    #         torch.set_float32_matmul_precision('high')
    #         model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['learning_rate'])
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True

    if training_params['mixed_precision']:
        scaler = torch.cuda.amp.GradScaler()
    
    #from chatGPT: The DistributedSampler shuffles the indices of the entire dataset, not just the portion assigned to a specific GPU. 
    if rank == 0:
        print("initializing dataset (extracting subtomograms)")
    train_dataset = Train_sets(training_params['data_path'], cube_size=training_params['cube_size'])
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_gpu, persistent_workers=True,
                                             num_workers=4, pin_memory=True, sampler=train_sampler)

    steps_per_epoch_train = training_params['steps_per_epoch']
    total_steps = min(len(train_loader)//training_params['acc_batches'], training_params['steps_per_epoch'])
    average_loss_list = []
    loss_fn = nn.L1Loss()
    from IsoNet.utils.utils import debug_matrix
    for epoch in range(training_params['epochs']):
        train_sampler.set_epoch(epoch)
        with tqdm(total=total_steps, unit="batch", disable=(rank!=0)) as progress_bar:
            model.train()
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            for i, batch in enumerate(train_loader):
                x1, x2 = batch[0], batch[1]
                x1 = x1.cuda()
                x2 = x2.cuda()
                optimizer.zero_grad(set_to_none=True)
                if training_params['mixed_precision']:
                    pass
                else:
                    preds = model(x1)
                    loss = loss_fn(x2,preds)
                    loss = loss / training_params['acc_batches']
                    loss.backward()
                loss_item = loss.item()
                              
                if ( (i+1)%training_params['acc_batches'] == 0 ) or (i+1) == min(len(train_loader), steps_per_epoch_train * training_params['acc_batches']):
                    if training_params['mixed_precision']:
                        pass
                    else:
                        optimizer.step()

                if rank == 0 and ( (i+1)%training_params['acc_batches'] == 0 ):
                   progress_bar.set_postfix({"Loss": loss_item})#, "t1": time2-time1, "t2": time3-time2, "t3": time4-time3})
                   progress_bar.update()
                average_loss += loss_item
                
                if i + 1 >= steps_per_epoch_train*training_params['acc_batches']:
                    break
            average_loss = average_loss / (i+1.)
        
                                      
        dist.barrier()
        dist.reduce(average_loss, dst=0)

        average_loss =  average_loss / dist.get_world_size()
        if rank == 0:
            average_loss_list.append(average_loss.cpu().numpy())
            print(f"Epoch [{epoch+1}/{training_params['epochs']}], Train Loss: {average_loss:.4f}")
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'average_loss': average_loss_list,
                }, training_params['outmodel_path'])
    dist.destroy_process_group()

def ddp_predict(rank, world_size, port_number, model, data, tmp_data_path):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_number
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = model.to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    model.eval()

    num_data_points = data.shape[0]
    steps_per_rank = (num_data_points + world_size - 1) // world_size

    output = torch.zeros(steps_per_rank,data.shape[1],data.shape[2],data.shape[3],data.shape[4]).to(rank)
    with torch.no_grad():
        for i in tqdm(range(rank * steps_per_rank, min((rank + 1) * steps_per_rank, num_data_points)),disable=(rank!=0)):
            batch_input  = data[i:i+1]
            batch_output  = model(batch_input.to(rank))
            output[i - rank * steps_per_rank] = batch_output
    gathered_outputs = [torch.zeros_like(output) for _ in range(world_size)]
    dist.all_gather(gathered_outputs, output)
    dist.barrier()
    if rank == 0:
        gathered_outputs = torch.cat(gathered_outputs).cpu().numpy()
        gathered_outputs = gathered_outputs[:data.shape[0]]
        np.save(tmp_data_path,gathered_outputs)
    dist.destroy_process_group()
