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
