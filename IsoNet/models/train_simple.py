import numpy as np
import torch
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from IsoNet.utils.utils import debug_matrix
import random
from IsoNet.models.masked_loss import masked_loss
from IsoNet.utils.plot_metrics import plot_metrics
from IsoNet.utils.rotations import rotation_list, sample_rot_axis_and_angle, rotate_vol_around_axis_torch
import torch.optim.lr_scheduler as lr_scheduler
import shutil
from packaging import version
if version.parse(torch.__version__) >= version.parse("2.3.0"):
    from torch.amp import GradScaler
else:
    from torch.cuda.amp import GradScaler

def normalize_percentage(tensor, percentile=5):
    original_shape = tensor.shape
    
    batch_size = tensor.size(0)
    flattened_tensor = tensor.reshape(batch_size, -1)
    factor = percentile/100.
    lower_bound = torch.quantile(flattened_tensor, factor, dim=1, keepdim=True)
    upper_bound = torch.quantile(flattened_tensor, 1-factor, dim=1, keepdim=True)

    normalized_flattened = (flattened_tensor - lower_bound) / (upper_bound - lower_bound)
    normalized_flattened = normalized_flattened.view(original_shape)
    return normalized_flattened


def rotate_vol(volume, rotation):
    # B, C, Z, Y, X
    new_vol = torch.rot90(volume, rotation[0][1], [rotation[0][0][0]-3,rotation[0][0][1]-3])
    new_vol = torch.rot90(new_vol, rotation[1][1], [rotation[1][0][0]-3,rotation[1][0][1]-3])
    return new_vol

def apply_F_filter_torch(input_map,F_map):
    fft_input = torch.fft.fftshift(torch.fft.fftn(input_map, dim=(-1, -2, -3)),dim=(-1, -2, -3))
    # mw_shift = torch.fft.fftshift(F_map, dim=(-1, -2, -3))
    out = torch.fft.ifftn(torch.fft.fftshift(fft_input*F_map, dim=(-1, -2, -3)),dim=(-1, -2, -3))
    out =  torch.real(out)
    return out
def process_batch(batch):
    if len(batch) == 6:
        return [b.cuda() for b in batch]
    return batch[0].cuda(), batch[1].cuda(), None, None, None, None

def ddp_train(rank, world_size, port_number, model, train_dataset, training_params):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_number
    
    batch_size_gpu = training_params['batch_size'] // (training_params['acc_batches'] * world_size)
       
    n_workers = max(training_params["ncpus"] // world_size, 1)

    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size_gpu, persistent_workers=True,
            num_workers=n_workers, pin_memory=True, sampler=train_sampler)
    else:
        model = model.to(rank)
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size_gpu, persistent_workers=True,
            num_workers=n_workers, pin_memory=True, sampler=train_sampler, shuffle=True)

    if training_params['compile_model'] == True:
        if torch.__version__ >= "2.0.0":
            GPU_capability = torch.cuda.get_device_capability()
            if GPU_capability[0] >= 7:
                torch.set_float32_matmul_precision('high')
                model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['learning_rate'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_params['T_max'], eta_min=training_params['learning_rate_min'])

    loss_funcs = {"L2": nn.MSELoss(), "Huber": nn.HuberLoss(), "L1": nn.L1Loss()}
    loss_func = loss_funcs.get(training_params['loss_func'])
    
    if training_params['mixed_precision']:
        scaler = GradScaler()

    steps_per_epoch_train = training_params['steps_per_epoch']
    total_steps = min(len(train_loader)//training_params['acc_batches'], training_params['steps_per_epoch'])

    for epoch in range(training_params['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        
        with tqdm(total=total_steps, unit="batch", disable=(rank!=0),desc=f"Epoch {epoch+1}") as progress_bar:
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_inside_mw_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_outside_mw_loss = torch.tensor(0, dtype=torch.float).to(rank)

            for i_batch, batch in enumerate(train_loader):  
                optimizer.zero_grad(set_to_none=True) 
                x1, x2, mw, ctf, wiener, noise_vol = process_batch(batch)

               

                # if training_params['method'] in ["n2n", "regular"]:
                with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                    preds = model(x1)
                    loss = loss_func(x2, preds)
                    outside_mw_loss = loss
                    inside_mw_loss = loss

              
                loss = loss / training_params['acc_batches']
                inside_mw_loss = inside_mw_loss / training_params['acc_batches']
                outside_mw_loss = outside_mw_loss / training_params['acc_batches']

                if training_params['mixed_precision']:
                    scaler.scale(loss).backward() 
                else:
                    loss.backward()
                                        
                if ( (i_batch+1)%training_params['acc_batches'] == 0 ) or (i_batch+1) == min(len(train_loader), steps_per_epoch_train * training_params['acc_batches']):
                    if training_params['mixed_precision']:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                if rank == 0 and ( (i_batch+1)%training_params['acc_batches'] == 0 ):        
                    loss_str = (
                        f"Loss: {loss.item():6.4f} | "
                        f"in_mw_loss: {inside_mw_loss.item():6.4f} | "
                        f"out_mw_loss: {outside_mw_loss.item():6.4f}"
                    )
                    progress_bar.set_postfix_str(loss_str)
                    progress_bar.update()

                average_loss += loss.item()
                average_inside_mw_loss += inside_mw_loss.item()
                average_outside_mw_loss += outside_mw_loss.item()
                
                if i_batch + 1 >= steps_per_epoch_train*training_params['acc_batches']:
                    break

        scheduler.step()

        if world_size > 1:
            dist.barrier()
            dist.reduce(average_loss, dst=0)
            dist.reduce(average_inside_mw_loss, dst=0)
            dist.reduce(average_outside_mw_loss, dst=0)
        average_loss /= (world_size * (i_batch + 1))
        average_inside_mw_loss /= (world_size * (i_batch + 1))
        average_outside_mw_loss /= (world_size * (i_batch + 1))

        if rank == 0:
            training_params["metrics"]["average_loss"].append(average_loss.cpu().numpy()) 
            training_params["metrics"]["inside_mw_loss"].append(average_inside_mw_loss.cpu().numpy()) 
            training_params["metrics"]["outside_mw_loss"].append(average_outside_mw_loss.cpu().numpy()) 

            outmodel_path = f"{training_params['output_dir']}/network_{training_params['method']}_{training_params['arch']}_{training_params['cube_size']}_{training_params['split']}.pt"
            
            print(f"Epoch [{epoch+1:3d}/{training_params['epochs']:3d}], "
                f"Loss: {average_loss:6.4f}, "
                f"in_mw_loss: {average_inside_mw_loss:6.4f}, "
                f"out_mw_loss: {average_outside_mw_loss:6.4f}, "
                f"learning_rate: {scheduler.get_last_lr()[0]:.4e}")

            plot_metrics(training_params["metrics"],f"{training_params['output_dir']}/loss_{training_params['split']}.png")
            if world_size > 1:
                model_params = model.module.state_dict()
            else:
                model_params = model.state_dict()
            
            torch.save({
                    'method':training_params['method'],
                    'arch':training_params['arch'],
                    'model_state_dict': model_params,
                    'metrics': training_params["metrics"],
                    'cube_size': training_params['cube_size']
                    }, outmodel_path)
                        
            if (epoch+1)%training_params['T_max'] == 0:
                total_epochs = epoch+1+training_params["starting_epoch"]
                outmodel_path_epoch = f"{training_params['output_dir']}/network_{training_params['method']}_{training_params['arch']}_{training_params['cube_size']}_epoch{total_epochs}_{training_params['split']}.pt"
                shutil.copy(outmodel_path, outmodel_path_epoch)

    if world_size > 1:
        dist.destroy_process_group()


def ddp_predict(rank, world_size, port_number, model, data, tmp_data_path, F_mask):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port_number)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = model.to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    model.eval()

    num_data_points = data.shape[0]
    steps_per_rank = (num_data_points + world_size - 1) // world_size

    outputs = []

    with torch.no_grad():
        for i in tqdm(
            range(rank * steps_per_rank, min((rank + 1) * steps_per_rank, num_data_points)),
            disable=(rank != 0),desc='predict: '
        ):
            batch_input = data[i:i + 1].to(rank)
            if F_mask is not None:
                F_m = torch.from_numpy(F_mask[np.newaxis,np.newaxis,:,:,:]).to(rank)
                batch_input = apply_F_filter_torch(batch_input, F_m)
            batch_output = model(batch_input).cpu()  # Move output to CPU immediately

            outputs.append(batch_output)

    output = torch.cat(outputs, dim=0).cpu().numpy().astype(np.float32)
    rank_output_path = f"{tmp_data_path}_rank_{rank}.npy"
    np.save(rank_output_path, output)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    # output = torch.zeros(steps_per_rank,data.shape[1],data.shape[2],data.shape[3],data.shape[4]).to(rank)
    # with torch.no_grad():
    #     for i in tqdm(range(rank * steps_per_rank, min((rank + 1) * steps_per_rank, num_data_points)),disable=(rank!=0)):
    #         batch_input  = data[i:i+1]
    #         batch_output  = model(batch_input.to(rank))
    #         output[i - rank * steps_per_rank] = batch_output
    # gathered_outputs = [torch.zeros_like(output) for _ in range(world_size)]
    # dist.all_gather(gathered_outputs, output)
    # dist.barrier()
    # if rank == 0:
    #     gathered_outputs = torch.cat(gathered_outputs).cpu().numpy()
    #     gathered_outputs = gathered_outputs[:data.shape[0]]
    #     np.save(tmp_data_path,gathered_outputs)
    # dist.destroy_process_group()