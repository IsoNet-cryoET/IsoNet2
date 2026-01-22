from glob import glob
import string
import logging
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
from IsoNet.models.masked_loss import masked_loss, apply_fourier_mask_to_tomo
from IsoNet.models.masked_loss import masked_loss, apply_fourier_mask_to_tomo
from IsoNet.utils.plot_metrics import plot_metrics
from IsoNet.utils.normalize import normalize_percentage, normalize_mean_std 
from IsoNet.utils.rotations import rotation_list_90, generate_random_rotation, rotate_vol_around_axis_torch, rotate_vol_90
import torch.optim.lr_scheduler as lr_scheduler
import shutil
from packaging import version
from scipy.stats import linregress
if version.parse(torch.__version__) >= version.parse("2.3.0"):
    from torch.amp import GradScaler
else:
    from torch.cuda.amp import GradScaler
import math


# def normalize_percentage(tensor, percentile=4, lower_bound = None, upper_bound=None):
#     original_shape = tensor.shape
    
#     batch_size = tensor.size(0)
#     flattened_tensor = tensor.reshape(batch_size, -1)

#     factor = percentile/100.
#     lower_bound_subtomo = torch.quantile(flattened_tensor, factor, dim=1, keepdim=True)
#     upper_bound_subtomo = torch.quantile(flattened_tensor, 1-factor, dim=1, keepdim=True)

#     if lower_bound is None: 
#         normalized_flattened = (flattened_tensor - lower_bound_subtomo) / (upper_bound_subtomo - lower_bound_subtomo)
#         normalized_flattened = normalized_flattened.view(original_shape)
#     else:
#         normalized_flattened = (flattened_tensor - lower_bound) / (upper_bound - lower_bound)
#         normalized_flattened = normalized_flattened.view(original_shape)        
    
#     return normalized_flattened, lower_bound_subtomo, upper_bound_subtomo


def normalize_mean_std(tensor, mean_val = None, std_val=None, matching = False, stats_only = False):
    # merge_factor = 0.99
    normalized_tensor = None
    mean_subtomo = tensor.mean(dim=(-3,-2,-1), keepdim=True)
    std_subtomo = tensor.std(correction=0,dim=(-3,-2,-1), keepdim=True)
    
    if not stats_only:
        if mean_val is None: 
            normalized_tensor = (tensor - mean_subtomo) / std_subtomo
        else:
            if matching:
                normalized_tensor = (tensor - mean_subtomo) / std_subtomo * std_val + mean_val
            else:
                # mean_val = mean_val*merge_factor + mean_subtomo*(1-merge_factor)
                # std_val = std_val*merge_factor + std_subtomo*(1-merge_factor)
                normalized_tensor = (tensor - mean_val) / std_val
        
    return {"normalized_tensor":normalized_tensor, "mean":mean_subtomo, "std":std_subtomo}

def cross_correlate(M1, M2):
    M1_norm = (M1-M1.mean()) / M1.std(correction = False)
    M2_norm = (M2-M2.mean()) / M2.std(correction = False)
    return (M1_norm*M2_norm).mean()


def rotate_vol(volume, rotation):
    # B, C, Z, Y, X
    new_vol = torch.rot90(volume, rotation[0][1], [rotation[0][0][0]-3,rotation[0][0][1]-3])
    new_vol = torch.rot90(new_vol, rotation[1][1], [rotation[1][0][0]-3,rotation[1][0][1]-3])
    return new_vol

def apply_F_filter_torch(tensornput_map,F_map):
    fft_input = torch.fft.fftshift(torch.fft.fftn(tensornput_map, dim=(-1, -2, -3)),dim=(-1, -2, -3))
    # mw_shift = torch.fft.fftshift(F_map, dim=(-1, -2, -3))
    out = torch.fft.ifftn(torch.fft.fftshift(fft_input*F_map, dim=(-1, -2, -3)),dim=(-1, -2, -3))
    out =  torch.real(out)
    return out

def process_batch(batch):
    if len(batch) == 7:
    if len(batch) == 7:
        return [b.cuda() for b in batch]
    return batch[0].cuda(), batch[1].cuda(), None, None, None, None, None
    return batch[0].cuda(), batch[1].cuda(), None, None, None, None, None

def ddp_train(rank, world_size, port_number, model, train_dataset, training_params):
    converged = False
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
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_params['T_max'], eta_min=training_params['learning_rate_min'])
    # compute effective steps-per-epoch (account for gradient accumulation cap by training_params['steps_per_epoch'])
    eff_steps_per_epoch = max(1, min(len(train_loader) // max(1, training_params['acc_batches']), training_params['steps_per_epoch']))
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training_params['learning_rate'],
        steps_per_epoch=eff_steps_per_epoch,
        epochs=training_params['epochs'],
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    loss_funcs = {"L2": nn.MSELoss(), "Huber": nn.HuberLoss(), "L1": nn.L1Loss()}
    loss_func = loss_funcs.get(training_params['loss_func'])
    
    if training_params['mixed_precision']:
        scaler = torch.amp.GradScaler('cuda')   

    steps_per_epoch_train = training_params['steps_per_epoch']
    total_steps = min(len(train_loader)//training_params['acc_batches'], training_params['steps_per_epoch'])

    for epoch in range(training_params['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad() 
        with tqdm(total=total_steps, unit=" batch", disable=(rank!=0),desc=f"Epoch {epoch+1}") as progress_bar:
        with tqdm(total=total_steps, unit=" batch", disable=(rank!=0),desc=f"Epoch {epoch+1}") as progress_bar:
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_inside_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_outside_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_inside_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_outside_loss = torch.tensor(0, dtype=torch.float).to(rank)

            for i_batch, batch in enumerate(train_loader):  
                x1, x2, gt, mw, ctf, wiener, noise_vol = process_batch(batch)
                
                if training_params['CTF_mode'] in  ["phase_only", 'wiener','network']:
                    if training_params["phaseflipped"]:
                        ctf = torch.abs(ctf)
                        wiener = torch.abs(wiener) 
                    elif training_params['do_phaseflip_input']:
                        x1 = apply_F_filter_torch(x1, torch.sign(ctf))
                        x2 = apply_F_filter_torch(x2, torch.sign(ctf)) if x2 is not None else None
                        ctf = torch.abs(ctf)
                        wiener = torch.abs(wiener) 

                if training_params['method'] in ["isonet2", "isonet2-n2n"]:
                    if random.random()<training_params['random_rot_weight']:
                        rotate_func = rotate_vol_around_axis_torch
                        rot = sample_rot_axis_and_angle()
                    else:
                        rotate_func = rotate_vol
                        rot = random.choice(rotation_list)
                    
                    x1 = apply_F_filter_torch(x1, mw)
                    x2 = apply_F_filter_torch(x2, mw) if x2 is not None else None
                    x = [x1, x2]

                    x1_stats = normalize_mean_std(x1,stats_only=True)
                    x2_stats = normalize_mean_std(x2,stats_only=True) if x2 is not None else None

                    with torch.no_grad(), torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                        pred_x1 = model(x1).to(torch.float32)
                        pred_x2 = model(x2).to(torch.float32) if x2 is not None else None

                    orig_noise_std = torch.std(x1 - x2) / 1.414 if x2 is not None else torch.tensor(0.0, device=x1.device)
                    new_noise_std = torch.std(pred_x1 - pred_x2) / 1.414 if (pred_x2 is not None) else torch.tensor(0.0, device=pred_x1.device)
                    delta_noise_std = torch.sqrt(torch.abs(orig_noise_std**2 - new_noise_std**2))

                    pred_x1 = pred_x1 + torch.randn_like(pred_x1) * delta_noise_std
                    pred_x2 = pred_x2 + torch.randn_like(pred_x2) * delta_noise_std if pred_x2 is not None else None

                    # apply ctf if required
                    if training_params['CTF_mode'] in ['network', 'wiener']:
                        pred_x1 = apply_F_filter_torch(pred_x1, ctf)
                        pred_x2 = apply_F_filter_torch(pred_x2, ctf) if pred_x2 is not None else None

                    # fill with original content and inverse mw
                    x_filled1 = apply_F_filter_torch(pred_x1, 1 - mw) + x1
                    x_filled2 = apply_F_filter_torch(pred_x2, 1 - mw) + x2 if (pred_x2 is not None and x2 is not None) else None

                    # rotate filled volumes
                    x_filled_rot1 = rotate_func(x_filled1, rot)
                    x_filled_rot2 = rotate_func(x_filled2, rot) if x_filled2 is not None else None

                    # apply mw to rotated filled volumes
                    x_filled_rot_mw1 = apply_F_filter_torch(x_filled_rot1, mw)
                    x_filled_rot_mw2 = apply_F_filter_torch(x_filled_rot2, mw) if x_filled_rot2 is not None else None

                    # normalize per-item using stored original stats
                    x_filled_rot_mw_normalized1 = normalize_mean_std(
                        tensor=x_filled_rot_mw1,
                        mean_val=x1_stats["mean"],
                        std_val=x1_stats["std"],
                        matching=True
                    )
                    x_filled_rot_mw_normalized2 = None
                    if x_filled_rot_mw2 is not None and x2_stats is not None:
                        x_filled_rot_mw_normalized2 = normalize_mean_std(
                            tensor=x_filled_rot_mw2,
                            mean_val=x2_stats["mean"],
                            std_val=x2_stats["std"],
                            matching=True
                        )
                    rotated_mw = rotate_func(mw, rot)
                    
                    net_input = x_filled_rot_mw_normalized1
                    # net_input2 = x_filled_rot_mw_normalized2 if x_filled_rot_mw_normalized2 is not None else None
                    # net_target1 = x_filled_rot1
                    net_target = x_filled_rot2 if x_filled_rot2 is not None else x_filled_rot1
                
                else:
                    net_input = x1
                    
                    net_target = x2
                
                with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                    pred_y = model(net_input).to(torch.float32)

                if training_params['CTF_mode']  == 'network':
                    pred_y = apply_F_filter_torch(pred_y, ctf)
                    
                elif training_params['CTF_mode']  == 'wiener':
                    net_target = apply_F_filter_torch(net_target, wiener)
                
                # Loss Calculation
                if training_params['method'] == "isonet2":
                    outside_loss, inside_loss = masked_loss(pred_y[0], net_target[0], rotated_mw, mw, loss_func = loss_func)
                    if training_params['mw_weight'] > 0:
                        loss = inside_loss + training_params['mw_weight'] * outside_loss
                    else:
                        loss = loss_func(pred_y[0], net_target[0]) 

                elif training_params['method'] == "isonet2_n2n":
                    outside_loss1, inside_loss1 = masked_loss(pred_y[0], net_target[1], rotated_mw, mw, loss_func = loss_func)
                    outside_loss2, inside_loss2 = masked_loss(pred_y[1], net_target[0], rotated_mw, mw, loss_func = loss_func)
                    outside_loss = (outside_loss1 + outside_loss2)/2.
                    inside_loss = (inside_loss1+ inside_loss2)/2.
                    
                    if training_params['mw_weight'] > 0:
                        loss = inside_loss + training_params['mw_weight'] * outside_loss
                    else:
                        loss1 = loss_func(pred_y[0], net_target[1])
                        loss2 = loss_func(pred_y[1], net_target[0])
                        loss = (loss1 + loss2)/2.
                
                elif training_params['method'] == "n2n":
                    # loss1 = loss_func(pred_y[0], net_target[1])
                    # loss2 = loss_func(pred_y[1], net_target[0])

                    # loss = (loss1 + loss2)/2.
                    loss = loss_func(pred_y[0], net_target[1])
                    
                    outside_loss = loss
                    inside_loss = loss

                loss = loss / training_params['acc_batches']
                inside_loss = inside_loss / training_params['acc_batches']
                outside_loss = outside_loss / training_params['acc_batches']

                if training_params['mixed_precision']:
                    scaler.scale(loss).backward() 
                else:
                    loss.backward()
                                        
                if ( (i_batch+1)%training_params['acc_batches'] == 0 ) or (i_batch+1) == min(len(train_loader), steps_per_epoch_train * training_params['acc_batches']):
                    if training_params['mixed_precision']:
                        # Unscale first so we can check for inf/nan in gradients.
                        scaler.unscale_(optimizer)
                        found_inf = False
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is not None:
                                    # If any gradient contains non-finite values, skip the step.
                                    if not torch.isfinite(p.grad).all():
                                        found_inf = True
                                        break
                            if found_inf:
                                break

                        if not found_inf:
                            # safe to do the scaled step
                            scaler.step(optimizer)
                            scaler.update()
                            # advance scheduler only when optimizer actually stepped
                            scheduler.step()
                        else:
                            # skip optimizer step but still update the scaler so scale may be reduced
                            scaler.update()
                    else:
                        optimizer.step()
                        # advance scheduler only when optimizer actually stepped
                        scheduler.step()
                    optimizer.zero_grad() 

                if rank == 0 and ( (i_batch+1)%training_params['acc_batches'] == 0 ):        
                    loss_str = (f"Loss: {loss.item():6.5f}, Learning rate: {scheduler.get_last_lr()[0]:.4e}")
                    progress_bar.set_postfix_str(loss_str)
                    progress_bar.update()

                average_loss += loss.item()
                average_inside_loss += inside_loss.item()
                average_outside_loss += outside_loss.item()
                average_inside_loss += inside_loss.item()
                average_outside_loss += outside_loss.item()
                
                if i_batch + 1 >= steps_per_epoch_train*training_params['acc_batches']:
                    break
        # removed extra optimizer.step() and epoch-level scheduler.step() because scheduler is stepped per optimizer step

        if world_size > 1:
            dist.barrier()
            dist.reduce(average_loss, dst=0)
            dist.reduce(average_inside_loss, dst=0)
            dist.reduce(average_outside_loss, dst=0)
        average_loss /= (world_size * (i_batch + 1))
        average_inside_loss /= (world_size * (i_batch + 1))
        average_outside_loss /= (world_size * (i_batch + 1))

        if rank == 0:
            training_params["metrics"]["average_loss"].append(average_loss.cpu().numpy()) 
            training_params["metrics"]["inside_loss"].append(average_inside_loss.cpu().numpy()) 
            training_params["metrics"]["outside_loss"].append(average_outside_loss.cpu().numpy()) 

            outmodel_path = f"{training_params['output_dir']}/network_{training_params['method']}_{training_params['arch']}_{training_params['cube_size']}_{training_params['split']}.pt"
            
            loss_str = f"Epoch [{epoch+1:3d}/{training_params['epochs']:3d}] Loss: {average_loss:6.5f}"

            if training_params['method'] in ['isonet2', 'isonet2-n2n']:
                loss_str += f", inside_loss: {average_inside_loss:6.5f}, outside_loss: {average_outside_loss:6.5f}"
            print(loss_str)

            plot_metrics(training_params["metrics"],f"{training_params['output_dir']}/loss_{training_params['split']}.png")

            if world_size > 1:
                model_params = model.module.state_dict()
            else:
                model_params = model.state_dict()

            torch.save({
                    'method':training_params['method'],
                    'CTF_mode': training_params['CTF_mode'],
                    'do_phaseflip_input': training_params['do_phaseflip_input'],
                    'arch':training_params['arch'],
                    'model_state_dict': model_params,
                    'metrics': training_params["metrics"],
                    'cube_size': training_params['cube_size']
                    }, outmodel_path)
                        
            total_epochs = epoch+1+training_params["starting_epoch"]
            outmodel_path_epoch = f"{training_params['output_dir']}/network_{training_params['method']}_{training_params['arch']}_{training_params['cube_size']}_epoch{total_epochs}_{training_params['split']}.pt"

# =================================================================================================================================================================================================================================================================
            window = 10
            pvalue_thresh = 0.01
            epochs = np.arange(window)
            if len(training_params["metrics"]["average_loss"]) > window:    
                
                if (training_params["metrics"]["average_loss"][-1] == min(training_params["metrics"]["average_loss"])):
                    min_files = glob(os.path.join(training_params['output_dir'], "*_min_loss.pt"))
                    for mf in min_files:
                        try:
                            os.remove(mf)
                        except OSError as e:
                            logging.warning(f"Failed to remove {mf}: {e}")
                    shutil.copy(outmodel_path, f"{outmodel_path_epoch[:-3]}_min_loss.pt")         

                else:
                    recent_losses = training_params["metrics"]["average_loss"][-window:]
                    result = linregress(epochs, recent_losses, alternative='less')
                    converged = (result.pvalue > pvalue_thresh or result.rvalue**2 < 0.1)
# =================================================================================================================================================================================================================================================================            
            
            if (epoch+1)%training_params['T_max'] == 0:
                shutil.copy(outmodel_path, outmodel_path_epoch)

    if world_size > 1:
        dist.destroy_process_group()
    
    return(outmodel_path, converged)


def ddp_predict(rank, world_size, port_number, model, data, tmp_data_path, F_mask,idx=None):

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
        for tensor in tqdm(
            range(rank * steps_per_rank, min((rank + 1) * steps_per_rank, num_data_points)),
            disable=(rank != 0), desc=f'Predicting Tomogram{f" {idx}" if idx is not None else ""}: '
        ):
            batch_input = data[i:i + 1].to(rank)
            if F_mask is not None:
                F_m = torch.from_numpy(F_mask[np.newaxis,np.newaxis,:,:,:]).to(rank)
                batch_input = apply_F_filter_torch(batch_input, F_m)
            batch_output = model(batch_input).cpu()  # Move output to CPU immediately
            # if rank == 0:
            #     write_mrc('testIN.mrc', batch_input[0][0].cpu().numpy().astype(np.float32))
            #     write_mrc('testOUT.mrc', batch_output[0][0].numpy().astype(np.float32))
            outputs.append(batch_output)

    output = torch.cat(outputs, dim=0).cpu().numpy().astype(np.float32)
    rank_output_path = f"{tmp_data_path}_rank_{rank}.npy"
    np.save(rank_output_path, output)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

