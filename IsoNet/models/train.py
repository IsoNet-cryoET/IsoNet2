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
from IsoNet.utils.plot_metrics import plot_metrics
from IsoNet.utils.rotations import rotation_list_90, generate_random_rotation, rotate_vol_around_axis_torch
import torch.optim.lr_scheduler as lr_scheduler
import shutil
from packaging import version
from IsoNet.utils.fileio import write_mrc
if version.parse(torch.__version__) >= version.parse("2.3.0"):
    from torch.amp import GradScaler
else:
    from torch.cuda.amp import GradScaler


def normalize_percentage(tensor, percentile=4, lower_bound = None, upper_bound=None):
    original_shape = tensor.shape
    
    batch_size = tensor.size(0)
    flattened_tensor = tensor.reshape(batch_size, -1)

    factor = percentile/100.
    lower_bound_subtomo = torch.quantile(flattened_tensor, factor, dim=1, keepdim=True)
    upper_bound_subtomo = torch.quantile(flattened_tensor, 1-factor, dim=1, keepdim=True)

    if lower_bound is None: 
        normalized_flattened = (flattened_tensor - lower_bound_subtomo) / (upper_bound_subtomo - lower_bound_subtomo)
        normalized_flattened = normalized_flattened.view(original_shape)
    else:
        normalized_flattened = (flattened_tensor - lower_bound) / (upper_bound - lower_bound)
        normalized_flattened = normalized_flattened.view(original_shape)        
    
    return normalized_flattened, lower_bound_subtomo, upper_bound_subtomo


def normalize_mean_std(tensor, mean_val = None, std_val=None, matching = False):
    # merge_factor = 0.99
    mean_subtomo = tensor.mean()
    std_subtomo = tensor.std(correction=0)
    if mean_val is None: 
        return (tensor - mean_subtomo) / std_subtomo, mean_subtomo, std_subtomo
    else:
        if matching:
            return (tensor - mean_subtomo) / std_subtomo * std_val + mean_val, mean_subtomo, std_subtomo 
        else:
            # mean_val = mean_val*merge_factor + mean_subtomo*(1-merge_factor)
            # std_val = std_val*merge_factor + std_subtomo*(1-merge_factor)
            return (tensor - mean_val) / std_val, mean_subtomo, std_subtomo 
    
    # not (tensor - mean_val) / std_val
    
    # return normalized_flattened, lower_bound_subtomo, upper_bound_subtomo
def cross_correlate(M1, M2):
    M1_norm = (M1-M1.mean()) / M1.std(correction = False)
    M2_norm = (M2-M2.mean()) / M2.std(correction = False)
    return (M1_norm*M2_norm).mean()


def rotate_vol_90(volume, rotation):
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
    if len(batch) == 7:
        return [b.cuda() for b in batch]
    return batch[0].cuda(), batch[1].cuda(), None, None, None, None, None

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

    move_norm = training_params["move_norm"]

    for epoch in range(training_params['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad() 
        with tqdm(total=total_steps, unit="batch", disable=(rank!=0),desc=f"Epoch {epoch+1}") as progress_bar:
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_inside_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_outside_loss = torch.tensor(0, dtype=torch.float).to(rank)

            for i_batch, batch in enumerate(train_loader):  
                x1, x2, gt, mw, ctf, wiener, noise_vol = process_batch(batch)

                if training_params['CTF_mode'] in  ["phase_only", 'wiener','network']:
                    if not training_params["phaseflipped"]:
                        ctf = torch.abs(ctf)
                        wiener = torch.abs(wiener) 
                    else:
                        if training_params['do_phaseflip_input']:
                            x1 = apply_F_filter_torch(x1, torch.sign(ctf))
                            x2 = apply_F_filter_torch(x2, torch.sign(ctf))
                            ctf = torch.abs(ctf)
                            wiener = torch.abs(wiener) 

                # if training_params['method'] in ["n2n", "regular"]:   
                #     x1,_,_ = normalize_percentage(x1)    
                #     x2,_,_ = normalize_percentage(x2)                                    
                #     with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                #         preds = model(x1)
                #         preds = preds.to(torch.float32)

                #     if training_params['CTF_mode']  == 'network':
                #         preds = apply_F_filter_torch(preds, ctf)
                #     elif training_params['CTF_mode']  == 'wiener':
                #         x2 = apply_F_filter_torch(x2, wiener)

                #     loss = loss_func(x2, preds)

                #     if rank == 0 and i_batch%100 == 0 :
                #         debug_matrix(ctf, filename=f"{training_params['output_dir']}/debug_ctf_{i_batch}.mrc")
                #         debug_matrix(preds, filename=f"{training_params['output_dir']}/debug_preds_{i_batch}.mrc")
                #         debug_matrix(x1, filename=f"{training_params['output_dir']}/debug_x1_{i_batch}.mrc")
                #         debug_matrix(x2, filename=f"{training_params['output_dir']}/debug_x2_{i_batch}.mrc")
                    
                #     outside_loss = loss
                #     inside_loss = loss

                if random.random()<training_params['random_rot_weight']:
                    rotate_func = rotate_vol_around_axis_torch
                    rot = generate_random_rotation()
                else:
                    rotate_func = rotate_vol_90
                    rot = random.choice(rotation_list_90)

                if training_params['method'] in ["isonet2"]:
                    # x1half1 = x1[:, :, :, ::2, :]
                    # x1half2 = x1[:, :, :, 1::2, :]
                    # noise_std = torch.std(x1half1-x1half2)/1.414

                    x1 = apply_F_filter_torch(x1, mw)

                    x1_std_org, x1_mean_org = x1.std(correction=0,dim=(-3,-2,-1), keepdim=True), x1.mean(dim=(-3,-2,-1), keepdim=True)

                    with torch.no_grad():
                        with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                            preds_x1 = model(x1).to(torch.float32)

                    # preds_x1half1 = preds_x1[:, :, :, ::2, :]
                    # preds_x1half2 = preds_x1[:, :, :, 1::2, :]
                    # new_noise_std = torch.std(preds_x1half1-preds_x1half2)/1.414
                    # delta_noise_std = torch.sqrt(torch.abs(noise_std**2 - new_noise_std**2))

                    # preds_x1 = preds_x1 + torch.randn_like(preds_x1) * delta_noise_std

                    if training_params['CTF_mode'] in ['network', 'wiener']:
                        preds_x1 = apply_F_filter_torch(preds_x1, ctf)

                    x1_filled = apply_F_filter_torch(preds_x1, 1-mw) + x1 

                    x1_filled_rot = rotate_func(x1_filled, rot)

                    x1_filled_rot_mw = apply_F_filter_torch(x1_filled_rot, mw)

                    x1_filled_rot_mw_mean, x1_filled_rot_mw_std = x1_filled_rot_mw.mean(dim=(-3,-2,-1), keepdim=True), x1_filled_rot_mw.std(correction=0,dim=(-3,-2,-1), keepdim=True)

                    x1_filled_rot_mw = (x1_filled_rot_mw-x1_filled_rot_mw_mean)/x1_filled_rot_mw_std * x1_std_org + x1_mean_org


                    rotated_mw = rotate_func(mw, rot)

                    net_input = x1_filled_rot_mw
                    net_target = x1_filled_rot

                    if training_params["noise_level"] > 0:
                        perm = torch.randperm(noise_vol.size(3), device=noise_vol.device)
                        noise_vol = noise_vol[:, :, :, perm, :]
                        net_input = net_input + training_params["noise_level"] * (noise_vol - noise_vol.mean()) / torch.std(noise_vol, correction=0) #* random.random()

                    with torch.autocast('cuda', enabled = training_params["mixed_precision"]): 
                        pred_y = model(net_input).to(torch.float32)

                    if training_params['CTF_mode']  == 'network':
                        pred_y = apply_F_filter_torch(pred_y, ctf)
                    elif training_params['CTF_mode']  == 'wiener':
                        net_target = apply_F_filter_torch(net_target, wiener)

                    outside_loss, inside_loss = masked_loss(pred_y, net_target, rotated_mw, mw, loss_func = loss_func)

                    loss = inside_loss + training_params['mw_weight'] * outside_loss if training_params['mw_weight'] > 0 else loss_func(pred_y, net_target)
                             
                elif training_params['method'] in ['isonet2-n2n']:
                    # x1_std_org, x1_mean_org = x1.std(correction=0,dim=(-3,-2,-1), keepdim=True), x1.mean(dim=(-3,-2,-1), keepdim=True)
                    # x1,_,_ = normalize_mean_std(x1)
                    # x2,_,_ = normalize_mean_std(x2)
                    noise_std = torch.std(x1-x2)/1.414
                    x1 = apply_F_filter_torch(x1, mw)
                    x2 = apply_F_filter_torch(x2, mw)

                    x1_std_org, x1_mean_org = x1.std(correction=0,dim=(-3,-2,-1), keepdim=True), x1.mean(dim=(-3,-2,-1), keepdim=True)
                    x2_std_org, x2_mean_org = x2.std(correction=0,dim=(-3,-2,-1), keepdim=True), x2.mean(dim=(-3,-2,-1), keepdim=True)

                    with torch.no_grad():
                        with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                            preds_x1 = model(x1).to(torch.float32)
                            preds_x2 = model(x2).to(torch.float32)

                    new_noise_std = torch.std(preds_x1-preds_x2)/1.414
                    delta_noise_std = torch.sqrt(torch.abs(noise_std**2 - new_noise_std**2))

                    preds_x1 = preds_x1 + torch.randn_like(preds_x1) * delta_noise_std
                    preds_x2 = preds_x2 + torch.randn_like(preds_x2) * delta_noise_std

                    if training_params['CTF_mode'] in ['network', 'wiener']:
                        preds_x1 = apply_F_filter_torch(preds_x1, ctf)
                        preds_x2 = apply_F_filter_torch(preds_x2, ctf)

                    x1_filled = apply_F_filter_torch(preds_x1, 1-mw) + x1#apply_F_filter_torch(x1, mw)
                    x2_filled = apply_F_filter_torch(preds_x2, 1-mw) + x2#apply_F_filter_torch(x2, mw)

                    x1_filled_rot = rotate_func(x1_filled, rot)
                    x2_filled_rot = rotate_func(x2_filled, rot)

                    x1_filled_rot_mw = apply_F_filter_torch(x1_filled_rot, mw)
                    x2_filled_rot_mw = apply_F_filter_torch(x2_filled_rot, mw)

                    x1_filled_rot_mw_mean, x1_filled_rot_mw_std = x1_filled_rot_mw.mean(dim=(-3,-2,-1), keepdim=True), x1_filled_rot_mw.std(correction=0,dim=(-3,-2,-1), keepdim=True)
                    x2_filled_rot_mw_mean, x2_filled_rot_mw_std = x2_filled_rot_mw.mean(dim=(-3,-2,-1), keepdim=True), x2_filled_rot_mw.std(correction=0,dim=(-3,-2,-1), keepdim=True)

                    x1_filled_rot_mw = (x1_filled_rot_mw-x1_filled_rot_mw_mean)/x1_filled_rot_mw_std * x1_std_org + x1_mean_org
                    x2_filled_rot_mw = (x2_filled_rot_mw-x2_filled_rot_mw_mean)/x2_filled_rot_mw_std * x2_std_org + x2_mean_org


                    rotated_mw = rotate_func(mw, rot)
                    
                    net_input1 = x1_filled_rot_mw
                    net_input2 = x2_filled_rot_mw

                    net_target1 = x1_filled_rot
                    net_target2 = x2_filled_rot

                    if training_params["noise_level"] > 0:
                        perm = torch.randperm(noise_vol.size(3), device=noise_vol.device)
                        noise_vol = noise_vol[:, :, :, perm, :]
                        N = training_params["noise_level"] * (noise_vol - noise_vol.mean()) / torch.std(noise_vol, correction=0) #* random.random()
                        net_input1 = net_input1 + N
                        net_input2 = net_input2 + N


                    with torch.autocast('cuda', enabled = training_params["mixed_precision"]): 
                        pred_y1 = model(net_input1).to(torch.float32)
                        pred_y2 = model(net_input2).to(torch.float32)

                    if training_params['CTF_mode']  == 'network':
                        pred_y1 = apply_F_filter_torch(pred_y1, ctf)
                        pred_y2 = apply_F_filter_torch(pred_y2, ctf)
                    elif training_params['CTF_mode']  == 'wiener':
                        net_target1 = apply_F_filter_torch(net_target1, wiener)
                        net_target2 = apply_F_filter_torch(net_target2, wiener)

                    outside_loss1, inside_loss1 = masked_loss(pred_y1, net_target2, rotated_mw, mw, loss_func = loss_func)
                    outside_loss2, inside_loss2 = masked_loss(pred_y2, net_target1, rotated_mw, mw, loss_func = loss_func)
                    outside_loss = (outside_loss1 + outside_loss2)/2.
                    inside_loss = (inside_loss1+ inside_loss2)/2.
                    
                    loss1 = loss_func(pred_y1, net_target2)
                    loss2 = loss_func(pred_y2, net_target1)
                    

                    if training_params['mw_weight'] > 0:
                        loss = inside_loss + training_params['mw_weight'] * outside_loss
                    else:
                        loss = (loss1 + loss2)/2.

                    # if len(gt.shape) > 2:
                    #     gt_inside_loss = cross_correlate(apply_F_filter_torch(gt, mw), apply_F_filter_torch(preds, mw))
                    #     gt_outside_loss = cross_correlate(apply_F_filter_torch(gt, 1-mw), apply_F_filter_torch(preds, 1-mw))
                    #     inside_loss = gt_inside_loss
                    #     outside_loss = gt_outside_loss     
 


                # if rank == 0 and i_batch%100 == 0 :
                #     print(delta_noise_std, noise_std, new_noise_std)

                #     debug_matrix(x2, filename=f"{training_params['output_dir']}/debug_x2_{i_batch}.mrc")
                #     debug_matrix(gt, filename=f"{training_params['output_dir']}/debug_gt_{i_batch}.mrc")
                #     debug_matrix(net_input1, filename=f"{training_params['output_dir']}/debug_net_input1_{i_batch}.mrc")
                #     debug_matrix(ctf, filename=f"{training_params['output_dir']}/debug_ctf_{i_batch}.mrc")

                #     if training_params["noise_level"] > 0:
                #         debug_matrix(noise_vol, filename=f"{training_params['output_dir']}/debug_noise_vol_{i_batch}.mrc")

                #     debug_matrix(x1_filled, filename=f"{training_params['output_dir']}/debug_x1_filled_{i_batch}.mrc")
                #     debug_matrix(x1_filled_rot, filename=f"{training_params['output_dir']}/debug_x1_filled_rot_{i_batch}.mrc")
                #     debug_matrix(x1_filled_rot_mw, filename=f"{training_params['output_dir']}/debug_x1_filled_rot_mw_{i_batch}.mrc")
                #     debug_matrix(x2_filled_rot, filename=f"{training_params['output_dir']}/debug_x2_filled_rot_{i_batch}.mrc")

                #     # debug_matrix(preds_x1_preCTF, filename=f"{training_params['output_dir']}/debug_preds_prectf_{i_batch}.mrc")

                #     debug_matrix(preds_x1, filename=f"{training_params['output_dir']}/debug_preds_{i_batch}.mrc")
                #     debug_matrix(pred_y1, filename=f"{training_params['output_dir']}/debug_pred_y1_{i_batch}.mrc")
                #     debug_matrix(preds_x2, filename=f"{training_params['output_dir']}/debug_preds_x2_{i_batch}.mrc")

                #     debug_matrix(x1, filename=f"{training_params['output_dir']}/debug_x1_{i_batch}.mrc")
                # if rank == 0 and i_batch%100 == 0 :
                #     # debug_matrix(x2, filename=f"{training_params['output_dir']}/debug_x2_{i_batch}.mrc")
                #     debug_matrix(gt, filename=f"{training_params['output_dir']}/debug_gt_{i_batch}.mrc")
                #     # debug_matrix(net_input1, filename=f"{training_params['output_dir']}/debug_net_input1_{i_batch}.mrc")
                #     if training_params["noise_level"] > 0:
                #         debug_matrix(noise_vol, filename=f"{training_params['output_dir']}/debug_noise_vol_{i_batch}.mrc")

                #     debug_matrix(x1_filled, filename=f"{training_params['output_dir']}/debug_x1_filled_{i_batch}.mrc")
                #     debug_matrix(x1_filled_rot, filename=f"{training_params['output_dir']}/debug_x1_filled_rot_{i_batch}.mrc")
                #     debug_matrix(x1_filled_rot_mw, filename=f"{training_params['output_dir']}/debug_x1_filled_rot_mw_{i_batch}.mrc")
                #     # debug_matrix(x2_filled_rot, filename=f"{training_params['output_dir']}/debug_x2_filled_rot_{i_batch}.mrc")

                #     debug_matrix(pred_y, filename=f"{training_params['output_dir']}/debug_preds_{i_batch}.mrc")
                #     # debug_matrix(pred_y1, filename=f"{training_params['output_dir']}/debug_pred_y1_{i_batch}.mrc")
                #     # debug_matrix(preds_x2, filename=f"{training_params['output_dir']}/debug_preds_x2_{i_batch}.mrc")

                #     debug_matrix(x1, filename=f"{training_params['output_dir']}/debug_x1_{i_batch}.mrc")                

                loss = loss / training_params['acc_batches']
                inside_loss = inside_loss / training_params['acc_batches']
                outside_loss = outside_loss / training_params['acc_batches']

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
                optimizer.zero_grad() 

                if rank == 0 and ( (i_batch+1)%training_params['acc_batches'] == 0 ):        
                    loss_str = (
                        f"Loss: {loss.item():6.6f}"
                    )
                    progress_bar.set_postfix_str(loss_str)
                    progress_bar.update()

                average_loss += loss.item()
                average_inside_loss += inside_loss.item()
                average_outside_loss += outside_loss.item()
                
                if i_batch + 1 >= steps_per_epoch_train*training_params['acc_batches']:
                    break
        optimizer.step()            
        scheduler.step()

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
            
            print(f"Epoch [{epoch+1:3d}/{training_params['epochs']:3d}], "
                f"Loss: {average_loss:6.10f}, "
                f"inside_loss: {average_inside_loss:6.10f}, "
                f"outside_loss: {average_outside_loss:6.10f}, "
                f"learning_rate: {scheduler.get_last_lr()[0]:.4e}")

            plot_metrics(training_params["metrics"],f"{training_params['output_dir']}/loss_{training_params['split']}.png")
            if world_size > 1:
                model_params = model.module.state_dict()
            else:
                model_params = model.state_dict()
            
            torch.save({
                    'method':training_params['method'],
                    'CTF_mode': training_params['CTF_mode'],
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
            if rank == 0:
                write_mrc('testIN.mrc', batch_input[0][0].cpu().numpy().astype(np.float32))
                write_mrc('testOUT.mrc', batch_output[0][0].numpy().astype(np.float32))
            outputs.append(batch_output)

    output = torch.cat(outputs, dim=0).cpu().numpy().astype(np.float32)
    rank_output_path = f"{tmp_data_path}_rank_{rank}.npy"
    np.save(rank_output_path, output)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

