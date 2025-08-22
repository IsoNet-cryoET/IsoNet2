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
from IsoNet.utils.rotations import rotation_list, sample_rot_axis_and_angle, rotate_vol_around_axis_torch
import torch.optim.lr_scheduler as lr_scheduler
import shutil
from packaging import version
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

    for epoch in range(training_params['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad() 
        with tqdm(total=total_steps, unit="batch", disable=(rank!=0),desc=f"Epoch {epoch+1}") as progress_bar:
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_inside_mw_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_outside_mw_loss = torch.tensor(0, dtype=torch.float).to(rank)

            for i_batch, batch in enumerate(train_loader):  
                x1, x2, gt, mw, ctf, wiener, noise_vol = process_batch(batch)
                if training_params["phaseflipped"]:
                    ctf = torch.abs(ctf)
                    wiener = torch.abs(wiener)

                if training_params['CTF_mode'] == "phase_only":
                    x1 = apply_F_filter_torch(x1, torch.sign(ctf))
                    x2 = apply_F_filter_torch(x2, torch.sign(ctf))
                elif  training_params['CTF_mode'] == 'wiener':
                    x1 = apply_F_filter_torch(x1, torch.sign(ctf))
                    x2 = apply_F_filter_torch(x2, wiener)
                    

                if training_params['method'] in ["n2n", "regular"]:                       
                    with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                        preds = model(x1)
                        preds = preds.to(torch.float32)

                    if training_params['CTF_mode']=='network':    
                        preds = apply_F_filter_torch(preds, ctf)

                    loss = loss_func(x2, preds)

                    if rank == 0 and i_batch%100 == 0 :
                        debug_matrix(ctf, filename=f"{training_params['output_dir']}/debug_ctf_{i_batch}.mrc")
                        debug_matrix(preds, filename=f"{training_params['output_dir']}/debug_preds_{i_batch}.mrc")
                        debug_matrix(x1, filename=f"{training_params['output_dir']}/debug_x1_{i_batch}.mrc")
                        debug_matrix(x2, filename=f"{training_params['output_dir']}/debug_x2_{i_batch}.mrc")

                    
                    outside_mw_loss = loss
                    inside_mw_loss = loss

                elif training_params['method'] in ["isonet2",'isonet2-n2n']:

                    if training_params['random_rotation'] == True and random.random()<0:
                        rotate_func = rotate_vol_around_axis_torch
                        rot = sample_rot_axis_and_angle()
                    else:
                        rotate_func = rotate_vol
                        rot = random.choice(rotation_list)

                    # x1_std_org, x1_mean_org = x1.std(correction=0,dim=(-3,-2,-1), keepdim=True), x1.mean(dim=(-3,-2,-1), keepdim=True)

                    # with torch.no_grad():
                    #     with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                    #         preds = model(x1)
                    # preds = preds.to(torch.float32)
                    # if 'CTF_mode' in training_params:
                    #     if training_params['CTF_mode'] in ['wiener']:
                    #         preds = apply_F_filter_torch(preds, torch.abs(ctf))

                    #     if training_params['CTF_mode'] == 'network':
                    #         preds = apply_F_filter_torch(preds, ctf)

                    # subtomos = apply_F_filter_torch(preds, 1-mw) + x1
                    # rotated_subtomo = rotate_func(subtomos, rot)
                    # mw_rotated_subtomos=apply_F_filter_torch(rotated_subtomo,mw)
                    
                    # mw_rotated_subtomos_std = mw_rotated_subtomos.std(correction=0,dim=(-3,-2,-1), keepdim=True)
                    # mw_rotated_subtomos = mw_rotated_subtomos/mw_rotated_subtomos_std * x1_std_org

                    # mw_rotated_subtomos = rotate_func(x1_org, rot)
                    # mw_rotated_subtomos=apply_F_filter_torch(mw_rotated_subtomos,mw)
                    # mw_rotated_subtomos = apply_F_filter_torch(x1_org,mw)

                    
                    # outside_mw_mask = rotated_mw * mw
                    # outside_mw_tomo = apply_F_filter_torch(x1_rot, outside_mw_mask)
                    # outside_mw_mask = rotated_mw * mw
                    # outside_mw_tomo = apply_F_filter_torch(x1_rot, outside_mw_mask)

                    # x1_rot = rotate_func(x1_org, rot)
                    # rotated_mw = rotate_func(mw, rot)
                    # x1_mw = apply_F_filter_torch(x1_rot, mw)
                    # x1_mw = apply_F_filter_torch(x1_org, mw)
                    # x1_mw = apply_F_filter_torch(x1_org, mw)
                    # x1_std_org, x1_mean_org = x1.std(correction=0,dim=(-3,-2,-1), keepdim=True), x1.mean(dim=(-3,-2,-1), keepdim=True)
                    # x1 = (x1-x1_mean_org)/x1_std_org
                    # x = x1
                    # x1 = x1 + torch.randn((64, 64, 64)).cuda() * 0.5
                    # x2 = x2 + torch.randn((64, 64, 64)).cuda() * 0.5
                    # print('x1shape',x1.shape)
                    # print(tomo_index)
                    
                    # indices = tomo_index.tolist()   # convert tensor([1, 3, 5]) -> [1,3,5]
                    # vals = [norm_val[i] for i in indices]
                    # print(vals)
                    #x1 = x1 + torch.randn((64, 64, 64)).cuda()*10
                    #x2 = x2 + torch.randn((64, 64, 64)).cuda()*10
                    x1 = apply_F_filter_torch(x1, mw)
                    x2 = apply_F_filter_torch(x2, mw)
                    x1_std_org, x1_mean_org = x1.std(correction=0,dim=(-3,-2,-1), keepdim=True), x1.mean(dim=(-3,-2,-1), keepdim=True)
                    x1,_,_ = normalize_percentage(x1)
                    x2,_,_ = normalize_percentage(x2)

                    # for bs_index in range(x1.shape[0]): 
                    #     x1[bs_index], mean_x1, std_x1 = normalize_mean_std(x1[bs_index])
                    #     x2[bs_index], mean_x2, std_x2 = normalize_mean_std(x2[bs_index])
                    # for bs_index in range(x1.shape[0]):
                    #     if rank == 0:
                    #         print(bs_index)
                    #         print(vals[indices[bs_index]])
                    #     if vals[indices[bs_index]][0] is None:
                    #         x1[bs_index], mean_x1, std_x1 = normalize_mean_std(x1[bs_index])
                    #         x2[bs_index], mean_x2, std_x2 = normalize_mean_std(x2[bs_index])
                    #         if rank == 0:
                    #             print((mean_x1 + mean_x2)/2)
                    #             print(vals[indices[bs_index]][0])
                    #         vals[indices[bs_index]][0] = (mean_x1 + mean_x2)/2
                    #         vals[indices[bs_index]][1] = (std_x1 + std_x2)/2
                    #         if rank == 0:
                    #             print((mean_x1 + mean_x2)/2)
                    #             print(vals[indices[bs_index]][0])
                    #     else:
                    #         x1[bs_index], mean_x1, std_x1 = normalize_mean_std(x1[bs_index],
                    #                                                             mean_val = vals[indices[bs_index]][0], 
                    #                                                             std_val = vals[indices[bs_index]][1])
                    #         x2[bs_index], mean_x2, std_x2 = normalize_mean_std(x2[bs_index],
                    #                                                             mean_val = vals[indices[bs_index]][0], 
                    #                                                             std_val = vals[indices[bs_index]][1])
                    #         vals[indices[bs_index]][0] = 0.9 * vals[indices[bs_index]][0]  + 0.1*(mean_x1 + mean_x2)/2
                    #         vals[indices[bs_index]][1] = 0.9 * vals[indices[bs_index]][1] + 0.1*(std_x1 + std_x2)/2

                    with torch.no_grad():
                        with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                            preds = model(x1)
                            preds_x2 = model(x2)
                    # preds = normalize_percentage(preds)
                    preds = preds.to(torch.float32)
                    preds_x2 = preds_x2.to(torch.float32)
                    preds,_,_ = normalize_percentage(preds)
                    preds_x2,_,_ = normalize_percentage(preds)


                    x1_filled = apply_F_filter_torch(preds, 1-mw) + x1
                    x2_filled = apply_F_filter_torch(preds_x2, 1-mw) + x2
                    x1_filled,_,_ = normalize_percentage(x1_filled)
                    x2_filled,_,_ = normalize_percentage(x2_filled)


                    x1_filled_rot = rotate_func(x1_filled, rot)
                    x1_filled_rot_mw = apply_F_filter_torch(x1_filled_rot, mw)

                    # for bs_index in range(x1.shape[0]):
                    #     # x1_filled_rot_mw[bs_index],_,_ = normalize_percentage(x1_filled_rot_mw[bs_index],
                    #     #                                         lower_bound=vals[indices[bs_index]][0],
                    #     #                                         upper_bound=vals[indices[bs_index]][1])
                    #     x1_filled_rot_mw[bs_index],_,_ = normalize_mean_std(x1_filled_rot_mw[bs_index],
                    #                                                         mean_val = vals[indices[bs_index]][0],
                    #                                                         std_val = vals[indices[bs_index]][1])

                    # for bs_index in range(x1.shape[0]):
                    #     x1_filled_rot_mw[bs_index],_,_ = normalize_mean_std(x1_filled_rot_mw[bs_index])
                    # for bs_index in range(x1.shape[0]):
                    #     x1_filled_rot_mw[bs_index], _, _ = normalize_mean_std(x1_filled_rot_mw[bs_index],
                    #                                                             mean_val = vals[indices[bs_index]][0], 
                    #                                                             std_val = vals[indices[bs_index]][1],
                    #                                                             matching = True)
                    # x1_filled_rot_mw_mean, x1_filled_rot_mw_std = x1_filled_rot_mw.mean(dim=(-3,-2,-1), keepdim=True), x1_filled_rot_mw.std(correction=0,dim=(-3,-2,-1), keepdim=True)
                    # x1_filled_rot_mw = (x1_filled_rot_mw-x1_filled_rot_mw_mean)/x1_filled_rot_mw_std * x1_std_org + x1_mean_org


                    x2_rot = rotate_func(x2, rot)
                    x2_filled_rot = rotate_func(x2_filled, rot)
                    rotated_mw = rotate_func(mw, rot)
                    
                    net_input = x1_filled_rot_mw
                    # net_input_std, net_input_org = net_input.std(correction=0,dim=(-3,-2,-1), keepdim=True), net_input.mean(dim=(-3,-2,-1), keepdim=True)
                    # net_input = (net_input-net_input_org)/net_input_std
                    net_target = x2_filled_rot
                    # net_input,_,_ = normalize_percentage(net_input)

                    if training_params["noise_level"] > 0:
                        # noise_vol = apply_F_filter_torch(noise_vol, mw)
                        #net_input = net_input + x1_std_org * (noise_vol - noise_vol.mean())  * training_params["noise_level"] / torch.std(noise_vol, correction=0) #* random.random()
                        # Generate a permutation of Y indices
                        perm = torch.randperm(noise_vol.size(3), device=noise_vol.device)

                        # Apply the permutation to shuffle along Y
                        noise_vol = noise_vol[:, :, :, perm, :]
                        net_input = net_input + training_params["noise_level"] * (noise_vol - noise_vol.mean()) / torch.std(noise_vol, correction=0) #* random.random()

                    with torch.autocast('cuda', enabled = training_params["mixed_precision"]): 
                        pred_y = model(net_input).to(torch.float32)

                        # if training_params['CTF_mode'] == 'network':
                        #     pred_y = apply_F_filter_torch(pred_y, ctf)

                        if rank == 0 and i_batch%100 == 0 :
                            # debug_matrix(outside_mw_tomo, filename=f"{training_params['output_dir']}/debug_outside_mw_tomo_{i_batch}.mrc")

                            # debug_matrix(x1_org, filename=f"{training_params['output_dir']}/debug_x1_org_{i_batch}.mrc")
                            # print(vals)
                            debug_matrix(x2, filename=f"{training_params['output_dir']}/debug_x2_{i_batch}.mrc")
                            if len(gt.shape) > 2:
                                debug_matrix(gt, filename=f"{training_params['output_dir']}/debug_gt_{i_batch}.mrc")
                            debug_matrix(net_input, filename=f"{training_params['output_dir']}/debug_net_input_{i_batch}.mrc")
                            debug_matrix(noise_vol, filename=f"{training_params['output_dir']}/debug_noise_vol_{i_batch}.mrc")

                            debug_matrix(x1_filled, filename=f"{training_params['output_dir']}/debug_x1_filled_{i_batch}.mrc")
                            debug_matrix(x1_filled_rot, filename=f"{training_params['output_dir']}/debug_x1_filled_rot_{i_batch}.mrc")
                            debug_matrix(x1_filled_rot_mw, filename=f"{training_params['output_dir']}/debug_x1_filled_rot_mw_{i_batch}.mrc")
                            debug_matrix(x2_filled_rot, filename=f"{training_params['output_dir']}/debug_x2_filled_rot_{i_batch}.mrc")

                            debug_matrix(preds, filename=f"{training_params['output_dir']}/debug_preds_{i_batch}.mrc")
                            debug_matrix(pred_y, filename=f"{training_params['output_dir']}/debug_pred_y_{i_batch}.mrc")
                            debug_matrix(preds_x2, filename=f"{training_params['output_dir']}/debug_preds_x2_{i_batch}.mrc")

                            debug_matrix(x1, filename=f"{training_params['output_dir']}/debug_x1_{i_batch}.mrc")
                            # debug_matrix(mw_rotated_subtomos, filename=f"{training_params['output_dir']}/debug_mw_rotated_subtomos_{i_batch}.mrc")
                            # debug_matrix(x1_rot, filename=f"{training_params['output_dir']}/debug_x1_rot.mrc")
                            # debug_matrix(rotated_mw, filename=f"{training_params['output_dir']}/debug_rotated_mw.mrc")

                        if training_params['method'] ==  'isonet2':
                            loss = loss_func(pred_y,rotated_subtomo)
                            outside_mw_loss = loss
                            inside_mw_loss = loss                            
                        elif training_params['method'] ==  'isonet2-n2n':
                            # r = 24
                            # outside_mw_loss = loss_func(pred_y[..., pred_y.size(-2)//2-r:pred_y.size(-2)//2+r, pred_y.size(-1)//2-r:pred_y.size(-1)//2+r], \
                            #                            net_target[..., net_target.size(-2)//2-r:net_target.size(-2)//2+r, net_target.size(-1)//2-r:net_target.size(-1)//2+r])

                            # outside_mw_loss = loss_func(pred_y, net_target)
                            # inside_mw_loss = outside_mw_loss

                            outside_mw_loss, inside_mw_loss = masked_loss(pred_y, net_target, rotated_mw, mw, loss_func = loss_func)
                            # training_params['mw_weight'] = 2
                            # loss =  outside_mw_loss + training_params['mw_weight'] * inside_mw_loss# + consistency_loss                             
                            loss = loss_func(pred_y, net_target)
                            if len(gt.shape) > 2:
                                gt_loss = cross_correlate(gt, preds)
                                outside_mw_loss = gt_loss
                                inside_mw_loss = outside_mw_loss


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
                optimizer.zero_grad() 

                if rank == 0 and ( (i_batch+1)%training_params['acc_batches'] == 0 ):        
                    # loss_str = (
                    #     f"Loss: {loss.item():6.4f} | "
                    #     f"in_mw_loss: {inside_mw_loss.item():6.4f} | "
                    #     f"out_mw_loss: {outside_mw_loss.item():6.4f}"
                    # )
                    loss_str = (
                        f"outside_mw_loss: {inside_mw_loss.item()} | "
                    )
                    progress_bar.set_postfix_str(loss_str)
                    progress_bar.update()

                average_loss += loss.item()
                average_inside_mw_loss += inside_mw_loss.item()
                average_outside_mw_loss += outside_mw_loss.item()
                
                if i_batch + 1 >= steps_per_epoch_train*training_params['acc_batches']:
                    break
        optimizer.step()            
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
                f"Loss: {average_loss:6.10f}, "
                f"gt_loss: {average_inside_mw_loss:6.10f}, "
                f"out_mw_loss: {average_outside_mw_loss:6.10f}, "
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

            outputs.append(batch_output)

    output = torch.cat(outputs, dim=0).cpu().numpy().astype(np.float32)
    rank_output_path = f"{tmp_data_path}_rank_{rank}.npy"
    np.save(rank_output_path, output)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()



                        # if option == 1:
                        #     x1 = apply_F_filter_torch(x1, mw)
                        #     x2 = apply_F_filter_torch(x2, mw)
                        #     x1_std_org, x1_mean_org = x1.std(correction=0,dim=(-3,-2,-1), keepdim=True), x1.mean(dim=(-3,-2,-1), keepdim=True)
                        #     x2_std_org, x2_mean_org = x2.std(correction=0,dim=(-3,-2,-1), keepdim=True), x2.mean(dim=(-3,-2,-1), keepdim=True)

                        #     with torch.no_grad():
                        #         with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                        #             preds = model(x1)
                        #             preds2 = model(x2)
                        #     preds = preds.to(torch.float32)
                        #     preds2 = preds2.to(torch.float32)

                        #     if 'CTF_mode' in training_params and training_params['CTF_mode'] not in [None, "None"]:
                        #         preds = apply_F_filter_torch(preds, torch.abs(ctf))

                        #     subtomos1 = apply_F_filter_torch(preds, 1-mw) + x1
                        #     rotated_subtomo1 = rotate_func(subtomos1, rot)

                        #     rotated_subtomo1_sd1 = rotated_subtomo1.std(correction=0,dim=(-3,-2,-1), keepdim=True)
                        #     rotated_subtomo1 = rotated_subtomo1/rotated_subtomo1_sd1 * x1_std_org

                        #     mw_rotated_subtomos1=apply_F_filter_torch(rotated_subtomo1,mw)
                        #     mw_rotated_subtomos_std1 = mw_rotated_subtomos1.std(correction=0,dim=(-3,-2,-1), keepdim=True)
                        #     mw_rotated_subtomos1 = mw_rotated_subtomos1/mw_rotated_subtomos_std1 * x1_std_org

                        #     subtomos2 = apply_F_filter_torch(preds2, 1-mw) + x2
                        #     rotated_subtomo2 = rotate_func(subtomos2, rot)

                        #     rotated_subtomo2_sd2 = rotated_subtomo2.std(correction=0,dim=(-3,-2,-1), keepdim=True)
                        #     rotated_subtomo2 = rotated_subtomo2/rotated_subtomo2_sd2 * x2_std_org 

                        #     mw_rotated_subtomos2=apply_F_filter_torch(rotated_subtomo2,mw)
                        #     mw_rotated_subtomos_std2 = mw_rotated_subtomos2.std(correction=0,dim=(-3,-2,-1), keepdim=True)
                        #     mw_rotated_subtomos2 = mw_rotated_subtomos2/mw_rotated_subtomos_std2 * x2_std_org                    
                            

                        #     if training_params["noise_level"] > 0:
                        #         noise_vol = apply_F_filter_torch(noise_vol, mw)
                        #         mw_rotated_subtomos += x1_std_org * noise_vol * training_params["noise_level"] / torch.std(noise_vol, correction=0) * random.random()

                        #     with torch.autocast('cuda', enabled = training_params["mixed_precision"]): 
                        #         pred_y1 = model(mw_rotated_subtomos1).to(torch.float32)
                        #         pred_y2 = model(mw_rotated_subtomos2).to(torch.float32)

                        #         if training_params['method'] ==  'isonet2':
                        #             loss = loss_func(pred_y,rotated_subtomo)
                        #             outside_mw_loss = loss
                        #             inside_mw_loss = loss                            
                        #         elif training_params['method'] ==  'isonet2-n2n':
                        #             outside_mw_loss = loss_func(pred_y1, rotated_subtomo2)
                        #             inside_mw_loss = loss_func(pred_y2, rotated_subtomo1)
                        #             loss = outside_mw_loss + inside_mw_loss
