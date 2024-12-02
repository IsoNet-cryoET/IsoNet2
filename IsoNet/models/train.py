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


def rotate_vol(volume, rotation):
    # B, C, Z, Y, X
    new_vol = torch.rot90(volume, rotation[0][1], [rotation[0][0][0]-3,rotation[0][0][1]-3])
    new_vol = torch.rot90(new_vol, rotation[1][1], [rotation[1][0][0]-3,rotation[1][0][1]-3])
    return new_vol

def apply_F_filter_torch(input_map,F_map):
    fft_input = torch.fft.fftn(input_map, dim=(-1, -2, -3))
    mw_shift = torch.fft.fftshift(F_map, dim=(-1, -2, -3))
    out = torch.fft.ifftn(mw_shift*fft_input,dim=(-1, -2, -3))
    out =  np.real(out).real
    return out

def ddp_train(rank, world_size, port_number, model, train_dataset, training_params):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_number
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
    else:
        model = model.to(rank)

    if training_params['compile_model'] == True:
        if torch.__version__ >= "2.0.0":
            GPU_capability = torch.cuda.get_device_capability()
            if GPU_capability[0] >= 7:
                torch.set_float32_matmul_precision('high')
                model = torch.compile(model)


    if rank == 0:
        print("calculate subtomograms position")

    batch_size_gpu = training_params['batch_size'] // (training_params['acc_batches'] * world_size)
       
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size_gpu, persistent_workers=True,
            num_workers=training_params["ncpus"], pin_memory=True, sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size_gpu, persistent_workers=True,
            num_workers=training_params["ncpus"], pin_memory=True, sampler=train_sampler, shuffle=True)




    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['learning_rate'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_params['T_max'], eta_min=training_params['learning_rate_min'])

    if training_params['loss_func'] == "L2":
        loss_func = nn.MSELoss()
    elif  training_params['loss_func']  == "Huber":
        loss_func = nn.HuberLoss()
    
    if training_params['mixed_precision']:
        scaler = torch.cuda.amp.GradScaler()

    steps_per_epoch_train = training_params['steps_per_epoch']
    total_steps = min(len(train_loader)//training_params['acc_batches'], training_params['steps_per_epoch'])

    for epoch in range(training_params['epochs']):
        if train_sampler:
            # What is this
            train_sampler.set_epoch(epoch)
        model.train()

        with tqdm(total=total_steps, unit="batch", disable=(rank!=0)) as progress_bar:
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_inside_mw_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_outside_mw_loss = torch.tensor(0, dtype=torch.float).to(rank)

            for i_batch, batch in enumerate(train_loader):  
                optimizer.zero_grad(set_to_none=True) 

                x1, x2, mw, ctf, wiener = batch[0].cuda(), batch[1].cuda(), \
                                              batch[2].cuda(), batch[3].cuda(), batch[4].cuda()
                # print("correct_CTF",training_params['correct_CTF'])
                if training_params['correct_CTF'] and not training_params["isCTFflipped"]:
                    x1 = apply_F_filter_torch(x1, torch.sign(ctf))
                    x2 = apply_F_filter_torch(x2, wiener)

                if training_params['correct_CTF'] and training_params["isCTFflipped"]:
                    x2 = apply_F_filter_torch(x2, torch.abs(wiener))

                if training_params['method'] in ["n2n", "regular"]:
                    with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                        preds = model(x1)
                        loss = loss_func(x2, preds)

                elif training_params['method'] in ["isonet2",'isonet2-n2n']:

                    if training_params['random_rotation'] == True:
                        rotate_func = rotate_vol_around_axis_torch
                        rot = sample_rot_axis_and_angle()
                    else:
                        rotate_func = rotate_vol
                        rot = random.choice(rotation_list)

                    std_org = x1.std()
                    mean_org = x1.mean()

                    # TODO whether need to apply wedge to x1
                    with torch.no_grad():
                        with torch.autocast("cuda", enabled=training_params["mixed_precision"]): 
                            preds = model(x1)
                    
                    # need to confirm whether to float32 is necessary
                    preds = preds.to(torch.float32)
                    if training_params['correct_CTF']:
                        preds = apply_F_filter_torch(preds, torch.abs(ctf))

                    if training_params['apply_mw_x1']:
                        subtomos = apply_F_filter_torch(preds, 1-mw) + apply_F_filter_torch(x1, mw)
                    else:
                        subtomos = apply_F_filter_torch(preds, 1-mw) + x1

                    rotated_subtomo = rotate_func(subtomos, rot)
                    mw_rotated_subtomos=apply_F_filter_torch(rotated_subtomo,mw)
                    rotated_mw = rotate_func(mw, rot)
                    x2_rot = rotate_func(x2, rot)

                    # This normalization need to be tested
                    mw_rotated_subtomos = (mw_rotated_subtomos - mw_rotated_subtomos.mean())/mw_rotated_subtomos.std() \
                                                *std_org + mean_org
                    
                    with torch.autocast('cuda', enabled = training_params["mixed_precision"]): 
                        pred_y = model(mw_rotated_subtomos).to(torch.float32)
                        outside_mw_loss, inside_mw_loss = masked_loss(pred_y, x2_rot, rotated_mw, mw, loss_func = loss_func)
                        if training_params['mw_weight'] > 0:
                            loss =  outside_mw_loss + training_params['mw_weight'] * inside_mw_loss
                        else:
                            loss =  inside_mw_loss       
                    # if rank == np.random.randint(0, world_size):
                    #     debug_matrix(-x1-x2,filename='debug-_x1_x2.mrc')
                    #     debug_matrix(x1,filename='debug_x1.mrc')
                    #     debug_matrix(preds,filename='debug_preds.mrc')
                    #     debug_matrix(x2,filename='debug_x2.mrc')          
                    #     debug_matrix(subtomos,filename='debug_subtomos.mrc')
                    #     debug_matrix(rotated_subtomo,filename='debug_rotated_subtomo.mrc')   
                    #     debug_matrix(mw_rotated_subtomos,filename='debug_mw_rotated_subtomos.mrc')
                    #     debug_matrix(rotated_mw,filename='rotated_mw.mrc')  
                    #     debug_matrix(pred_y,filename='debug_pred_y.mrc') 
                    #     debug_matrix(x2_rot,filename='debug_x2_rot.mrc')          



                loss = loss / training_params['acc_batches']
                inside_mw_loss = inside_mw_loss / training_params['acc_batches']
                outside_mw_loss = outside_mw_loss / training_params['acc_batches']

                if training_params['mixed_precision']:
                    scaler.scale(loss).backward()  # Scaled backward pass
                else:
                    loss.backward()  # Normal backward pass

                loss_item = loss.item()
                inside_mw_loss_item = inside_mw_loss.item()
                outside_mw_loss_item = outside_mw_loss.item()
            
                              
                if ( (i_batch+1)%training_params['acc_batches'] == 0 ) or (i_batch+1) == min(len(train_loader), steps_per_epoch_train * training_params['acc_batches']):
                    if training_params['mixed_precision']:
                        # Unscale the gradients and apply the optimizer step
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                if rank == 0 and ( (i_batch+1)%training_params['acc_batches'] == 0 ):
                   progress_bar.set_postfix({"Loss": loss_item,"inside_mw_loss": inside_mw_loss_item,"outside_mw_loss": outside_mw_loss_item})#, "t1": time2-time1, "t2": time3-time2, "t3": time4-time3})
                   progress_bar.update()

                average_loss += loss_item
                average_inside_mw_loss += inside_mw_loss_item
                average_outside_mw_loss += outside_mw_loss_item
                
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

            outmodel_path = f"{training_params['output_dir']}/network_{training_params['arch']}_{training_params['cube_size']}_{training_params['split']}.pt"
            print(f"Epoch [{epoch+1}/{training_params['epochs']}], Loss:{average_loss:.4f},\
                    in_mw_loss:{average_inside_mw_loss:.4f},\
                    out_mw_loss:{average_outside_mw_loss:.4f},\
                    learning_rate:{scheduler.get_last_lr()[0]:.4e}")

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
                        
            # if (epoch+1)%training_params['T_max'] == 0:
            #     outmodel_path_epoch = f"{training_params['output_dir']}/network_{training_params['arch']}_{training_params['cube_size']}_epoch{epoch+1}_{training_params['split']}.pt"
            #     shutil.copy(outmodel_path, outmodel_path_epoch)

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
            disable=(rank != 0)
        ):
            batch_input = data[i:i + 1].to(rank)
            if F_mask is not None:
                F_m = torch.from_numpy(F_mask[np.newaxis,np.newaxis,:,:,:]).to(rank)
                batch_input = apply_F_filter_torch(batch_input, F_m)
            batch_output = model(batch_input).cpu()  # Move output to CPU immediately
            # if rank == 1:
            #     debug_matrix(batch_input, filename='debug1_predict1.mrc')
            #     debug_matrix(batch_output, filename='debug1_predict2.mrc')

            outputs.append(batch_output)

    output = torch.cat(outputs, dim=0).cpu().numpy()
    rank_output_path = f"{tmp_data_path}_rank_{rank}.npy"
    np.save(rank_output_path, output)

    dist.barrier()

    if rank == 0:
        all_outputs = []
        for r in range(world_size):
            rank_output_path = f"{tmp_data_path}_rank_{r}.npy"
            rank_output = np.load(rank_output_path)
            all_outputs.append(rank_output)
        
        gathered_outputs = np.concatenate(all_outputs, axis=0)[:num_data_points]
        np.save(tmp_data_path, gathered_outputs)
    
        for r in range(world_size):
            os.remove(f"{tmp_data_path}_rank_{r}.npy")

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