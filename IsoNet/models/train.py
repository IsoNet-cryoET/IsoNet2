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


# def apply_filter(data, mwshift):
#     # mwshift [x, y, z]
#     # data [x, y, z]
#     t1 = mwshift*torch.fft.fftn(data)
#     return torch.real(torch.fft.ifftn(t1))#.astype(np.float32)
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


def ddp_train(rank, world_size, port_number, model, training_params):
    #data_path, batch_size, acc_batches, epochs, steps_per_epoch, learning_rate, mixed_precision, model_path
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

    batch_size_gpu = training_params['batch_size'] // (training_params['acc_batches'] * world_size)

    #### preparing data
    # from chatGPT: The DistributedSampler shuffles the indices of the entire dataset, not just the portion assigned to a specific GPU. 
    if training_params['method'] == 'regular':
        from IsoNet.models.data_sequence import Train_sets_regular
        train_dataset = Train_sets_regular(training_params['star_file'])

    elif training_params['method'] in ['n2n', 'isonet2', 'isonet2-n2n']:
        from IsoNet.models.data_sequence import Train_sets_n2n
        if rank == 0:
            print("calculate subtomograms position")
        train_dataset = Train_sets_n2n(training_params['star_file'],method=training_params['method'], 
                                       cube_size=training_params['cube_size'], input_column=training_params['input_column'],  isCTFflipped=training_params['input_column'])
        

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    else:
        train_sampler = None  # No sampler for single GPU

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_gpu, persistent_workers=True,
        num_workers=training_params["ncpus"], pin_memory=True, sampler=train_sampler)
    
    if training_params['compile_model'] == True:
        if torch.__version__ >= "2.0.0":
            GPU_capability = torch.cuda.get_device_capability()
            if GPU_capability[0] >= 7:
                torch.set_float32_matmul_precision('high')
                model = torch.compile(model)
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


    average_loss_list = []
    average_inside_mw_loss_list = []
    average_outside_mw_loss_list = []

    for epoch in range(training_params['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        with tqdm(total=total_steps, unit="batch", disable=(rank!=0)) as progress_bar:
            
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_inside_mw_loss = torch.tensor(0, dtype=torch.float).to(rank)
            average_outside_mw_loss = torch.tensor(0, dtype=torch.float).to(rank)
            for i_batch, batch in enumerate(train_loader):  
                
                if training_params['method'] in ["n2n", "regular"]:
                    x1, x2 = batch[0], batch[1]
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    optimizer.zero_grad(set_to_none=True)          
                    if training_params["mixed_precision"]:
                        with torch.cuda.amp.autocast():  # Mixed precision forward pass
                            preds = model(x1)
                            loss = loss_func(x2, preds)    
                    else:                            
                        preds = model(x1)  
                        loss = loss_func(x2,preds)

                elif training_params['method'] in ["isonet2",'isonet2-n2n']:
                    # x [B, C, Z, Y, X]
                    x1, x2, mw = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

                    if training_params['correct_CTF']:
                        ctf =  batch[3].cuda()                       
                        wiener = batch[4].cuda()

                    if training_params['random_rotation'] == True:
                        rotate = rotate_vol_around_axis_torch
                        rot = sample_rot_axis_and_angle()
                    else:
                        rotate = rotate_vol
                        rot = random.choice(rotation_list)

                    optimizer.zero_grad(set_to_none=True)

                    std_org = x1.std()
                    mean_org = x1.mean()

                    # TODO whether need to apply wedge to x1
                    with torch.no_grad():
                        if training_params["mixed_precision"]:
                            with torch.cuda.amp.autocast():  # Mixed precision forward pass
                                preds = model(x1)
                        else:
                            preds = model(x1)

                    preds = preds.to(torch.float32)
                    if training_params['correct_CTF']:
                        preds = apply_F_filter_torch(preds, ctf)

                    if training_params['apply_mw_x1']:
                        subtomos = apply_F_filter_torch(preds, 1-mw) + apply_F_filter_torch(x1, mw)
                    else:
                        subtomos = apply_F_filter_torch(preds, 1-mw) + x1

                    rotated_subtomo = rotate(subtomos, rot)
                    mw_rotated_subtomos=apply_F_filter_torch(rotated_subtomo,mw)
                    rotated_mw = rotate(mw, rot)
                    x2_rot_0 = rotate(x2, rot)
                    

                    if training_params['correct_CTF']:
                        x2_rot = apply_F_filter_torch(x2_rot_0, wiener)
                    else:
                        x2_rot = x2_rot_0
                        

                    mw_rotated_subtomos = (mw_rotated_subtomos - mw_rotated_subtomos.mean())/mw_rotated_subtomos.std() \
                                                *std_org + mean_org
                    

                    if training_params["mixed_precision"]:
                        with torch.cuda.amp.autocast():  # Mixed precision forward pass
                            pred_y = model(mw_rotated_subtomos).to(torch.float32)
                            outside_mw_loss, inside_mw_loss = masked_loss(pred_y, x2_rot, rotated_mw, mw, loss_func = loss_func)
                            if training_params['mw_weight'] > 0:
                                loss =  outside_mw_loss + training_params['mw_weight'] * inside_mw_loss
                            else:
                                loss =  inside_mw_loss
                            # if training_params['gamma'] > 0:
                            #     loss = masked_loss(pred_y, x2_rot, rotated_mw, mw, mw_weight=training_params['gamma'])
                            # else:
                            #     loss = simple_loss(pred_y,x2_rot,rotated_mw)
                    else:
                        pred_y = model(mw_rotated_subtomos).to(torch.float32)
                        outside_mw_loss, inside_mw_loss = masked_loss(pred_y, x2_rot, rotated_mw, mw, loss_func = loss_func)
                        if training_params['mw_weight'] > 0:
                            loss =  outside_mw_loss + training_params['mw_weight'] * inside_mw_loss
                        else:
                            loss =  inside_mw_loss
                        # if training_params['gamma'] > 0:
                        #     loss = masked_loss(pred_y, x2_rot, rotated_mw, mw, mw_weight=training_params['gamma'])
                        # else:
                        #     loss = simple_loss(pred_y,x2_rot,rotated_mw)
                    # if rank == np.random.randint(0, world_size):
                    #     debug_matrix(x1, filename='debug_x1.mrc')
                    #     debug_matrix(subtomos, filename='debug_subtomos.mrc')
                    #     debug_matrix(pred_y, filename='debug_pred_y.mrc')
                    #     debug_matrix(rotated_subtomo, filename='debug_rotated_subtomo.mrc')
                    #     debug_matrix(mw_rotated_subtomos, filename='debug_mw_rotated_subtomos.mrc')
                    #     debug_matrix(x2_rot_0, filename='debug_x2_rot_0.mrc')
                    #     debug_matrix(x2_rot, filename='debug_x2_rot.mrc')
                    #     debug_matrix(mw, filename='debug_mw.mrc')
                    #     debug_matrix(rotated_mw, filename='debug_rotated_mw.mrc')

                loss = loss / training_params['acc_batches']
                inside_mw_loss = inside_mw_loss / training_params['acc_batches']
                outside_mw_loss = outside_mw_loss / training_params['acc_batches']

                if training_params['mixed_precision']:
                    scaler.scale(loss).backward()  # Scaled backward pass
                else:
                    loss.backward()  # Normal backward pass
                #loss.backward()
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

        # Normalize loss across GPUs
        if world_size > 1:
            dist.barrier()
            dist.reduce(average_loss, dst=0)
            dist.reduce(average_inside_mw_loss, dst=0)
            dist.reduce(average_outside_mw_loss, dst=0)
        average_loss /= (world_size * (i_batch + 1))
        average_inside_mw_loss /= (world_size * (i_batch + 1))
        average_outside_mw_loss /= (world_size * (i_batch + 1))
        # else:
        #     average_loss /= (i_batch + 1)
        #     inside_mw_loss /= (world_size * (i_batch + 1))
        #     outside_mw_loss /= (world_size * (i_batch + 1))
                                      
        
        #dist.reduce(average_loss, dst=0)

        #average_loss =  average_loss / dist.get_world_size()

        if rank == 0:
            average_loss_list.append(average_loss.cpu().numpy())
            average_inside_mw_loss_list.append(average_inside_mw_loss.cpu().numpy())
            average_outside_mw_loss_list.append(average_outside_mw_loss.cpu().numpy())
            outmodel_path = f"{training_params['output_dir']}/network_{training_params['arch']}_{training_params['cube_size']}.pt"
            print(f"Epoch [{epoch+1}/{training_params['epochs']}], Loss:{average_loss:.4f},\
                    in_mw_loss:{average_inside_mw_loss:.4f},\
                    out_mw_loss:{average_outside_mw_loss:.4f},\
                    learning_rate:{scheduler.get_last_lr()[0]:.4e}")

            metrics = {"average_loss":average_loss_list,
                       "inside_mw_loss":average_inside_mw_loss_list,
                       "outside_mw_loss":average_outside_mw_loss_list}
            plot_metrics(metrics,f"{training_params['output_dir']}/loss.png")
            if world_size > 1:
                torch.save({
                    'method':training_params['method'],
                    'arch':training_params['arch'],
                    'model_state_dict': model.module.state_dict(),
                    'metrics': metrics,
                    'cube_size': training_params['cube_size']
                    }, outmodel_path)
            else:
                torch.save({
                    'method':training_params['method'],
                    'arch':training_params['arch'],
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics,
                    'cube_size': training_params['cube_size']
                    }, outmodel_path)                
            if (epoch+1)%training_params['T_max'] == 0:
                outmodel_path_epoch = f"{training_params['output_dir']}/network_{training_params['arch']}_{training_params['cube_size']}_epoch{epoch}.pt"
                shutil.copy(outmodel_path, outmodel_path_epoch)
    if world_size > 1:
        dist.destroy_process_group()


def ddp_predict(rank, world_size, port_number, model, data, tmp_data_path, wedge):

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
            if wedge is not None:
                mw = torch.from_numpy(wedge[np.newaxis,np.newaxis,:,:,:]).to(rank)
                batch_input = apply_F_filter_torch(batch_input, mw)
            batch_output = model(batch_input).cpu()  # Move output to CPU immediately
            if rank == 1:
                debug_matrix(batch_input, filename='debug1_predict1.mrc')
                debug_matrix(batch_output, filename='debug1_predict2.mrc')

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