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

# def apply_filter(data, mwshift):
#     # mwshift [x, y, z]
#     # data [x, y, z]
#     t1 = mwshift*torch.fft.fftn(data)
#     return torch.real(torch.fft.ifftn(t1))#.astype(np.float32)

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
        train_dataset = Train_sets_regular(training_params['data_path'])

    elif training_params['method'] in ['n2n', 'spisonet', 'spisonet-ddw']:
        from IsoNet.models.data_sequence import Train_sets_n2n
        if rank == 0:
            print("calculate subtomograms position")
        train_dataset = Train_sets_n2n(training_params['data_path'],method=training_params['method'], cube_size=training_params['cube_size'])
        from IsoNet.utils.rotations import rotation_list_24
        mwshift = np.ones((training_params['cube_size'])*3, dtype=np.float32)

    # print(train_dataset.tomo_paths_even)
    # print(train_dataset.tomo_paths_odd)

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    else:
        train_sampler = None  # No sampler for single GPU

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_gpu, persistent_workers=True,
        num_workers=4, pin_memory=True, sampler=train_sampler)
    
    # if torch.__version__ >= "2.0.0":
    #     GPU_capability = torch.cuda.get_device_capability()
    #     if GPU_capability[0] >= 7:
    #         torch.set_float32_matmul_precision('high')
    #         model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['learning_rate'])
    loss_fn = nn.MSELoss()
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True

    # if training_params['mixed_precision']:
    #     scaler = torch.cuda.amp.GradScaler()
    
    average_loss_list = []

    steps_per_epoch_train = training_params['steps_per_epoch']
    total_steps = min(len(train_loader)//training_params['acc_batches'], training_params['steps_per_epoch'])

    for epoch in range(training_params['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        with tqdm(total=total_steps, unit="batch", disable=(rank!=0)) as progress_bar:
            
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            for i, batch in enumerate(train_loader):

                # if rank == np.random.randint(0, world_size):
                #     debug_matrix(x1, filename='debug1.mrc')
                #     debug_matrix(x2, filename='debug2.mrc')
                #     debug_matrix(preds, filename='debug3.mrc')             
                
                if training_params['method'] == "n2n" or training_params['method'] == "regular":
                    x1, x2 = batch[0], batch[1]
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    optimizer.zero_grad(set_to_none=True)

                    
                    preds = model(x1)  
                    print(training_params['method'])
                    loss = loss_fn(x2,preds)
                elif training_params['method'] == "spisonet":

                    x1, x2 = batch[0], batch[1]
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    optimizer.zero_grad(set_to_none=True)

                    
                    preds = model(x1)  
                    mw = batch[2].cuda()
                    mwshift = torch.fft.fftshift(mw, dim=(-1, -2, -3))
                    data = torch.zeros_like(preds)
                    for j,d in enumerate(preds):
                        data[j][0] = torch.real(torch.fft.ifftn(mwshift[j]*torch.fft.fftn(d[0])))#.astype(np.float32)
                    loss_consistency_1 = loss_fn(data,x1)

                    if training_params['beta'] > 0:
                        loss_consistency_2 = loss_fn(data,x2)
                        pred_2 = model(x2)
                        data_rot_2 = torch.zeros_like(pred_2)   
                    else:
                        loss_consistency_2 = 0
                        training_params['beta'] = 0

                    data_rot = torch.zeros_like(preds)
                    data_e = torch.zeros_like(preds)
                    for k,d in enumerate(preds):
                        rot = random.choice(rotation_list_24)
                        tmp = torch.rot90(d[0],rot[0][1],rot[0][0])
                        data_rot[k][0] = torch.rot90(tmp,rot[1][1],rot[1][0])
                        if training_params['beta'] > 0:
                            tmp_2 = torch.rot90(pred_2[k][0],rot[0][1],rot[0][0])
                            data_rot_2[k][0] = torch.rot90(tmp_2,rot[1][1],rot[1][0])
                        data_e[k][0] = torch.real(torch.fft.ifftn(mwshift[k]*torch.fft.fftn(data_rot[k][0])))#+noise[i][0]#.astype(np.float32)
                    pred_y = model(data_e)
                    loss_equivariance_1 = loss_fn(pred_y, data_rot)

                    if training_params['beta'] > 0:
                        loss_equivariance_2 = loss_fn(pred_y, data_rot_2)
                    else:
                        loss_equivariance_2 = 0
                    loss = training_params['alpha'] * loss_equivariance_1 + loss_consistency_1 + \
                           training_params['beta'] * ( training_params['alpha'] * loss_equivariance_2 + loss_consistency_2)
                elif training_params['method'] == "spisonet-single":
                    x1, x2 = batch[0], batch[1]
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    optimizer.zero_grad(set_to_none=True)

                    
                    preds = model(x1)  
                    mw = x2
                    mwshift = torch.fft.fftshift(mw, dim=(-1, -2, -3))
                    data = torch.zeros_like(preds)
                    for j,d in enumerate(preds):
                        data[j][0] = torch.real(torch.fft.ifftn(mwshift*torch.fft.fftn(d[0])))#.astype(np.float32)
                    loss_consistency_1 = loss_fn(data,x1)
                    data_rot = torch.zeros_like(preds)
                    data_e = torch.zeros_like(preds)
                    for k,d in enumerate(preds):
                        rot = random.choice(rotation_list_24)
                        tmp = torch.rot90(d[0],rot[0][1],rot[0][0])
                        data_rot[k][0] = torch.rot90(tmp,rot[1][1],rot[1][0])
                        data_e[k][0] = torch.real(torch.fft.ifftn(mwshift*torch.fft.fftn(data_rot[k][0])))#+noise[i][0]#.astype(np.float32)
                    pred_y = model(data_e)
                    loss_equivariance_1 = loss_fn(pred_y, data_rot)
                    loss = training_params['alpha'] * loss_equivariance_1 + loss_consistency_1

                elif training_params['method'] == "spisonet-ddw":
                    x1, x2 = batch[0], batch[1]
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    optimizer.zero_grad(set_to_none=True)

                    #with torch.no_grad():
                    preds = model(x1)  
                    mw = batch[2].cuda()
                    mwshift = torch.fft.fftshift(mw, dim=(-1, -2, -3))
                    if rank == np.random.randint(0, world_size):
                        debug_matrix(mwshift, filename='debugmwshift.mrc')

                    data = torch.zeros_like(preds)
                    for j,d in enumerate(preds):
                        data[j][0] = torch.real(torch.fft.ifftn(mwshift[j]*torch.fft.fftn(d[0])))#.astype(np.float32)
                    loss = loss_fn(data,x2)

                    if training_params['alpha'] > 0:
                        data_combine_rot_mw = torch.empty_like(preds)
                        x2_rot = torch.empty_like(preds)
                        rotated_mw = torch.empty_like(mw)
                        new_data = preds - data + x1
                        for k,d in enumerate(preds):
                            rot = random.choice(rotation_list_24)


                            #outside_mw = torch.real(torch.fft.ifftn((1-mwshift[k])*torch.fft.fftn(preds[k][0])))
                            #inside_mw = torch.real(torch.fft.ifftn(mwshift[k]*torch.fft.fftn(x1[k][0])))
                            tmp = new_data[k][0]#outside_mw + inside_mw                   
                            tmp = torch.rot90(tmp,rot[0][1],rot[0][0])
                            tmp = torch.rot90(tmp,rot[1][1],rot[1][0])
                            data_combine_rot_mw[k][0] = torch.real(torch.fft.ifftn(mwshift[k]*torch.fft.fftn(tmp)))

                            tmp_mw = torch.rot90(mw[k],rot[0][1],rot[0][0])
                            rotated_mw[k] = torch.rot90(tmp_mw,rot[1][1],rot[1][0])

                            tmp_x2_rot = torch.rot90(x2[k][0],rot[0][1],rot[0][0])
                            x2_rot[k][0] = torch.rot90(tmp_x2_rot,rot[1][1],rot[1][0])


                        pred_y = model(data_combine_rot_mw)
                        # if rank == np.random.randint(0, world_size):
                        #     debug_matrix(x1, filename='debug1.mrc')
                        #     debug_matrix(x2, filename='debug2.mrc')
                        #     debug_matrix(preds, filename='debug3.mrc')   
                        #     debug_matrix(x2_rot, filename='debug4.mrc')
                        #     debug_matrix(data_combine_rot_mw, filename='debug5.mrc')
                        #     debug_matrix(pred_y, filename='debug6.mrc')
                        #     debug_matrix(rotated_mw, filename='debug7.mrc')
                        #loss_ddw = masked_loss(pred_y, x2_rot, rotated_mw, torch.ones_like(mw), mw_weight=2.0)
                        loss_ddw = masked_loss(pred_y, x2_rot, rotated_mw, mw, mw_weight=training_params['gamma'])


                        loss += loss_ddw * training_params['alpha']
                                
                
                loss = loss / training_params['acc_batches']
                loss.backward()
                loss_item = loss.item()
                              
                if ( (i+1)%training_params['acc_batches'] == 0 ) or (i+1) == min(len(train_loader), steps_per_epoch_train * training_params['acc_batches']):
                    # if training_params['mixed_precision']:
                    #     pass
                    # else:
                    optimizer.step()

                if rank == 0 and ( (i+1)%training_params['acc_batches'] == 0 ):
                   progress_bar.set_postfix({"Loss": loss_item})#, "t1": time2-time1, "t2": time3-time2, "t3": time4-time3})
                   progress_bar.update()
                average_loss += loss_item
                
                if i + 1 >= steps_per_epoch_train*training_params['acc_batches']:
                    break

        # Normalize loss across GPUs
        if world_size > 1:
            dist.barrier()
            dist.reduce(average_loss, dst=0)
            average_loss /= (world_size * (i + 1))
        else:
            average_loss /= (i + 1)
        
                                      
        
        #dist.reduce(average_loss, dst=0)

        #average_loss =  average_loss / dist.get_world_size()

        if rank == 0:
            average_loss_list.append(average_loss.cpu().numpy())
            print(f"Epoch [{epoch+1}/{training_params['epochs']}], Train Loss: {average_loss:.4f}")
            if world_size > 1:
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'average_loss': average_loss_list,
                    }, training_params['outmodel_path'])
            else:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'average_loss': average_loss_list,
                    }, training_params['outmodel_path'])                
    if world_size > 1:
        dist.destroy_process_group()


def ddp_predict(rank, world_size, port_number, model, data, tmp_data_path):

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