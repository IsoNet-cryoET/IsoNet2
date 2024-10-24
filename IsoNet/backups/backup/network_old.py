import numpy as np

from .unet import Unet
import torch
import os
from .data_sequence import Train_sets, Predict_sets
import mrcfile
from IsoNet.preprocessing.img_processing import normalize
import torch.nn as nn
import logging
from IsoNet.utils.toTile import reform3D
import sys
from tqdm import tqdm
import socket
import copy
import random
from IsoNet.utils.utils import debug_matrix
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
#import torch._dynamo as dynamo
def find_unused_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    _, port = sock.getsockname()
    sock.close()
    return port

from .masked_loss import masked_loss

def ddp_train(rank, world_size, port_number, model,alpha, beta, data_path, batch_size, acc_batches, epochs, steps_per_epoch, learning_rate, mixed_precision, model_path, fsc3d):
    rotation_list_24 = [(((0,1),1),((0,2),0)), (((0,1),1),((0,2),1)), (((0,1),1),((0,2),2)), (((0,1),1),((0,2),3)), 
                    (((0,1),3),((0,2),0)), (((0,1),3),((0,2),1)), (((0,1),3),((0,2),2)), (((0,1),3),((0,2),3)), 
                    (((1,2),1),((0,2),0)), (((1,2),1),((0,2),1)), (((1,2),1),((0,2),2)), (((1,2),1),((0,2),3)), 
                    (((1,2),3),((0,2),0)), (((1,2),3),((0,2),1)), (((1,2),3),((0,2),2)), (((1,2),3),((0,2),3)), 

                    (((0,1),0),((0,2),0)), (((0,1),0),((0,2),1)), (((0,1),0),((0,2),2)), (((0,1),0),((0,2),3)), 
                    (((0,1),2),((0,2),0)), (((0,1),2),((0,2),1)), (((0,1),2),((0,2),2)), (((0,1),2),((0,2),3))]
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_number
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    batch_size = batch_size // acc_batches
    batch_size_gpu = batch_size // world_size

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    model = DDP(model, device_ids=[rank])
    if torch.__version__ >= "2.0.0":
        GPU_capability = torch.cuda.get_device_capability()
        if GPU_capability[0] >= 7:
            torch.set_float32_matmul_precision('high')
            model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    #from chatGPT: The DistributedSampler shuffles the indices of the entire dataset, not just the portion assigned to a specific GPU. 
    train_dataset = Train_sets_sp_n2n(data_path)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_gpu, persistent_workers=True,
                                             num_workers=4, pin_memory=True, sampler=train_sampler)


    steps_per_epoch_train = steps_per_epoch
    total_steps = min(len(train_loader)//acc_batches, steps_per_epoch)
    average_loss_list = []
    loss_fn = nn.L1Loss()
    mw = torch.from_numpy(fsc3d).cuda()
    mwshift = torch.fft.fftshift(mw)
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        with tqdm(total=total_steps, unit="batch", disable=(rank!=0)) as progress_bar:
            model.train()
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            for i, batch in enumerate(train_loader):
                x1, x2 = batch
                x1 = x1.cuda()
                x2 = x2.cuda()
                optimizer.zero_grad(set_to_none=True)
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        preds = model(x1)
                    with torch.cuda.amp.autocast(dtype=torch.float32):
                        data = torch.zeros_like(preds)
                        for i,d in enumerate(preds):
                            data[i][0] = torch.real(torch.fft.ifftn(mwshift*torch.fft.fftn(d[0])))#.astype(np.float32)
                        loss_consistency = loss_fn(data,x1)
                        data_rot = torch.zeros_like(preds)
                        data_e = torch.zeros_like(preds)
                        for j,d in enumerate(preds):
                            rot = random.choice(rotation_list_24)
                            tmp = torch.rot90(d[0],rot[0][1],rot[0][0])
                            data_rot[j][0] = torch.rot90(tmp,rot[1][1],rot[1][0])
                            data_e[j][0] = torch.real(torch.fft.ifftn(mwshift*torch.fft.fftn(data_rot[j][0])))#+noise[i][0]#.astype(np.float32)
                    with torch.cuda.amp.autocast():
                        pred_y = model(data_e)
                        loss_equivariance = loss_fn(pred_y, data_rot)
                        loss = alpha*loss_equivariance + loss_consistency
                        loss = loss / acc_batches
                    scaler.scale(loss).backward()
                else:
                    #noise_level = 4
                    #if rank == 0:
                    #    print("noise_level", noise_level)
                    #noise = torch.rand(1)[0] * noise_level * torch.normal(0, 1, size=x.shape)

                    preds = model(x1)# + noise.cuda())
                    # no_grad_h2 = False
                    # if no_grad_h2:
                    #     with torch.no_grad():
                    #         pred_2 = model(x2)
                    #         data_rot_2 = torch.zeros_like(pred_2, requires_grad=False)
                    # else:
                    #     pred_2 = model(x2)
                    #     data_rot_2 = torch.zeros_like(pred_2)    
                    
                    data = torch.zeros_like(preds)
                    for j,d in enumerate(preds):
                        data[j][0] = torch.real(torch.fft.ifftn(mwshift*torch.fft.fftn(d[0])))#.astype(np.float32)
                    loss_consistency_1 = loss_fn(data,x1)

                    if beta > 0:
                        loss_consistency_2 = loss_fn(data,x2)
                        pred_2 = model(x2)
                        data_rot_2 = torch.zeros_like(pred_2)   
                    else:
                        loss_consistency_2 = 0
                        beta = 0

 

                    data_rot = torch.zeros_like(preds)
                    data_e = torch.zeros_like(preds)
                    for k,d in enumerate(preds):
                        rot = random.choice(rotation_list_24)
                        tmp = torch.rot90(d[0],rot[0][1],rot[0][0])
                        data_rot[k][0] = torch.rot90(tmp,rot[1][1],rot[1][0])
                        if beta > 0:
                            tmp_2 = torch.rot90(pred_2[k][0],rot[0][1],rot[0][0])
                            data_rot_2[k][0] = torch.rot90(tmp_2,rot[1][1],rot[1][0])
                        data_e[k][0] = torch.real(torch.fft.ifftn(mwshift*torch.fft.fftn(data_rot[k][0])))#+noise[i][0]#.astype(np.float32)
                    pred_y = model(data_e)
                    if beta > 0:
                        loss_equivariance_1 = loss_fn(pred_y, data_rot)
                        loss_equivariance_2 = loss_fn(pred_y, data_rot_2)
                    else:
                        loss_equivariance_1 = loss_fn(pred_y, data_rot)
                        loss_equivariance_2 = 0
                    #loss = alpha*loss_equivariance_1 + loss_consistency_1 + beta * (alpha*loss_equivariance_2 + loss_consistency_2)
                    loss = beta * (alpha*loss_equivariance_2 + loss_consistency_2)
                    loss = loss / acc_batches
                    loss.backward()
                loss_item = loss.item()
                              
                if ( (i+1)%acc_batches == 0 ) or (i+1) == min(len(train_loader), steps_per_epoch_train * acc_batches):
                    if mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                if rank == 0 and ( (i+1)%acc_batches == 0 ):
                   progress_bar.set_postfix({"Loss": loss_item})#, "t1": time2-time1, "t2": time3-time2, "t3": time4-time3})
                   progress_bar.update()
                average_loss += loss_item

                
                if i + 1 >= steps_per_epoch_train*acc_batches:
                    break
            average_loss = average_loss / (i+1.)
                                      
        dist.barrier()
        dist.reduce(average_loss, dst=0)

        average_loss =  average_loss / dist.get_world_size()
        if rank == 0:
            average_loss_list.append(average_loss.cpu().numpy())
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {average_loss:.4f}")
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'average_loss': average_loss_list,
                }, model_path)
    dist.destroy_process_group()


def ddp_simple_n2n(rank, world_size, port_number, model,alpha, beta, data_path, batch_size, acc_batches, epochs, steps_per_epoch, learning_rate, mixed_precision, model_path, fsc3d):
    rotation_list_24 = [(((0,1),1),((0,2),0)), (((0,1),1),((0,2),1)), (((0,1),1),((0,2),2)), (((0,1),1),((0,2),3)), 
                    (((0,1),3),((0,2),0)), (((0,1),3),((0,2),1)), (((0,1),3),((0,2),2)), (((0,1),3),((0,2),3)), 
                    (((1,2),1),((0,2),0)), (((1,2),1),((0,2),1)), (((1,2),1),((0,2),2)), (((1,2),1),((0,2),3)), 
                    (((1,2),3),((0,2),0)), (((1,2),3),((0,2),1)), (((1,2),3),((0,2),2)), (((1,2),3),((0,2),3)), 

                    (((0,1),0),((0,2),0)), (((0,1),0),((0,2),1)), (((0,1),0),((0,2),2)), (((0,1),0),((0,2),3)), 
                    (((0,1),2),((0,2),0)), (((0,1),2),((0,2),1)), (((0,1),2),((0,2),2)), (((0,1),2),((0,2),3))]
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_number
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    batch_size = batch_size // acc_batches
    batch_size_gpu = batch_size // world_size

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    model = DDP(model, device_ids=[rank])
    if torch.__version__ >= "2.0.0":
        GPU_capability = torch.cuda.get_device_capability()
        if GPU_capability[0] >= 7:
            torch.set_float32_matmul_precision('high')
            model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    #from chatGPT: The DistributedSampler shuffles the indices of the entire dataset, not just the portion assigned to a specific GPU. 
    train_dataset = Train_sets(data_path)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_gpu, persistent_workers=True,
                                             num_workers=4, pin_memory=True, sampler=train_sampler)


    steps_per_epoch_train = steps_per_epoch
    total_steps = min(len(train_loader)//acc_batches, steps_per_epoch)
    average_loss_list = []
    #loss_fn = nn.L1Loss()
    #mw = torch.from_numpy(fsc3d).cuda()
    #mwshift = torch.fft.fftshift(mw)
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        with tqdm(total=total_steps, unit="batch", disable=(rank!=0)) as progress_bar:
            model.train()
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            for i, batch in enumerate(train_loader):
                x1, x2, mw = batch
                # debug_matrix(x1, 'x1.mrc')
                # debug_matrix(x2, 'x2.mrc')
                # debug_matrix(mw, 'mw.mrc')
                x1 = x1.cuda()
                x2 = x2.cuda()
                mw = mw.cuda()
                #mwshift = torch.fft.fftshift(mw)
                #optimizer.zero_grad(set_to_none=True)
                if mixed_precision:
                    pass
                else:
                    x1_rot = torch.zeros_like(x1)
                    x2_rot = torch.zeros_like(x2)
                    mask_rot = torch.zeros_like(x1)
                    mask = torch.zeros_like(x1)
                    for j,d in enumerate(x1):
                        rot = random.choice(rotation_list_24)

                        tmp_mask = torch.rot90(mw[j],rot[0][1],rot[0][0])
                        mask_rot[j][0] = torch.rot90(tmp_mask,rot[1][1],rot[1][0])

                        mask[j][0] = mw[j]
                        mwshift = torch.fft.fftshift(mw[j])

                        tmp_x1 = torch.rot90(d[0],rot[0][1],rot[0][0])
                        x1_rot[j][0] = torch.rot90(tmp_x1,rot[1][1],rot[1][0])
                        x1_rot[j][0] = torch.real(torch.fft.ifftn(mwshift*torch.fft.fftn(x1_rot[j][0])))

                        tmp_x2 = torch.rot90(x2[j][0],rot[0][1],rot[0][0])
                        x2_rot[j][0] = torch.rot90(tmp_x2,rot[1][1],rot[1][0])

                    
                    preds = model(x1_rot)
                    loss = masked_loss(preds, x2_rot, mask_rot, mask, mw_weight=2)
                    #loss = loss_fn(x2,preds1)
                    loss = loss / acc_batches
                    loss.backward()
                loss_item = loss.item()
                              
                if ( (i+1)%acc_batches == 0 ) or (i+1) == min(len(train_loader), steps_per_epoch_train * acc_batches):
                    if mixed_precision:
                        pass
                    else:
                        optimizer.step()

                if rank == 0 and ( (i+1)%acc_batches == 0 ):
                   progress_bar.set_postfix({"Loss": loss_item})#, "t1": time2-time1, "t2": time3-time2, "t3": time4-time3})
                   progress_bar.update()
                average_loss += loss_item

                
                if i + 1 >= steps_per_epoch_train*acc_batches:
                    break
            average_loss = average_loss / (i+1.)

                                      
        dist.barrier()
        dist.reduce(average_loss, dst=0)

        average_loss =  average_loss / dist.get_world_size()
        if rank == 0:
            average_loss_list.append(average_loss.cpu().numpy())
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {average_loss:.4f}")
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'average_loss': average_loss_list,
                }, model_path)
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

class Net:
    def __init__(self,filter_base=64,unet_depth=4, add_last=False):
        torch.backends.cudnn.benchmark = True
        self.model = Unet(filter_base = filter_base,unet_depth=unet_depth, add_last=add_last)
        self.world_size = torch.cuda.device_count()
        self.port_number = str(find_unused_port())
        logging.info(f"Port number: {self.port_number}")
        self.metrics = {"average_loss":[],
                        "avg_val_loss":[] }

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metrics["average_loss"] = checkpoint['average_loss']

    def load_jit(self, path):
        #Using the TorchScript format, you will be able to load the exported model and run inference without defining the model class.
        self.model = torch.jit.load(path)
    
    def save(self, path):
        state = self.model.state_dict()
        torch.save(state, path)

    def save_jit(self, path):
        model_scripted = torch.jit.script(self.model) # Export to TorchScript
        model_scripted.save(path) # Save

    def train(self, data_path, output_dir,alpha, beta=1, batch_size=None, output_base='tmp',
              epochs = 10, steps_per_epoch=200, acc_batches =2,
              mixed_precision=False, learning_rate=3e-4, fsc3d = None):
        print('learning rate',learning_rate)

        self.model.zero_grad()

        model_path = f"{output_dir}/{output_base}.pt"
        #if os.path.exists(model_path):
        #    os.remove(model_path)
        print(data_path)
        try: 
            mp.spawn(ddp_simple_n2n, args=(self.world_size, self.port_number, self.model,alpha,beta,
                                       data_path, batch_size, acc_batches, epochs, steps_per_epoch, learning_rate, 
                                       mixed_precision, model_path, fsc3d), nprocs=self.world_size)

        # try: 
        #     print("simple n2n")
        #     mp.spawn(ddp_simple_n2n, args=(self.world_size, self.port_number, self.model,alpha,beta,
        #                                data_path, batch_size, acc_batches, epochs, steps_per_epoch, learning_rate, 
        #                                mixed_precision, model_path, fsc3d), nprocs=self.world_size)
        except KeyboardInterrupt:
           logging.info('KeyboardInterrupt: Terminating all processes...')
           dist.destroy_process_group() 
           os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')")

        checkpoint = torch.load(model_path)
        self.metrics['average_loss'].extend(checkpoint['average_loss'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.save({
            'model_state_dict': checkpoint['model_state_dict'],
            'average_loss': self.metrics['average_loss'],
            }, model_path)
        


    def predict(self, mrc_list, result_dir, iter_count, inverted=True, mw3d=None):    

        bench_dataset = Predict_sets(mrc_list, inverted=inverted)
        bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=4, num_workers=1)
        model = copy.deepcopy(self.model)
        model = torch.nn.DataParallel(model.cuda())
        model.eval()

        predicted = []
        with torch.no_grad():
            for _, val_data in enumerate(bench_loader):
                    res = model(val_data.cuda()) 
                    miu = res.cpu().detach().numpy().astype(np.float32)
                    for item in miu:
                        it = item.squeeze(0)
                        predicted.append(it)
        for i,mrc in enumerate(mrc_list):
            root_name = mrc.split('/')[-1].split('.')[0]

            #outData = normalize(predicted[i], percentile = normalize_percentile)
            file_name = '{}/{}_iter{:0>2d}.mrc'.format(result_dir, root_name, iter_count-1)
 
            with mrcfile.new(file_name, overwrite=True) as output_mrc:
                output_mrc.set_data(-predicted[i])
    
    def predict_tomo(self, args, one_tomo, output_file=None):
    #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc

        root_name = one_tomo.split('/')[-1].split('.')[0]

        if output_file is None:
            if os.path.isdir(args.output_file):
                output_file = args.output_file+'/'+root_name+'_corrected.mrc'
            else:
                output_file = root_name+'_corrected.mrc'

        logging.info('predicting:{}'.format(root_name))

        with mrcfile.open(one_tomo) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
            voxelsize = mrcData.voxel_size

        real_data = normalize(real_data,percentile=args.normalize_percentile)
        data=np.expand_dims(real_data,axis=-1)
        reform_ins = reform3D(data)
        data = reform_ins.pad_and_crop_new(args.cube_size,args.crop_size)

        N = args.batch_size
        num_patches = data.shape[0]
        if num_patches%N == 0:
            append_number = 0
        else:
            append_number = N - num_patches%N
        data = np.append(data, data[0:append_number], axis = 0)
        num_big_batch = data.shape[0]//N
        outData = np.zeros(data.shape)

        logging.info("total batches: {}".format(num_big_batch))


        model = torch.nn.DataParallel(self.model.cuda())
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(num_big_batch), file=sys.stdout):#track(range(num_big_batch), description="Processing..."):
                in_data = torch.from_numpy(np.transpose(data[i*N:(i+1)*N],(0,4,1,2,3)))
                output = model(in_data)
                outData[i*N:(i+1)*N] = np.transpose(output.cpu().detach().numpy().astype(np.float32), (0,2,3,4,1) )

        outData = outData[0:num_patches]

        outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), args.cube_size, args.crop_size)

        outData = normalize(outData,percentile=args.normalize_percentile)
        with mrcfile.new(output_file, overwrite=True) as output_mrc:
            output_mrc.set_data(-outData)
            output_mrc.voxel_size = voxelsize

        logging.info('Done predicting')
    
    def predict_map(self, data, output_dir, cube_size = 64, crop_size=96, output_base=None):
     
        reform_ins = reform3D(data,cube_size,crop_size,7)
        data = reform_ins.pad_and_crop()
        # for i,item in enumerate(data):
        #     with mrcfile.new(f"resd_{i}.mrc", overwrite=True) as mrc:
        #         mrc.set_data(item)

        data = data[:,np.newaxis,:,:]
        data = torch.from_numpy(data)
        print('data_shape',data.shape)

        if output_base is None:
            tmp_data_path = f"{output_dir}/tmp.npy"
        else:
            tmp_data_path = f"{output_dir}/{output_base}.npy"
        mp.spawn(ddp_predict, args=(self.world_size, self.port_number, self.model, data, tmp_data_path), nprocs=self.world_size)

        outData = np.load(tmp_data_path)
        outData = outData.squeeze()

        outData=reform_ins.restore(outData)
        os.remove(tmp_data_path)
        return outData