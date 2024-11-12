import numpy as np

from .unet import Unet
import torch
import os
import logging
from IsoNet.utils.toTile import reform3D
import socket
import torch.multiprocessing as mp
import torch.distributed as dist
from IsoNet.models.train import ddp_train, ddp_predict

def find_unused_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    _, port = sock.getsockname()
    sock.close()
    return port
import torch

# Assuming 'model' is your PyTorch model
def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())



class Net:
    def __init__(self, method=None, arch = 'unet-default', cube_size = 96, pretrained_model=None):

        if pretrained_model != None and pretrained_model != "None":
            self.load(pretrained_model)
        else:
            self.initialize(method, arch,cube_size)
        torch.backends.cudnn.benchmark = True

    
    def initialize(self, method='regular', arch = 'unet-default', cube_size = 96):
        
        self.arch = arch
        self.method = method
        self.cube_size = cube_size

        if self.arch == 'unet-default':
            from .unet import Unet
            self.model = Unet(filter_base = 64,unet_depth=4, add_last=True)
        elif self.arch == 'unet-small':
            from .unet import Unet
            self.model = Unet(filter_base = 16,unet_depth=4, add_last=True)
        elif self.arch == 'unet-medium':
            from .unet import Unet
            self.model = Unet(filter_base = 32,unet_depth=4, add_last=True)
        elif self.arch == 'HSFormer':
            from IsoNet.models.HSFormer import swin_tiny_patch4_window8
            self.model = swin_tiny_patch4_window8(img_size=cube_size, embed_dim=128,num_classes =1)
        elif self.arch == 'HSFormer-small':
            from IsoNet.models.HSFormer import swin_tiny_patch4_window8
            self.model = swin_tiny_patch4_window8(img_size=cube_size, embed_dim=64, num_classes =1)
        elif self.arch == 'vtunet':
            from IsoNet.models.vtunet import VTUnet
            self.model = VTUnet()
        else:
            print(f"method {method} should be either unet-default, unet-small,unet-medium,HSFormer" )

        num_params = get_num_parameters(self.model)
        print(f'Total number of parameters: {num_params}')


        self.world_size = torch.cuda.device_count()
        self.port_number = str(find_unused_port())
        logging.info(f"Port number: {self.port_number}")

        self.metrics = {"average_loss":[],
                        "avg_val_loss":[] }

    def load(self, path):
        checkpoint = torch.load(path)
        self.method = checkpoint['method']
        self.arch = checkpoint['arch']
        self.cube_size = checkpoint['cube_size']

        self.initialize(self.method, self.arch, self.cube_size)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metrics = checkpoint['metrics']

    def save(self, path):
        state = self.model.state_dict()
        torch.save({
            'arch':self.arch,
            'method':self.method,
            'model_state_dict': state,
            'metrics': self.metrics['metrics'],
            }, path)
        
    def load_jit(self, path):
        # TODO
        #Using the TorchScript format, you will be able to load the exported model and run inference without defining the model class.
        self.model = torch.jit.load(path)

    def save_jit(self, path):
        # TODO
        model_scripted = torch.jit.script(self.model) # Export to TorchScript
        model_scripted.save(path) # Save

    #def train(self, data_path, output_dir, batch_size=None, outmodel_path='tmp.pt',
    #          epochs = 10, steps_per_epoch=200, acc_batches =2,
    #          mixed_precision=False, learning_rate=3e-4):
    def train(self, training_params):

        self.model.zero_grad()

        #if os.path.exists(model_path):
        #    os.remove(model_path)

        try: 
            # mp.spawn(self.ddp_train, args=(self.world_size, self.port_number, self.model,
            #                            data_path, batch_size, acc_batches, epochs, steps_per_epoch, learning_rate, 
            #                            mixed_precision, outmodel_path), nprocs=self.world_size)
            # mp.spawn(ddp_train, args=(self.world_size, self.port_number, self.model,
            #                            training_params), nprocs=self.world_size)
            if self.world_size > 1:
                # For multiple GPUs, use DistributedDataParallel and spawn multiple processes
                mp.spawn(ddp_train, args=(self.world_size, self.port_number, self.model, training_params), nprocs=self.world_size)
            else:
                # For single GPU, directly call ddp_train without using DDP
                ddp_train(0, self.world_size, self.port_number, self.model, training_params)

        except KeyboardInterrupt:
           logging.info('KeyboardInterrupt: Terminating all processes...')
           dist.destroy_process_group() 
           os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')")
        self.load(f"{training_params['output_dir']}/network_{training_params['arch']}_{training_params['method']}.pt")
        # checkpoint = torch.load(training_params['outmodel_path'])
        # self.metrics['average_loss'].extend(checkpoint['average_loss'])
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # torch.save({
        #     'arch':self.arch,
        #     'method':self.method,
        #     'model_state_dict': checkpoint['model_state_dict'],
        #     'average_loss': self.metrics['average_loss'],
        #     }, training_params['outmodel_path'])
        
    def predict_subtomos(self, settings):
        from IsoNet.utils.fileio import read_mrc,write_mrc
        first_map, pixel_size = read_mrc(settings.mrc_list[0])
        shape = first_map.shape
        data = np.zeros((len(settings.mrc_list),shape[0], shape[1], shape[2]), dtype=np.float32)
        for i, file_name in enumerate(settings.mrc_list):
            subtomo, _ = read_mrc(file_name)
            data[i] = subtomo
        tmp_data_path = f"{settings.output_dir}/tmp.npy"
        outData = self.predict(data, tmp_data_path=tmp_data_path)
        os.remove(tmp_data_path)

        for i, file_name in enumerate(settings.mrc_list):
            root_name = os.path.splitext(os.path.basename(file_name))[0]

            write_mrc('{}/{}_iter{:0>2d}.mrc'.format(settings.output_dir,
                                                     root_name,
                                                     settings.iter_count-1), 
                                                     -outData[i])
        return outData

    def predict(self, data, tmp_data_path, wedge=None):    
        data = data[:,np.newaxis,:,:]
        data = torch.from_numpy(data)
        print('data_shape',data.shape)
        mp.spawn(ddp_predict, args=(self.world_size, self.port_number, self.model, data, tmp_data_path, wedge), nprocs=self.world_size)
        outData = np.load(tmp_data_path)
        outData = outData.squeeze()
        return outData

    
    def predict_map(self, data, output_dir, cube_size = 64, crop_size=96, output_base=None, wedge=None):
        # change edge width from 7 to 5 to reduce computing
        reform_ins = reform3D(data,cube_size,crop_size,5)
        data = reform_ins.pad_and_crop()
        
        tmp_data_path = f"{output_dir}/tmp.npy"
        outData = self.predict(data, tmp_data_path=tmp_data_path, wedge=wedge)
        outData = outData.squeeze()
        outData=reform_ins.restore(outData)
        os.remove(tmp_data_path)
        return outData