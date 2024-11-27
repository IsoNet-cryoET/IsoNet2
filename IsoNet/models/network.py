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
    def __init__(self, method=None, arch = 'unet-default', cube_size = 96, pretrained_model=None, state="train"):
        self.state = state
        if pretrained_model != None and pretrained_model != "None":
            self.load(pretrained_model)
        else:
            self.initialize(method, arch,cube_size)
            self.metrics = {"average_loss":[],
                       "inside_mw_loss":[],
                       "outside_mw_loss":[]}
        torch.backends.cudnn.benchmark = True
        

    
    def initialize(self, method='regular', arch = 'unet-medium', cube_size = 96):
        
        self.arch = arch
        self.method = method
        self.cube_size = cube_size

        if self.arch == 'unet-large':
            from .unet import Unet
            self.model = Unet(filter_base = 64,unet_depth=4, add_last=True)
        elif self.arch == 'unet-medium':
            from .unet import Unet
            self.model = Unet(filter_base = 32,unet_depth=4, add_last=True)
        elif self.arch == 'unet-small':
            from .unet import Unet
            self.model = Unet(filter_base = 16,unet_depth=4, add_last=True)

        elif self.arch in ['scunet-large','scunet-medium','scunet-small','scunet-fast','scunet-fast-large']:
            if self.state == "train":
                drop_rate=0.1
            else:
                drop_rate=0

            if cube_size%3 == 0:
                window_size = 3
            elif cube_size%4 == 0:
                window_size = 4
            
            if self.arch == 'scunet-medium':
                dim=64
                #config=[1,1,1,1,1,1,1,1,1]
                config=[2,2,2,2,2,2,2,2,2]
            elif self.arch == 'scunet-small':
                dim=32
                #config=[1,1,1,1,1,1,1,1,1]
                config=[2,2,2,2,2,2,2,2,2]
            elif self.arch == 'scunet-fast':
                dim=32
                config=[0,2,2,2,2,2,2,2,0]
            elif self.arch == 'scunet-fast-large':
                dim=64
                config=[0,2,2,2,2,2,2,2,0]
            

            from IsoNet.models.scunet import SCUNet_depth4,  SCUNet, SCUNet_depth4_from2nd
            if self.arch in ['scunet-fast', 'scunet-fast-large']:
                self.model = SCUNet_depth4_from2nd(
                            in_nc=1,
                            config=config,
                            dim=dim,
                            drop_path_rate=drop_rate,
                            input_resolution=cube_size,
                            head_dim=16,
                            window_size=window_size,
                        )
            else:
                self.model = SCUNet_depth4(
                            in_nc=1,
                            config=config,
                            dim=dim,
                            drop_path_rate=drop_rate,
                            input_resolution=cube_size,
                            head_dim=16,
                            window_size=window_size,
                        )            
                self.model.apply(self.model._init_weights)

        else:
            print(f"method {method} should be either unet-default, unet-small,unet-medium,HSFormer" )
        # elif self.arch == 'HSFormer':
        #     from IsoNet.models.HSFormer import swin_tiny_patch4_window8
        #     self.model = swin_tiny_patch4_window8(img_size=cube_size, embed_dim=128,num_classes =1)
        # elif self.arch == 'HSFormer-small':
        #     from IsoNet.models.HSFormer import swin_tiny_patch4_window8
        #     self.model = swin_tiny_patch4_window8(img_size=cube_size, embed_dim=64, num_classes =1)
        # elif self.arch == 'vtunet':
        #     from IsoNet.models.vtunet import VTUnet
        #     self.model = VTUnet()
        num_params = get_num_parameters(self.model)
        print(f'Total number of parameters: {num_params}')


        self.world_size = torch.cuda.device_count()
        self.port_number = str(find_unused_port())
        logging.info(f"Port number: {self.port_number}")


    def load(self, path):
        
        checkpoint = torch.load(path)
        self.method = checkpoint['method']
        self.arch = checkpoint['arch']
        self.cube_size = checkpoint['cube_size']
        print(self.arch)
        self.initialize(self.method, self.arch, self.cube_size)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metrics = checkpoint['metrics']

    def save(self, path):
        pass
        # state = self.model.state_dict()
        # torch.save({
        #     'arch':self.arch,
        #     'method':self.method,
        #     'model_state_dict': state,
        #     'metrics': self.metrics['metrics'],
        #     }, path)
        
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
        training_params['metrics'] = self.metrics
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
        self.load(f"{training_params['output_dir']}/network_{training_params['arch']}_{training_params['cube_size']}.pt")
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
        # This is legacy
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
        data = data[:,np.newaxis,:,:].astype(np.float32)
        data = torch.from_numpy(data)
        print('data_shape',data.shape)
        mp.spawn(ddp_predict, args=(self.world_size, self.port_number, self.model, data, tmp_data_path, wedge), nprocs=self.world_size)
        outData = np.load(tmp_data_path)
        outData = outData.squeeze()
        return outData

    
    def predict_map(self, data, output_dir, cube_size = 64, crop_size=96, wedge=None):
        # change edge width from 7 to 5 to reduce computing
        reform_ins = reform3D(data,cube_size,crop_size,5)
        data = reform_ins.pad_and_crop()        
        tmp_data_path = f"{output_dir}/tmp.npy"
        outData = self.predict(data, tmp_data_path=tmp_data_path, wedge=wedge)
        outData = outData.squeeze()
        outData=reform_ins.restore(outData)
        os.remove(tmp_data_path)
        return outData
    

class DuoNet:
    def __init__(self, method=None, arch = 'unet-default', cube_size = 96, pretrained_model1=None, pretrained_model2=None, state="train"):
        self.net1 = Net(method=method, arch=arch, cube_size = cube_size, pretrained_model=pretrained_model1, state=state)
        self.net2 = Net(method=method, arch=arch, cube_size = cube_size, pretrained_model=pretrained_model2, state=state)

    def load(self, pretrained_model1, pretrained_model2):
        self.net1.load(pretrained_model1)
        self.net2.load(pretrained_model2)
    
    def train(self, training_params):
        assert training_params["epochs"] % training_params["T_max"] == 0

        epochs = training_params["epochs"]
        T_max = 10
        T_steps = training_params["epochs"] // training_params["T_max"] 

        training_params1 = training_params.copy()
        training_params1['split'] = "top"
        training_params1['epochs'] = T_steps

        training_params2 = training_params.copy()
        training_params2['split'] = "bottom"
        training_params2['epochs'] = T_steps

        for i in range(T_steps):
            print(f"training the top half of tomograms for {T_max} epochs, remaining epochs {epochs-T_max*i}")
            self.net1.train(training_params1)
            print(f"training the bottom half of tomograms for {T_max} epochs, remaining epochs {epochs-T_max*i}")
            self.net2.train(training_params2)


    def predict_map(self, data, output_dir, cube_size = 64, crop_size=96, wedge=None):
        predicted_map1 = self.net1.predict(data=data, output_dir=output_dir, cube_size = cube_size, crop_size=crop_size, wedge=wedge)
        predicted_map2 = self.net2.predict(data=data, output_dir=output_dir, cube_size = cube_size, crop_size=crop_size, wedge=wedge)
        return predicted_map1, predicted_map2