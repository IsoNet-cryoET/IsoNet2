import numpy as np
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
                       "inside_loss":[],
                       "outside_loss":[]}
        if state == "train":
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
        checkpoint = torch.load(path, weights_only=False)
        self.method = checkpoint['method']
        self.arch = checkpoint['arch']
        self.cube_size = checkpoint['cube_size']
        self.CTF_mode = checkpoint['CTF_mode']
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


    def train(self, training_params):

        self.model.zero_grad()
        training_params['metrics'] = self.metrics

        #### preparing data
        # from chatGPT: The DistributedSampler shuffles the indices of the entire dataset, not just the portion assigned to a specific GPU. 
        clip_first_peak = False
        if training_params['CTF_mode'] == 'network':
            clip_first_peak = True

        if training_params['method'] == 'regular':
            from IsoNet.models.data_sequence import Train_sets_regular
            train_dataset = Train_sets_regular(training_params['data_path'])

        elif training_params['method'] in ['n2n', 'isonet2', 'isonet2-n2n']:
            if training_params["noise_level"] > 0:
                noise_dir = f'{training_params["output_dir"]}/noise_volumes'
            else:
                noise_dir = None

            from IsoNet.models.data_sequence import Train_sets_n2n, MRCDataset
            train_dataset = Train_sets_n2n(training_params['star_file'],method=training_params['method'], 
                                        cube_size=training_params['cube_size'], input_column=training_params['input_column'],\
                                        split=training_params['split'], noise_dir=noise_dir, clip_first_peak=clip_first_peak,\
                                        start_bt_size=training_params["start_bt_size"])
            # train_dataset = MRCDataset('sphere','GT')
        try:
            if self.world_size > 1:
                mp.spawn(ddp_train, args=(self.world_size, self.port_number, self.model, train_dataset, training_params), nprocs=self.world_size)
            else:
                ddp_train(0, self.world_size, self.port_number, self.model, train_dataset, training_params)

        except KeyboardInterrupt:
           logging.info('KeyboardInterrupt: Terminating all processes...')
           dist.destroy_process_group() 
           os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')")
        self.load(f"{training_params['output_dir']}/network_{training_params['method']}_{training_params['arch']}_{training_params['cube_size']}_{training_params['split']}.pt")

        
    def predict_subtomos(self, settings):
        # This is legacy
        from IsoNet.utils.fileio import read_mrc,write_mrc
        first_map, pixel_size = read_mrc(settings['mrc_list'][0])
        shape = first_map.shape
        data = np.zeros((len(settings['mrc_list']),shape[0], shape[1], shape[2]), dtype=np.float32)
        for i, file_name in enumerate(settings['mrc_list']):
            subtomo, _ = read_mrc(file_name)
            data[i] = -subtomo
        tmp_data_path = f"{settings['output_dir']}/tmp.npy"
        outData = self.predict(data, tmp_data_path=tmp_data_path)
        os.remove(tmp_data_path)

        for i, file_name in enumerate(settings['mrc_list']):
            root_name = os.path.splitext(os.path.basename(file_name))[0]

            write_mrc('{}/{}_iter{:0>2d}.mrc'.format(settings['output_dir'],
                                                     root_name,
                                                     settings['iter_count']-1), 
                                                     -outData[i])
        return outData

    def predict(self, data, tmp_data_path, F_mask=None):    
        data = data[:,np.newaxis,:,:].astype(np.float32)
        data = torch.from_numpy(data)
        print('data_shape',data.shape)
        mp.spawn(ddp_predict, args=(self.world_size, self.port_number, self.model, data, tmp_data_path,\
                                     F_mask), nprocs=self.world_size)
        all_outputs = []
        for r in range(self.world_size):
            rank_output_path = f"{tmp_data_path}_rank_{r}.npy"
            rank_output = np.load(rank_output_path,mmap_mode='r')
            all_outputs.append(rank_output)
        outData = np.concatenate(all_outputs, axis=0)[:data.shape[0]]

        for r in range(self.world_size):
            os.remove(f"{tmp_data_path}_rank_{r}.npy")

        outData = outData.squeeze()
        return outData

    
    def predict_map(self, data, output_dir, cube_size = 64, crop_size=96, F_mask=None):
        # change edge width from 7 to 5 to reduce computing
        reform_ins = reform3D(data,cube_size,crop_size,5)
        data = reform_ins.pad_and_crop()        
        tmp_data_path = f"{output_dir}/tmp.npy"
        outData = self.predict(data, tmp_data_path=tmp_data_path, F_mask=F_mask)
        outData = outData.squeeze()
        outData=reform_ins.restore(outData)
        return outData
    

class DuoNet:
    def __init__(self, method=None, arch = 'unet-default', cube_size = 96, pretrained_model1=None, pretrained_model2=None, state="train"):
        self.net1 = Net(method=method, arch=arch, cube_size = cube_size, pretrained_model=pretrained_model1, state=state)
        self.net2 = Net(method=method, arch=arch, cube_size = cube_size, pretrained_model=pretrained_model2, state=state)
        self.method = method
        self.arch = arch
        self.cube_size = cube_size
        self.state = state
        if pretrained_model1 not in ["None", None] and pretrained_model1 not in ["None", None]:
            self.method = self.net1.method
            self.arch = self.net1.arch
            self.cube_size = self.net1.cube_size
            self.state = self.net1.state

    def load(self, pretrained_model1, pretrained_model2):
        self.net1.load(pretrained_model1)
        self.net2.load(pretrained_model2)
        self.method = self.net1.method
        self.arch = self.net1.arch
        self.cube_size = self.net1.cube_size
        self.state = self.net1.state
    
    def train(self, training_params):
        assert training_params["epochs"] % training_params["T_max"] == 0

        epochs = training_params["epochs"]
        T_max = training_params["T_max"]
        T_steps = training_params["epochs"] // training_params["T_max"] 

        training_params1 = training_params.copy()
        training_params1['split'] = "top"
        training_params1['epochs'] = T_max

        training_params2 = training_params.copy()
        training_params2['split'] = "bottom"
        training_params2['epochs'] = T_max

        for i in range(T_steps):
            print(f"training the top half of tomograms for {T_max} epochs, remaining epochs {epochs-T_max*i}")
            self.net1.train(training_params1)
            training_params1["starting_epoch"]+=T_max
            print(f"training the bottom half of tomograms for {T_max} epochs, remaining epochs {epochs-T_max*i}")
            self.net2.train(training_params2)
            training_params2["starting_epoch"]+=T_max


    def predict_map(self, data, output_dir, cube_size = 64, crop_size=96, F_mask=None):
        predicted_map1 = self.net1.predict_map(data=data, output_dir=output_dir, cube_size = cube_size, crop_size=crop_size, F_mask=F_mask)
        predicted_map2 = self.net2.predict_map(data=data, output_dir=output_dir, cube_size = cube_size, crop_size=crop_size, F_mask=F_mask)
        return [predicted_map1, predicted_map2]