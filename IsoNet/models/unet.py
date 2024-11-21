from typing import List
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    # conv_per_depth fixed to 2
    def __init__(self, in_channels, out_channels, n_conv, kernel_size =3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, bias=False), 
            #nn.InstanceNorm3d(num_features = out_channels),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(),
        ]
        for _ in range(max(n_conv-1,0)):
            layers.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            #layers.append(nn.InstanceNorm3d(num_features=out_channels))
            layers.append(nn.BatchNorm3d(num_features=out_channels))
            layers.append(nn.LeakyReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, filter_base, unet_depth, n_conv):
        super(EncoderBlock, self).__init__()
        self.module_dict = nn.ModuleDict()
        self.module_dict['first_conv'] = nn.Conv3d(in_channels=1, out_channels=filter_base[0], kernel_size=3, stride=1, padding=1)

        for n in range(unet_depth):
            self.module_dict["conv_stack_{}".format(n)] = ConvBlock(in_channels=filter_base[n], out_channels=filter_base[n], n_conv=n_conv)
            self.module_dict["stride_conv_{}".format(n)] = ConvBlock(in_channels=filter_base[n], out_channels=filter_base[n+1], n_conv=1, kernel_size=2, stride=2, padding=0)
        
        self.module_dict["bottleneck"] = ConvBlock(in_channels=filter_base[n+1], out_channels=filter_base[n+1], n_conv=n_conv-1)
    
    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            x = op(x)
            if k.startswith('conv'):
                down_sampling_features.append(x)
        return x, down_sampling_features

class DecoderBlock(nn.Module):
    def __init__(self, filter_base, unet_depth, n_conv):
        super(DecoderBlock, self).__init__()
        self.module_dict = nn.ModuleDict()
        for n in reversed(range(unet_depth)):
            self.module_dict["deconv_{}".format(n)] = nn.ConvTranspose3d(in_channels=filter_base[n+1],
                                                                         out_channels=filter_base[n],
                                                                         kernel_size=2,
                                                                         stride=2,
                                                                         padding=0)
            self.module_dict["activation_{}".format(n)] = nn.LeakyReLU()
            self.module_dict["conv_stack_{}".format(n)] = ConvBlock(filter_base[n]*2, filter_base[n],n_conv=n_conv)
        
    def forward(self, x,
        down_sampling_features: List[torch.Tensor]):
        for k, op in self.module_dict.items():
            x=op(x)
            if k.startswith("deconv"):
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
        return x

class Unet(nn.Module):
    def __init__(self,filter_base = 64,unet_depth=3, add_last=False):
        super(Unet, self).__init__()
        self.add_last = add_last
        if filter_base == 64:
            filter_base = [64,128,256,320,320,320]
        elif filter_base == 32:
            filter_base = [32,64,128,256,320,320]
        elif filter_base == 16:
            filter_base = [16,32,64,128,256,320]
        #filter_base = [1,1,1,1,1]
        # unet_depth = 4
        n_conv = 3
        self.encoder = EncoderBlock(filter_base=filter_base, unet_depth=unet_depth, n_conv=n_conv)
        self.decoder = DecoderBlock(filter_base=filter_base, unet_depth=unet_depth, n_conv=n_conv)
        self.final = nn.Conv3d(in_channels=filter_base[0], out_channels=1, kernel_size=3, stride=1, padding=1)
       
        # if metrics is None:
        #     self.metrics = {'train_loss':[], 'val_loss':[]}
        # else:
        #     self.metrics = metrics
        # self.training_loss_list = []
        # self.validation_loss_list = []
    
    def forward(self, x):
        x_org = x
        x, down_sampling_features = self.encoder(x)
        x = self.decoder(x, down_sampling_features)
        y_hat = self.final(x)
        if self.add_last:
            y_hat += x_org
        return y_hat

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     out = self(x)
    #     if self.variance_out:
    #         #loss = nn.L1Loss()(out[1], torch.abs(out[0]-y))
    #         c = 0.6931471805599453 # log(2)
    #         loss = torch.mean(torch.div(torch.abs(out[0]-y), out[1]) + torch.log(out[1])) + c
    #     else:
    #         loss = nn.L1Loss()(out, y)
    #     #loss_numpy = loss.item()#.detach().cpu().numpy()
    #     #self.training_loss_list.append(loss_numpy)
    #     #self.log("loss", loss, prog_bar=True,on_step=True,sync_dist=True)
    #     return loss #{'loss': loss}
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    #     return optimizer 

    # def validation_step(self, batch, batch_idx):
    #     #with torch.no_grad():
    #     x, y = batch
    #     out = self(x)
    #     if self.variance_out:
    #         #loss = nn.L1Loss()(out[1], torch.abs(out[0]-y))
    #         c = 0.6931471805599453 # log(2)
    #         val_loss = torch.mean(torch.div(torch.abs(out[0]-y), out[1]) + torch.log(out[1])) + c
    #     else:
    #         val_loss = nn.L1Loss()(out, y)
    #     loss_numpy = val_loss.item()#.cpu().numpy()
    #     self.validation_loss_list.append(loss_numpy)
    #     return {'val_loss': val_loss}

    # def predict(self, real_data, batch_size):
    # #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc
    #     from IsoNet.util.toTile import reform3D


    #     data=np.expand_dims(real_data,axis=-1)
    #     reform_ins = reform3D(data)
    #     data = reform_ins.pad_and_crop_new(cube_size,crop_size)

    #     N = batch_size
    #     num_patches = data.shape[0]
    #     if num_patches%N == 0:
    #         append_number = 0
    #     else:
    #         append_number = N - num_patches%N
    #     data = np.append(data, data[0:append_number], axis = 0)
    #     num_big_batch = data.shape[0]//N
    #     outData = np.zeros(data.shape)

    #     logging.info("total batches: {}".format(num_big_batch))

    #     with torch.no_grad():
    #         for i in tqdm(range(num_big_batch), file=sys.stdout):#track(range(num_big_batch), description="Processing..."):
    #             in_data = torch.from_numpy(np.transpose(data[i*N:(i+1)*N],(0,4,1,2,3)))
    #             output = self(in_data)
    #             out_tmp = output.cpu().detach().numpy().astype(np.float32)
    #             out_tmp = np.transpose(out_tmp, (0,2,3,4,1) )
    #             outData[i*N:(i+1)*N] = out_tmp

    #     outData = outData[0:num_patches]

    #     outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), cube_size, crop_size)



    #     logging.info('Done predicting')
    #     return outData

    #def on_train_epoch_end(self):
    #    loss = np.mean(self.training_loss_list).astype(float)#torch.stack(self.training_loss_list).mean().item()
    #    self.training_loss_list = []
        #loss = torch.stack([x['loss'] for x in outputs]).mean().item()
    #    self.metrics["train_loss"].append(loss)
    #    #self.log("train_loss", loss, logger=True,on_epoch=True)

    #def on_validation_epoch_end(self):
        #loss = torch.stack(self.validation_loss_list).mean().item()
    #    loss = np.mean(self.validation_loss_list).astype(float)
    #    self.validation_loss_list = []
    #    self.metrics["val_loss"].append(loss)
    #    self.log("val_loss", loss, prog_bar=True,on_epoch=True,sync_dist=True)

    
        
