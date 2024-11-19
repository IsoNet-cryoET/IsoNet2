# Copyright (C) 2024  Hong Cao, Jiahua He, Tao Li, Sheng-You Huang and Huazhong University of Science
# Copyright (C) 2022  Kai Zhang (cskaizhang@gmail.com, https://cszn.github.io/).

# All rights reserved.

# 3D version of Swin-Conv-UNet (https://arxiv.org/pdf/2203.13278.pdf)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import torch.nn as nn
from einops import rearrange 
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_, DropPath

from .frn import FilterResponseNorm3d

class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True) 

        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, 2*window_size-1, self.n_heads).transpose(2,3).transpose(1,2).transpose(0,1))
        # Modified by Jiahua He 
        cord = torch.tensor(np.array([[i, j, k] for i in range(self.window_size) for j in range(self.window_size) for k in range(self.window_size)]))
        self.relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1

    def generate_mask(self, h, w, d, p, shift):
        attn_mask = torch.zeros(h, w, d, p, p, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :,  :, :s, :,  :,  s:, :,  :] = True
        attn_mask[-1, :,  :, s:, :,  :,  :s, :,  :] = True
        attn_mask[:, -1,  :, :, :s,  :,  :, s:,  :] = True
        attn_mask[:, -1,  :, :, s:,  :,  :, :s,  :] = True
        attn_mask[:,  :, -1, :,  :, :s,  :,  :, s:] = True
        attn_mask[:,  :, -1, :,  :, s:,  :,  :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 w3 p1 p2 p3 p4 p5 p6 -> 1 1 (w1 w2 w3) (p1 p2 p3) (p4 p5 p6)')

        return attn_mask

    def forward(self, x):
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2), -(self.window_size//2)), dims=(1,2,3))
        x = rearrange(x, 'b (w1 p1) (w2 p2) (w3 p3) c -> b w1 w2 w3 p1 p2 p3 c', p1=self.window_size, p2=self.window_size, p3=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        d_windows = x.size(3)
        assert h_windows == w_windows == d_windows

        x = rearrange(x, 'b w1 w2 w3 p1 p2 p3 c -> b (w1 w2 w3) (p1 p2 p3) c', p1=self.window_size, p2=self.window_size, p3=self.window_size)

        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # print("sim",sim.shape)
        # print(self.relation.shape)
        # print(self.relative_position_params[:, self.relation[:,:,0].long(), self.relation[:,:,1].long(), self.relation[:,:,2].long()].shape)
        # print(self.relative_position_params.shape)
        # print(self.relation.shape)
        # print(self.relative_position_params[:, self.relation[:,:,0].long(), self.relation[:,:,1].long(), self.relation[:,:,2].long()].shape)
        sim = sim + rearrange(self.relative_position_params[:, self.relation[:,:,0].long(), self.relation[:,:,1].long(), self.relation[:,:,2].long()], 'h p q -> h 1 1 p q')

        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, d_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2 w3) (p1 p2 p3) c -> b (w1 p1) (w2 p2) (w3 p3) c', w1=h_windows, w2=w_windows, w3=d_windows, p1=self.window_size, p2=self.window_size, p3=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2, self.window_size//2), dims=(1,2,3))
        return output

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        #print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv3d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv3d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
                nn.Conv3d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
                FilterResponseNorm3d(self.conv_dim),
                nn.Conv3d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
                FilterResponseNorm3d(self.conv_dim),
                # nn.Conv3d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
                # FilterResponseNorm3d(self.conv_dim),
                )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w d -> b h w d c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w d c -> b c h w d')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x


class SCUNet(nn.Module):

    def __init__(self, in_nc=1, config=[2,2,2,2,2,2,2], dim=32, drop_path_rate=0.2, input_resolution=48, head_dim=16, window_size=3, n_classes=1):
        super(SCUNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv3d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[0])] + \
                      [nn.Conv3d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[1])] + \
                      [nn.Conv3d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[2])] + \
                      [nn.Conv3d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                    for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose3d(8*dim, 4*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[4])]
                      
        begin += config[4]
        self.m_up2 = [nn.ConvTranspose3d(4*dim, 2*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[5])]
                      
        begin += config[5]
        self.m_up1 = [nn.ConvTranspose3d(2*dim, dim, 2, 2, 0, bias=False),] + \
                    [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[6])]

        self.m_tail = [nn.Conv3d(dim, n_classes, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)  

    def forward(self, x0):

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        return x+x0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SCUNet_depth4(nn.Module):

    def __init__(self, in_nc=1, config=[2, 2, 2, 2, 2, 2, 2, 2], dim=32, drop_path_rate=0.2, input_resolution=48, head_dim=16, window_size=3, n_classes=1):
        super(SCUNet_depth4, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv3d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                        for i in range(config[0])] + \
                       [nn.Conv3d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                        for i in range(config[1])] + \
                       [nn.Conv3d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//4)
                        for i in range(config[2])] + \
                       [nn.Conv3d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_down4 = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                        for i in range(config[3])] + \
                       [nn.Conv3d(8*dim, 16*dim, 2, 2, 0, bias=False)]

        begin += config[3]
        self.m_body = [ConvTransBlock(8*dim, 8*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//16)
                       for i in range(config[4])]

        begin += config[4]
        self.m_up4 = [nn.ConvTranspose3d(16*dim, 8*dim, 2, 2, 0, bias=False),] + \
                     [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                      for i in range(config[5])]

        begin += config[5]
        self.m_up3 = [nn.ConvTranspose3d(8*dim, 4*dim, 2, 2, 0, bias=False),] + \
                     [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//4)
                      for i in range(config[6])]

        begin += config[6]
        self.m_up2 = [nn.ConvTranspose3d(4*dim, 2*dim, 2, 2, 0, bias=False),] + \
                     [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[7])]

        begin += config[7]
        self.m_up1 = [nn.ConvTranspose3d(2*dim, dim, 2, 2, 0, bias=False),] + \
                    [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[8])]

        self.m_tail = [nn.Conv3d(dim, n_classes, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_down4 = nn.Sequential(*self.m_down4)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up4 = nn.Sequential(*self.m_up4)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)

    def forward(self, x0):

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x5 = self.m_down4(x4)
        x = self.m_body(x5)
        x = self.m_up4(x + x5)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x+x0
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SCUNet_depth4_from2nd(nn.Module):

    def __init__(self, in_nc=1, config=[0, 2, 2, 2, 2, 2, 2, 0], dim=32, drop_path_rate=0.2, input_resolution=48, head_dim=16, window_size=3, n_classes=1):
        super(SCUNet_depth4_from2nd, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv3d(in_nc, dim, 3, 1, 1, bias=False)]
        begin = 0
        self.m_down1 = nn.Sequential(
                FilterResponseNorm3d(dim),
                nn.Conv3d(dim, dim, 3, 1, 1, bias=False),
                FilterResponseNorm3d(dim),
                nn.Conv3d(dim, dim, 3, 1, 1, bias=False),
                FilterResponseNorm3d(dim),
                nn.Conv3d(dim, dim, 3, 1, 1, bias=False),
                FilterResponseNorm3d(dim),
                nn.Conv3d(dim, 2*dim, 2, 2, 0, bias=False)
                )
        
        # self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
        #                 for i in range(config[0])] + \
        #                [nn.Conv3d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                        for i in range(config[1])] + \
                       [nn.Conv3d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//4)
                        for i in range(config[2])] + \
                       [nn.Conv3d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_down4 = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                        for i in range(config[3])] + \
                       [nn.Conv3d(8*dim, 16*dim, 2, 2, 0, bias=False)]

        begin += config[3]
        self.m_body = [ConvTransBlock(8*dim, 8*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//16)
                       for i in range(config[4])]

        begin += config[4]
        self.m_up4 = [nn.ConvTranspose3d(16*dim, 8*dim, 2, 2, 0, bias=False),] + \
                     [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                      for i in range(config[5])]

        begin += config[5]
        self.m_up3 = [nn.ConvTranspose3d(8*dim, 4*dim, 2, 2, 0, bias=False),] + \
                     [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//4)
                      for i in range(config[6])]

        begin += config[6]
        self.m_up2 = [nn.ConvTranspose3d(4*dim, 2*dim, 2, 2, 0, bias=False),] + \
                     [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[7])]

        begin += config[7]
        # self.m_up1 = [nn.ConvTranspose3d(2*dim, dim, 2, 2, 0, bias=False),] + \
        #             [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
        #               for i in range(config[8])]

        self.m_up1 = nn.Sequential(
                nn.ConvTranspose3d(2*dim, dim, 2, 2, 0, bias=False),
                nn.Conv3d(dim, dim, 3, 1, 1, bias=False),
                FilterResponseNorm3d(dim),
                nn.Conv3d(dim, dim, 3, 1, 1, bias=False),
                FilterResponseNorm3d(dim),
                nn.Conv3d(dim, dim, 3, 1, 1, bias=False),
                FilterResponseNorm3d(dim),
                )

        self.m_tail = [nn.Conv3d(dim, n_classes, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_down4 = nn.Sequential(*self.m_down4)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up4 = nn.Sequential(*self.m_up4)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)

    def forward(self, x0):

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x5 = self.m_down4(x4)
        x = self.m_body(x5)
        x = self.m_up4(x + x5)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x+x0
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)