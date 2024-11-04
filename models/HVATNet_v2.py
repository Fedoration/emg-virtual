import numpy as np
from pathlib import Path

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


# 1. Input data -> Conv1d 1x1 -> N Res Blocks. -> M downsample blocks with advanced Conv -> M upsample blocks with skip connections + 
# outputs -> Merge Multiscale features


class SimpleResBlock(nn.Module):
    """
    Input is [batch, emb, time]
    Res block.
    In features input and output the same.
    So we can apply this block several times.
    """
    def __init__(self, in_channels, kernel_size):
        super(SimpleResBlock, self).__init__()


        self.conv1 = nn.Conv1d(in_channels, in_channels,
                               kernel_size=kernel_size,
                               bias=True,
                               padding='same')

        self.activation = nn.GELU()

        self.conv2 = nn.Conv1d(in_channels, in_channels,
                               kernel_size=kernel_size,
                               bias=True,
                               padding='same')


    def forward(self, x_input):

        x = self.conv1(x_input)
        x = self.activation(x)
        x = self.conv2(x)

        res = x + x_input

        return res

class AdvancedConvBlock(nn.Module):
    """
    Input is [batch, emb, time]
    block [ conv -> layer norm -> act -> dropout ]

    To do:
        add res blocks.
    """
    def __init__(self, in_channels, kernel_size,dilation=1):
        super(AdvancedConvBlock, self).__init__()

        # use it instead stride.

        self.conv_dilated = nn.Conv1d(in_channels, in_channels,
                                      kernel_size=kernel_size,
                                      dilation = dilation,
                                      bias=True,
                                      padding='same')

        self.conv1_1 = nn.Conv1d(in_channels, in_channels,
                                 kernel_size=kernel_size,
                                 bias=True,
                                 padding='same')

        self.conv1_2 = nn.Conv1d(in_channels, in_channels,
                                 kernel_size=kernel_size,
                                 bias=True,
                                 padding='same')

        self.conv_final = nn.Conv1d(in_channels, in_channels,
                                    kernel_size=1,
                                    bias=True,
                                    padding='same')

    def forward(self, x_input):
        """
        input
            - dilation
            - gated convolution
            - conv final
            - maybe dropout. and LN
        - input + res
        """
        x = self.conv_dilated(x_input)

        flow = torch.tanh(self.conv1_1(x))
        gate = torch.sigmoid(self.conv1_2(x))
        res = flow * gate

        res = self.conv_final(res)

        res = res + x_input
        return res


class AdvancedEncoder(nn.Module):
    def __init__(self, n_blocks_per_layer=3, n_filters=64, kernel_size=3,
                 dilation=1, strides = (2, 2, 2)):
        super(AdvancedEncoder, self).__init__()

        self.n_layers = len(strides)
        self.downsample_blocks = nn.ModuleList([nn.Conv1d(n_filters, n_filters, kernel_size=1, stride=stride) for stride in strides])

        conv_layers = []
        for i in range(self.n_layers):
            blocks = nn.ModuleList([AdvancedConvBlock(n_filters,kernel_size,
                                                      dilation=dilation) for i in range(n_blocks_per_layer)])
            
            layer = nn.Sequential(*blocks)
            conv_layers.append(layer)
            
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        """
        Apply conv + downamsple
        Return uutputs of eahc conv + the last features after downsampling.
        """

        outputs =  []
        for conv_block, down in zip(self.conv_layers, self.downsample_blocks) :
            x_res = conv_block(x)
            x = down(x_res)
            outputs.append(x_res)

        outputs.append(x)

        return outputs



class AdvancedDecoder(nn.Module):
    def __init__(self, n_blocks_per_layer=3, n_filters=64, kernel_size=3,
                 dilation=1, strides = (2, 2, 2)):
        super(AdvancedDecoder, self).__init__()

        self.n_layers = len(strides)


        self.upsample_blocks = nn.ModuleList([nn.Upsample(scale_factor=scale,
                                                          mode='linear',
                                                          align_corners=False) for scale in strides])


        conv_layers = []
        for i in range(self.n_layers):
           
            reduce  = nn.Conv1d(n_filters*2, n_filters, kernel_size=kernel_size, padding='same')
            conv_blocks = nn.ModuleList([AdvancedConvBlock(n_filters, kernel_size, dilation=dilation) for i in range(n_blocks_per_layer)])
            
            conv_blocks.insert(0, reduce)
            layer = nn.Sequential(*conv_blocks)

            conv_layers.append(layer)
        
        self.conv_layers = nn.ModuleList(conv_layers)
        

    def forward(self, skips):
        """
        Apply conv + downamsple
        Return uutputs of each conv + the last features after downsampling.
        """
        skips = skips[::-1]
        x = skips[0]

        outputs =  []
        for idx, (conv_block, up) in enumerate(zip(self.conv_layers, self.upsample_blocks)) :
            x = up(x)

            x = torch.cat([x, skips[idx+1]], 1)
            x = conv_block(x)

            outputs.append(x)

        return outputs


def _aggregate_multi_scale_features(features):
    """
    Get several features with differetnt temporal resolutions.
    [32, 64, 128, 256]
    """
    batch, filters, time = features[-1].shape


    fixed_features = []
    for x in features[:-1]:
        x_tmp = F.interpolate(x, size=time, mode='linear',align_corners=False )
        fixed_features.append(x_tmp)

    fixed_features.append(features[-1])
    res = torch.cat(fixed_features, 1)
    return res


class HVATNetv2(nn.Module):
    def __init__(self,
                 n_electrodes=30, n_channels_out=21,
                 n_res_blocks = 2, n_blocks_per_layer = 3,
                 n_filters=64,kernel_size=3,
                 strides=(4, 4, 4),
                 dilation=1):

        super(HVATNetv2, self).__init__()

        # configuration of model
        self.n_inp_features = n_electrodes
        self.n_channels_out = n_channels_out
        self.model_depth = len(strides)

        # change number of features to custom one
        self.spatial_reduce = nn.Conv1d(n_electrodes, n_filters, kernel_size=1, padding='same')
        
        denoise_blocks = nn.ModuleList([SimpleResBlock(n_filters, kernel_size) for i in range(n_res_blocks)])
        self.denoiser = nn.Sequential(*denoise_blocks)

        self.encoder = AdvancedEncoder(n_blocks_per_layer=n_blocks_per_layer,
                                       n_filters=n_filters, kernel_size=kernel_size,
                                       dilation=dilation, strides = strides)

        self.mapper = nn.Sequential(nn.Conv1d(n_filters, n_filters, kernel_size, padding='same'),
                                    nn.GELU())

        self.decoder = AdvancedDecoder(n_blocks_per_layer=n_blocks_per_layer,
                                       n_filters=n_filters, kernel_size=kernel_size,
                                       dilation=dilation, strides = strides[::-1])

        ## heads for prediction.
        # main head of prediction on 200 Hz.
        self.main_pred_head = nn.Conv1d(int(n_filters*self.model_depth), n_channels_out,
                                        kernel_size = kernel_size, padding='same')

        # additional head for each scale prediction.
        self.shallow_pred_head = nn.ModuleList([nn.Conv1d(n_filters, n_channels_out,
                                                          kernel_size= kernel_size, padding='same')
                                                for i in range(self.model_depth-1)])



    def forward(self, x):
        batch, elec, time = x.shape

        x = self.spatial_reduce(x)

        # res blocks
        x = self.denoiser(x)
        outputs = self.encoder(x)
        outputs[-1] = self.mapper(outputs[-1])
        multi_scale_outs = self.decoder(outputs)


        # aggregation(concatenation) + pred to proper out channels
        main_feature = _aggregate_multi_scale_features(multi_scale_outs)
        main_pred = self.main_pred_head(main_feature)


        # additional transformetaiton for multi scale prediction.


        multi_scale_outs = multi_scale_outs[:-1]

        multi_scale_preds = []
        for feat, pred_head in zip(multi_scale_outs, self.shallow_pred_head) :

            pred_tmp = pred_head(feat)
            multi_scale_preds.append(pred_tmp)

        multi_scale_preds.append(main_pred)
        all_preds = multi_scale_preds
        all_preds_quats = [self._convert2quats(pred) for pred in all_preds]

        if self.training:
            return all_preds_quats
        else:
            return all_preds_quats[-1]

    def _convert2quats(self, x):
        batch, n_outs, time = x.shape
        x = x.reshape(batch, -1, 4, time)
        
        if self.training: 
            return x 
        else: 
            return F.normalize(x, p=2.0, dim=2)
            

#         phi_2, a1, a2, a3 = x[:, :, 0], x[:, :, 1],  x[:, :,  2], x[:, :, 3]
#         sin, cos = torch.sin(phi_2), torch.cos(phi_2)

#         qx, qy, qz = a1*sin, a2*sin, a3*sin
#         qw = cos
#         x = torch.stack([qx, qy, qz, qw], 2)
#         x = torch.clip(x, -1, 1)

        return x


    @torch.no_grad()
    def inference(self, myo, device='cpu', first_bone_is_constant=False):
        """
        Params:
            myo: is numpy array with shape (N_timestamps, 8)
        Return
            numpy array with shape (N_timestamps, N_bones, 4)
        """
        self.eval()
        self.to(device)

        x = torch.from_numpy(myo).T
        x = x.unsqueeze(0)
        x = x.to(device).float()
        
        y_pred = self(x)

        y_pred = y_pred[0].to('cpu').detach().numpy()
        y_pred = np.transpose(y_pred, (2, 0, 1))

        if first_bone_is_constant:
            y_pred[:, 0, :] = [0, 0, 0, -1]

        return y_pred








def freeze_all(model):
    for layer in model.modules():
        for param in layer.parameters():
            param.requires_grad = False
    return model

def unfreeze_layer_norms(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.LayerNorm):
            for param in layer.parameters():
                param.requires_grad = True
    return model

def unfreeze_first_layer(model):
    layers = []
    for i, child in enumerate(model.children()):
        layers.append(child)

    first_layer, encoder, decoder, last_layer = layers
    for param in first_layer.parameters():
        param.requires_grad = True

    return model






